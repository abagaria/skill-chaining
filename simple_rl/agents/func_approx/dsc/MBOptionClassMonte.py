import cv2
import ipdb
import torch
import random
import itertools
import numpy as np
from copy import deepcopy
from collections import deque
from scipy.spatial import distance
from thundersvm import OneClassSVM, SVC

from simple_rl.agents.func_approx.rainbow.RainbowAgentClass import RainbowAgent
from simple_rl.agents.func_approx.ppo.PPOAgentClass import PPOAgent
from simple_rl.agents.func_approx.dqn.DQNAgentClass import DQNAgent
from simple_rl.tasks.gym.GymStateClass import GymState as State
from simple_rl.tasks.gym.wrappers import LazyFrames


class ModelBasedOption(object):
    def __init__(self, *, name, parent, mdp, buffer_length, global_init,
                 gestation_period, timeout, max_steps, device, global_value_learner,
                 option_idx, gamma, target_salient_event=None,
                 preprocessing=None, solver_type="ppo"):
        self.mdp = mdp
        self.name = name
        self.gamma = gamma
        self.parent = parent
        self.device = device
        self.timeout = timeout
        self.max_steps = max_steps
        self.solver_type = solver_type
        self.global_init = global_init
        self.buffer_length = buffer_length
        self.preprocessing = preprocessing
        self.target_salient_event = target_salient_event
        self.global_value_learner = global_value_learner

        # TODO
        self.overall_mdp = mdp
        self.seed = 0
        self.option_idx = option_idx

        self.num_goal_hits = 0
        self.num_executions = 0
        self.gestation_period = gestation_period

        self.effect_set = deque(maxlen=100)
        self.positive_examples = deque(maxlen=20)
        self.negative_examples = deque(maxlen=60)
        self.optimistic_classifier = None
        self.pessimistic_classifier = None

        print(f"Using model-free controller for {name}")

        self.children = []
        self.success_curve = []

        self.global_value_learner = global_value_learner
        self.solver = self._get_model_free_solver()

        print(f"Created option {self.name} with option_idx={self.option_idx}")

    # ------------------------------------------------------------
    # Initialization Methods
    # ------------------------------------------------------------

    def _get_model_free_solver(self):
        if not self.global_init:
            return self.global_value_learner

        assert self.solver_type in ("rainbow", "dqn", "ppo"), self.solver_type
        
        if self.solver_type == "dqn":
            return DQNAgent(state_size=self.mdp.state_space_size(),
                            action_size=self.mdp.action_space_size(),
                            device=self.device,
                            goal_conditioning=True,
                            pixel_observation=True,
                            name=f"DQN-Agent-{self.option_idx}")
        
        if self.solver_type == "ppo":
            return PPOAgent(obs_n_channels=5,  # 4 for the state and 1 for the goal
                            n_actions=self.mdp.action_space_size(),
                            device_id=0)

        return RainbowAgent(obs_size=self.mdp.state_space_size(),
                            n_actions=self.mdp.action_space_size(),  
                            random_action_func=self.mdp.sample_random_action,
                            pixel_observation=True,
                            goal_conditioning=True,
                            device_id=0)

    # ------------------------------------------------------------
    # Learning Phase Methods
    # ------------------------------------------------------------

    def get_training_phase(self):
        if self.num_goal_hits < self.gestation_period:
            return "gestation"
        return "initiation_done"

    def is_init_true(self, state):
        if self.global_init or self.get_training_phase() == "gestation":
            return True
        
        features = self.get_features_for_initiation_classifier(state)
        return self.optimistic_classifier.predict([features])[0] == 1 or self.pessimistic_is_init_true(state)

    def is_term_true(self, state):
        if self.parent is None:
            return self.mdp.is_goal_state(state)

        return self.parent.pessimistic_is_init_true(state) and not self.mdp.is_dead(state.ram) and not self.mdp.falling(state)

    def pessimistic_is_init_true(self, state):
        if self.global_init or self.get_training_phase() == "gestation":
            return True

        features = self.get_features_for_initiation_classifier(state)
        return self.pessimistic_classifier.predict([features])[0] == 1

    # ------------------------------------------------------------
    # Control Loop Methods
    # ------------------------------------------------------------

    def _get_epsilon(self):
        # TODO find some better method
        return 0.1

    def act(self, state, train_mode, goal):
        """ Goal-conditioned action selection. """

        if goal is None:  # Assuming that the agent will internally also do exploration
            return self.mdp.sample_random_action()

        assert isinstance(self.solver, (DQNAgent, RainbowAgent, PPOAgent)), f"{type(self.solver)}"
        augmented_state = self.get_augmented_state(state, goal)
        return self.solver.act(augmented_state)

    def rollout(self, step_number, eval_mode=False):
        """ Main option control loop. """

        start_state = deepcopy(self.mdp.cur_state)
        assert self.is_init_true(start_state)

        num_steps = 0
        total_reward = 0
        visited_states = []
        option_transitions = []

        state = deepcopy(self.mdp.cur_state)
        goal = self.get_goal_for_rollout()
        goal_position = goal.get_position() if goal is not None else None

        print(f"[Step: {step_number}] Rolling out {self.name} from {self.mdp.get_player_position()} -> {goal_position}")

        self.num_executions += 1

        while not self.is_at_local_goal(state, goal) and step_number < self.max_steps and num_steps < self.timeout and not state.is_terminal():

            # Control
            action = self.act(state, not eval_mode, goal=goal)
            reward, next_state = self.mdp.execute_agent_action(action)

            # Update PPO
            if self.solver_type == "ppo" and goal is not None:
                self.update_ppo(next_state, goal)
            
            # Logging
            num_steps += 1
            step_number += 1
            total_reward += reward
            visited_states.append(state)
            option_transitions.append((state, action, reward, next_state))
            state = deepcopy(self.mdp.cur_state)

        visited_states.append(state)
        self.success_curve.append(self.is_term_true(state))

        if self.is_term_true(state):
            print(f"{self.name} reached termination condition at position {state.get_position()}")
            self.num_goal_hits += 1
            self.effect_set.append(state)

        self.derive_positive_and_negative_examples(visited_states)

        if not eval_mode and goal is not None and self.solver_type != "ppo":
            self.update_value_function(option_transitions, reached_goal=state, pursued_goal=goal)

        # Always be refining your initiation classifier
        if not self.global_init and not eval_mode and self.get_training_phase() != "gestation":
            self.fit_initiation_classifier()

        return option_transitions, total_reward

    # ------------------------------------------------------------
    # Experience Replay
    # ------------------------------------------------------------

    def update_ppo(self, next_state, goal):
        assert isinstance(next_state, State), f"{type(next_state)}"
        assert isinstance(goal, State), f"{type(goal)}"

        augmented_next_obs = self.get_augmented_state(next_state, goal)
        done = self.is_at_local_goal(next_state, goal)

        reward_func = self.overall_mdp.sparse_gc_reward_function
        reward, _ = reward_func(next_state, goal, info={"root_option": self.parent is None})

        self.solver.agent.observe(augmented_next_obs, reward, done, reset=False)

    def update_value_function(self, option_transitions, reached_goal, pursued_goal):
        """ Update the goal-conditioned option value function. """

        self.experience_replay(option_transitions, pursued_goal)
        self.experience_replay(option_transitions, reached_goal)

    def experience_replay(self, trajectory, goal_state):
        assert isinstance(goal_state, State), f"{type(goal_state)}"

        for state, action, reward, next_state in trajectory:
            augmented_state = self.get_augmented_state(state, goal=goal_state)
            augmented_next_state = self.get_augmented_state(next_state, goal=goal_state)
            done = self.is_at_local_goal(next_state, goal_state)

            reward_func = self.overall_mdp.sparse_gc_reward_function
            reward, _ = reward_func(next_state, goal_state, info={})

            self.solver.step(augmented_state, action, reward, augmented_next_state, done)
    
    def get_features_for_initiation_classifier(self, state):
        def downsample(image):
            return cv2.resize(image, (6, 6), interpolation=cv2.INTER_AREA)

        if self.preprocessing == "None":
            return state.features()[-1,:,:].flatten()
        elif self.preprocessing == "go-explore": # TODO scale range from 0-255 to 0-8?
            frame = state.features()[-1,10:,:]
            downsampled = downsample(frame)
            return downsampled.flatten()
        elif self.preprocessing == "go-explore-seq":
            frames = state.features()
            assert frames.shape[0] == 4, f"{frames.shape}"
            downsampled_frames = np.array([downsample(frame).flatten() for frame in frames])
            return downsampled_frames.flatten()
        elif self.preprocessing == "position":
            return state if isinstance(state, np.ndarray) else state.get_position()
        elif self.preprocessing == "dqn":  # TODO: Write this function for Rainbow
            return self.solver.feature_extract(state.features())[0]
        else:
            raise RuntimeError(f"{self.preprocessing} not implemented!")

    # ------------------------------------------------------------
    # Goal Conditioning
    # ------------------------------------------------------------

    def extract_goal_dimensions(self, goal):
        goal_features = goal if isinstance(goal, LazyFrames) else goal.features()
        
        assert isinstance(goal_features, LazyFrames), f"{type(goal_features)}"
        goal_frame = goal_features.get_frame(-1)
        return goal_frame

    def get_augmented_state(self, state, goal):
        assert isinstance(state, State), f"{type(state)}"
        assert isinstance(goal, State), f"{type(goal)}"

        obs_frames = state.features()._frames
        goal_image = self.extract_goal_dimensions(goal)
        augmented_frames = obs_frames + [goal_image]

        assert len(augmented_frames) == 5, len(augmented_frames)
        assert len(obs_frames) == 4, len(obs_frames)

        return LazyFrames(frames=augmented_frames, stack_axis=0)

    def is_at_local_goal(self, state, goal):
        """ Goal-conditioned termination condition. """

        reached_goal = self.mdp.sparse_gc_reward_function(state, goal, {'root_option': self.parent is None})[1] if goal is not None else True
        reached_term = self.is_term_true(state) or state.is_terminal()
        return reached_goal and reached_term

    def get_first_state_in_classifier(self, trajectory):
        """ Extract the first state in the trajectory that is inside the initiation classifier. """

        for state in trajectory:
            if self.pessimistic_is_init_true(state):
                return state
        return None

    def sample_from_initiation_region(self):
        """ Sample from the pessimistic initiation classifier. """
        num_tries = 0
        sampled_state = None
        while sampled_state is None and num_tries < 200:
            num_tries = num_tries + 1
            sampled_trajectory_idx = random.choice(range(len(self.positive_examples)))
            sampled_trajectory = self.positive_examples[sampled_trajectory_idx]
            sampled_state = self.get_first_state_in_classifier(sampled_trajectory)
        return sampled_state

    def get_goal_for_rollout(self):
        if self.parent is not None:
            sampled_goal = self.parent.sample_from_initiation_region()
            if sampled_goal is not None:
                return sampled_goal
        if self.num_goal_hits > 0:
            return random.choice(self.effect_set)
        assert self.num_goal_hits == 0, self.num_goal_hits
        return None

    # ------------------------------------------------------------
    # Learning Initiation Classifiers
    # ------------------------------------------------------------

    def derive_positive_and_negative_examples(self, visited_states):
        start_state = visited_states[0]
        final_state = visited_states[-1]

        if self.is_term_true(final_state):
            positive_states = [start_state] + visited_states[-self.buffer_length:]
            self.positive_examples.append(positive_states)
        else:
            negative_examples = [start_state] + visited_states[-self.buffer_length:]
            self.negative_examples.append(negative_examples)

    def fit_initiation_classifier(self):
        if len(self.negative_examples) > 0 and len(self.positive_examples) > 0:
            self.train_two_class_classifier()
        elif len(self.positive_examples) > 0:
            self.train_one_class_svm()

    def construct_feature_matrix(self, examples):
        states = list(itertools.chain.from_iterable(examples))
        positions = [self.get_features_for_initiation_classifier(state) for state in states]
        return np.array(positions)

    def get_gamma(self, feature_matrix):
        if self.gamma == "scale":
            num_features = feature_matrix.shape[1]
            return 1.0 / (num_features * feature_matrix.var())
        return self.gamma

    def train_one_class_svm(self, nu=0.1):
        positive_feature_matrix = self.construct_feature_matrix(self.positive_examples)
        gamma = self.get_gamma(positive_feature_matrix)
        
        self.pessimistic_classifier = OneClassSVM(gamma=gamma, nu=nu)
        self.pessimistic_classifier.fit(positive_feature_matrix)

        self.optimistic_classifier = OneClassSVM(gamma=gamma, nu=nu/10.)
        self.optimistic_classifier.fit(positive_feature_matrix)

    def train_two_class_classifier(self, nu=0.1):
        positive_feature_matrix = self.construct_feature_matrix(self.positive_examples)
        negative_feature_matrix = self.construct_feature_matrix(self.negative_examples)
        positive_labels = [1] * positive_feature_matrix.shape[0]
        negative_labels = [0] * negative_feature_matrix.shape[0]

        X = np.concatenate((positive_feature_matrix, negative_feature_matrix))
        Y = np.concatenate((positive_labels, negative_labels))
        
        gamma = self.get_gamma(X)

        if negative_feature_matrix.shape[0] >= 10:
            kwargs = {"kernel": "rbf", "gamma": gamma, "class_weight": "balanced"}
        else:
            kwargs = {"kernel": "rbf", "gamma": gamma}

        self.optimistic_classifier = SVC(**kwargs)
        self.optimistic_classifier.fit(X, Y)

        training_predictions = self.optimistic_classifier.predict(X) == 1
        correct_training_predictions = np.logical_and(training_predictions, Y)
        self.positive_training_examples = X[correct_training_predictions == 1]

        if self.positive_training_examples.shape[0] > 0:
            gamma = self.get_gamma(self.positive_training_examples)
            self.pessimistic_classifier = OneClassSVM(kernel="rbf", gamma=gamma, nu=nu)
            self.pessimistic_classifier.fit(self.positive_training_examples)
    
    # ------------------------------------------------------------
    # Convenience functions
    # ------------------------------------------------------------

    def get_option_success_rate(self):
        """
        TODO: implement success rate at test time as well (Jason)
        """
        if self.num_executions > 0:
            return self.num_goal_hits / self.num_executions
        return 1.

    def get_success_rate(self):
        if len(self.success_curve) == 0:
            return 0.
        return np.mean(self.success_curve)

    def __str__(self):
        return self.name

    def __repr__(self):
        return str(self)

    def __eq__(self, other):
        if isinstance(other, ModelBasedOption):
            return self.name == other.name
        return False
