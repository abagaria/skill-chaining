import cv2
import ipdb
import torch
import random
import itertools
import numpy as np
from copy import deepcopy
from collections import deque
from scipy.spatial import distance
# from thundersvm import OneClassSVM, SVC
from sklearn.svm import OneClassSVM, SVC

from simple_rl.mdp.StateClass import State
from simple_rl.tasks.gym.wrappers import LazyFrames
from simple_rl.agents.func_approx.dsc.dynamics.mpc import MPC
from simple_rl.agents.func_approx.td3.TD3AgentClass import TD3
from simple_rl.agents.func_approx.ppo.PPOAgentClass import PPOAgent


class ModelBasedOption(object):
    def __init__(self, *, name, parent, mdp, global_solver, global_value_learner, buffer_length, global_init,
                 gestation_period, timeout, max_steps, device, use_vf, use_model, dense_reward,
                 option_idx, chain_id, preprocessing="position", lr_c=3e-4, lr_a=3e-4, max_num_children=2,
                 solver_type="ppo", target_salient_event=None, init_salient_event=None, path_to_model="", multithread_mpc=False):
        self.mdp = mdp
        self.name = name
        self.lr_c = lr_c
        self.lr_a = lr_a
        self.parent = parent
        self.device = device
        self.use_vf = use_vf
        self.solver_type = solver_type
        self.preprocessing = preprocessing
        self.global_solver = global_solver
        self.use_global_vf = use_vf
        self.timeout = timeout
        self.use_model = use_model
        self.max_steps = max_steps
        self.global_init = global_init
        self.dense_reward = dense_reward
        self.buffer_length = buffer_length
        self.max_num_children = max_num_children
        self.init_salient_event = init_salient_event
        self.target_salient_event = target_salient_event
        self.multithread_mpc = multithread_mpc

        # TODO
        self.overall_mdp = mdp
        self.seed = 0
        self.chain_id = chain_id
        self.option_idx = option_idx

        self.num_goal_hits = 0
        self.num_executions = 0
        self.gestation_period = gestation_period

        self.positive_examples = deque(maxlen=100)
        self.negative_examples = deque(maxlen=100)

        self.optimistic_classifier = None
        self.pessimistic_classifier = None
        self.last_episode_term_triggered = -1
        self.started_at_target_salient_event = False

        self.expansion_classifier = None

        lr_a = lr_c = 1e-5 if self.use_model else 3e-4

        # In the model-free setting, the output norm doesn't seem to work
        # But it seems to stabilize off policy value function learning
        # Therefore, only use output norm if we are using MPC for action selection
        self.use_output_norm = self.use_model
        self.value_learner = None

        if use_vf and (not self.use_global_vf or global_init):
            print(f"Initializing VF with lra={lr_a}, lrc={lr_c} and output-norm={self.use_output_norm}")
            self.value_learner = self._get_model_free_solver()

        self.global_value_learner = global_value_learner if not self.global_init else None  # type: TD3

        if use_model:
            print(f"Using model-based controller for {name}")
            self.solver = self._get_model_based_solver()
        else:
            print(f"Using model-free controller for {name}")
            self.solver = self._get_model_free_solver()

        self.children = []
        self.success_curve = []
        self.effect_set = deque(maxlen=100)

        if path_to_model:
            print(f"Loading model from {path_to_model} for {self.name}")
            self.solver.load_model(path_to_model)

        if self.use_vf and not self.use_global_vf and self.parent is not None:
            self.initialize_value_function_with_global_value_function()

        print(f"Created model-based option {self.name} with option_idx={self.option_idx}")

        self.is_last_option = False

    def _get_model_based_solver(self):
        assert self.use_model

        if self.global_init:
            return MPC(mdp=self.mdp,
                       state_size=self.mdp.state_space_size(),
                       action_size=self.mdp.action_space_size(),
                       dense_reward=self.dense_reward,
                       device=self.device,
                       multithread=self.multithread_mpc)

        assert self.global_solver is not None
        return self.global_solver

    def _get_model_free_solver(self):
        if not self.global_init:
            return self.global_value_learner

        assert self.solver_type in ("td3", "ppo"), self.solver_type

        if self.solver_type == "td3":
            return TD3(state_dim=self.mdp.state_space_size()+self.overall_mdp.goal_space_size(),
                                    action_dim=self.mdp.action_space_size(),
                                    max_action=1.,
                                    name=f"{self.name}-td3-agent",
                                    device=self.device,
                                    lr_c=self.lr_c, lr_a=self.lr_a,
                                    use_output_normalization=self.use_output_norm)
        
        if self.solver_type == "ppo":
            return PPOAgent(obs_n_channels=5,  # 4 for the state and 1 for the goal
                            n_actions=self.mdp.action_space_size(),
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
        if self.global_init:
            return True

        if self.parent is None:
            return self.target_salient_event(state)

        return self.parent.pessimistic_is_init_true(state)

    def pessimistic_is_init_true(self, state):
        if self.global_init or self.get_training_phase() == "gestation":
            return True

        assert isinstance(state, State), f"{type(state)}"
        features = self.get_features_for_initiation_classifier(state)
        expansion_clf_decision = self.expansion_classifier(state) if self.expansion_classifier else False
        return self.pessimistic_classifier.predict([features])[0] == 1 or expansion_clf_decision

    def batched_is_init_true(self, state_matrix):  # TODO: Incorporate expansion and pessimistic classifiers
        if self.global_init or self.get_training_phase() == "gestation":
            return np.ones((state_matrix.shape[0]))

        position_matrix = self.mdp.batched_get_position(state_matrix)
        return self.optimistic_classifier.predict(position_matrix) == 1

    def pessimistic_batched_is_init_true(self, state_matrix):
        if self.global_init or self.get_training_phase() == "gestation":
            return np.ones((state_matrix.shape[0]))

        position_matrix = self.mdp.batched_get_position(state_matrix)
        defaults = np.zeros((position_matrix.shape[0],))
        expansion_clf_decision = self.expansion_classifier(state_matrix) if self.expansion_classifier else defaults
        return np.logical_or(self.pessimistic_classifier.predict(position_matrix) == 1, expansion_clf_decision)

    def batched_is_term_true(self, state_matrix):
        if self.parent is None:
            return self.target_salient_event(state_matrix)

        position_matrix = self.mdp.batched_get_position(state_matrix)
        return self.parent.pessimistic_classifier.predict(position_matrix) == 1

    def is_in_effect_set(self, state):
        if not self.is_term_true(state):
            return False

        return self.is_close_to_effect_set(state)

    def is_close_to_effect_set(self, state):

        states = np.repeat([self.mdp.get_position(state)], repeats=len(self.effect_set), axis=0)
        goals = np.vstack([self.mdp.get_position(goal) for goal in self.effect_set])

        assert states.shape[0] == goals.shape[0] == len(self.effect_set), f"{states.shape, goals.shape}"

        dones = self.mdp.batched_dense_gc_reward_function(states, goals)[1]
        return dones.any()

    def expand_initiation_classifier(self, classifier):  # TODO: add support
        print(f"Expanding {self.name}'s pessimistic classifier to include {classifier}")
        self.expansion_classifier = classifier

    def is_at_local_goal(self, state, goal):
        """ Goal-conditioned termination condition. """

        info = {}

        if self.target_salient_event is not None:
            targets_key = self.target_salient_event.target_state is None # TODO: This event no longer has a target_state
            info = {'targets_key': targets_key}

        reached_goal = self.mdp.sparse_gc_reward_function(state, goal, info)[1]
        reached_term = self.is_term_true(state) or state.is_terminal()
        reached_key = 'targets_key' in info and info['targets_key'] and reached_goal

        return reached_goal and (reached_term or reached_key)

    # ------------------------------------------------------------
    # Control Loop Methods
    # ------------------------------------------------------------

    def _get_epsilon(self):
        if self.use_model:
            return 0.1
        if not self.dense_reward and self.num_goal_hits <= 3:
            return 0.8
        return 0.2

    def act(self, state, goal):
        """ Epsilon-greedy action selection. """

        if goal is None:
            return self.mdp.sample_random_action()

        if self.use_model:
            assert isinstance(self.solver, MPC), f"{type(self.solver)}"
            vf = self.value_function if self.use_vf else None
            return self.solver.act(state, goal, vf=vf)

        augmented_state = self.get_augmented_state(state, goal)
        return self.solver.act(augmented_state)

    def update_model(self, state, action, reward, next_state):
        """ Learning update for option model/actor/critic. """

        self.solver.step(state.features(), action, reward, next_state.features(), next_state.is_terminal())

    def get_goal_for_rollout(self):
        """ Sample goal to pursue for option rollout. """

        if self.parent is not None:
            sampled_goal = self.parent.sample_from_initiation_region()
            if sampled_goal is not None:
                return sampled_goal
        
        if self.num_goal_hits > 0:
            return random.choice(self.effect_set)

        if self.parent is None and len(self.target_salient_event.trigger_points) > 0:
            return random.choice(self.target_salient_event.trigger_points)
        
        assert self.num_goal_hits == 0, self.num_goal_hits
        return None

    def rollout(self, episode, step_number, rollout_goal=None, eval_mode=False):
        """ Main option control loop. """

        start_state = deepcopy(self.mdp.cur_state)
        assert self.is_init_true(start_state)
        if self.global_init: assert rollout_goal is not None

        num_steps = 0
        total_reward = 0
        visited_states = []
        option_transitions = []

        state = deepcopy(self.mdp.cur_state)
        goal = self.get_goal_for_rollout() if rollout_goal is None else rollout_goal

        print(f"[Step: {step_number}] Rolling out {self.name}, from {state.position} targeting {goal.position}")

        self.num_executions += 1

        while not self.is_at_local_goal(state, goal) and step_number < self.max_steps and num_steps < self.timeout and not state.is_terminal():

            # Control
            action = self.act(state, goal)
            reward, next_state = self.mdp.execute_agent_action(action)

            # Update Model
            if self.use_model:
                self.update_model(state, action, reward, next_state)

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
        reached_parent = self.is_term_true(state)
        self.success_curve.append(reached_parent)

        if reached_parent and not self.global_init:
            print(f"****** {self.name} execution successful ******")
            self.effect_set.append(state)

        if reached_parent and self.last_episode_term_triggered != episode:
            self.num_goal_hits += 1
            self.last_episode_term_triggered = episode

        if self.target_salient_event is not None and self.target_salient_event(state):
            self.target_salient_event.trigger_points.append(state)

        if self.use_vf and not eval_mode and self.solver_type != "ppo":
            self.update_value_function(option_transitions,
                                    pursued_goal=goal,
                                    reached_goal=self.extract_goal_dimensions(state))

        self.derive_positive_and_negative_examples(visited_states)

        # Always be refining your initiation classifier
        if not self.global_init and not eval_mode:
            self.fit_initiation_classifier()

        return option_transitions, total_reward

    # ------------------------------------------------------------
    # Hindsight Experience Replay
    # ------------------------------------------------------------

    def update_ppo(self, next_state, goal):
        assert isinstance(next_state, State), f"{type(next_state)}"
        assert isinstance(goal, State), f"{type(goal)}"

        augmented_next_obs = self.get_augmented_state(next_state, goal)
        done = self.is_at_local_goal(next_state, goal)

        info = {}
        if self.target_salient_event is not None:
            targets_key = self.target_salient_event.target_state is None
            info = {"targets_key": targets_key}

        reward_func = self.overall_mdp.sparse_gc_reward_function
        reward, _ = reward_func(next_state, goal, info=info)

        self.solver.agent.observe(augmented_next_obs, reward, done, reset=False)


    def update_value_function(self, option_transitions, reached_goal, pursued_goal):
        """ Update the goal-conditioned option value function. """

        self.experience_replay(option_transitions, pursued_goal)
        self.experience_replay(option_transitions, reached_goal)

    def initialize_value_function_with_global_value_function(self):
        self.value_learner.actor.load_state_dict(self.global_value_learner.actor.state_dict())
        self.value_learner.critic.load_state_dict(self.global_value_learner.critic.state_dict())
        self.value_learner.target_actor.load_state_dict(self.global_value_learner.target_actor.state_dict())
        self.value_learner.target_critic.load_state_dict(self.global_value_learner.target_critic.state_dict())

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


    def experience_replay(self, trajectory, goal_state):
        for state, action, reward, next_state in trajectory:
            augmented_state = self.get_augmented_state(state, goal=goal_state)
            augmented_next_state = self.get_augmented_state(next_state, goal=goal_state)
            done = self.is_at_local_goal(next_state, goal_state)

            reward_func = self.overall_mdp.dense_gc_reward_function if self.dense_reward \
                else self.overall_mdp.sparse_gc_reward_function
            reward, global_done = reward_func(next_state, goal_state, info={})

            if not self.use_global_vf or self.global_init:
                self.value_learner.step(augmented_state, action, reward, augmented_next_state, done)

            # Off-policy updates to the global option value function
            if not self.global_init:
                assert self.global_value_learner is not None
                self.global_value_learner.step(augmented_state, action, reward, augmented_next_state, global_done)

    def value_function(self, states, goals):
        assert isinstance(states, np.ndarray)
        assert isinstance(goals, np.ndarray)

        if len(states.shape) == 1:
            states = states[None, ...]
        if len(goals.shape) == 1:
            goals = goals[None, ...]

        goal_positions = goals[:, :self.mdp.goal_space_size()]
        augmented_states = np.concatenate((states, goal_positions), axis=1)
        augmented_states = torch.as_tensor(augmented_states).float().to(self.device)

        if self.use_global_vf and not self.global_init:
            values = self.global_value_learner.get_values(augmented_states)
        else:
            values = self.value_learner.get_values(augmented_states)

        return values

    # ------------------------------------------------------------
    # Learning Initiation Classifiers
    # ------------------------------------------------------------

    @staticmethod
    def get_gamma(gamma, feature_matrix):
        if gamma == "scale":
            num_features = feature_matrix.shape[1]
            return 1.0 / (num_features * feature_matrix.var())
        return gamma

    @staticmethod
    def is_between(x1, x2, x3):
        """ is x3 in between x1 and x2? """
        d0 = x2 - x1
        d0m = np.linalg.norm(d0)
        v = d0 / d0m
        d1 = x3 - x1
        r = v.dot(d1)
        return 0 <= r <= d0m

    def is_valid_init_data(self, state_buffer):

        def get_points_in_between(init_point, final_point, positions):
            return [position for position in positions if self.is_between(init_point, final_point, position)]

        # Use the data if it could complete the chain
        if self.init_salient_event is not None:
            if any([self.is_init_true(s) for s in self.init_salient_event.trigger_points]):
                return True

        length_condition = len(state_buffer) >= self.buffer_length // 10

        if not length_condition:
            return False

        if "monte" in self.mdp.env_name.lower():
            return True

        position_buffer = [self.overall_mdp.get_position(state) for state in state_buffer]
        points_in_between = get_points_in_between(self.init_salient_event.get_target_position(),
                                                  self.target_salient_event.get_target_position(),
                                                  position_buffer)
        in_between_condition = len(points_in_between) >= self.buffer_length // 10
        return in_between_condition

    def get_first_state_in_classifier(self, trajectory, classifier_type="pessimistic"):
        """ Extract the first state in the trajectory that is inside the initiation classifier. """

        assert classifier_type in ("pessimistic", "optimistic"), classifier_type
        classifier = self.pessimistic_is_init_true if classifier_type == "pessimistic" else self.is_init_true
        for state in trajectory:
            if classifier(state):
                return state
        return None

    def sample_from_initiation_region(self):  # TODO
        return self.sample_from_initiation_region_fast()

    def sample_from_initiation_region_fast(self):
        """ Sample from the pessimistic initiation classifier. """
        num_tries = 0
        sampled_state = None
        while sampled_state is None and num_tries < 200:
            num_tries = num_tries + 1
            sampled_trajectory_idx = random.choice(range(len(self.positive_examples)))
            sampled_trajectory = self.positive_examples[sampled_trajectory_idx]
            sampled_state = self.get_first_state_in_classifier(sampled_trajectory)
        return sampled_state

    def sample_from_initiation_region_fast_and_epsilon(self):
        """ Sample from the pessimistic initiation classifier. """
        def compile_states(s):
            pos0 = self.mdp.get_position(s)
            pos1 = np.copy(pos0)
            pos1[0] -= self.target_salient_event.tolerance
            pos2 = np.copy(pos0)
            pos2[0] += self.target_salient_event.tolerance
            pos3 = np.copy(pos0)
            pos3[1] -= self.target_salient_event.tolerance
            pos4 = np.copy(pos0)
            pos4[1] += self.target_salient_event.tolerance
            return pos0, pos1, pos2, pos3, pos4

        idxs = [i for i in range(len(self.positive_examples))]
        random.shuffle(idxs)

        for idx in idxs:
            sampled_trajectory = self.positive_examples[idx]
            states = []
            for s in sampled_trajectory:
                states.extend(compile_states(s))

            position_matrix = np.vstack(states)
            # optimistic_predictions = self.optimistic_classifier.predict(position_matrix) == 1
            # pessimistic_predictions = self.pessimistic_classifier.predict(position_matrix) == 1
            # predictions = np.logical_or(optimistic_predictions, pessimistic_predictions)
            predictions = self.pessimistic_classifier.predict(position_matrix) == 1
            predictions = np.reshape(predictions, (-1, 5))
            valid = np.all(predictions, axis=1)
            indices = np.argwhere(valid == True)
            if len(indices) > 0:
                return sampled_trajectory[indices[0][0]]

        print(f"[{self.name}] Could not find epsilon-safe sample, using random sample.")
        return self.sample_from_initiation_region_fast()

    def derive_positive_and_negative_examples(self, visited_states):
        start_state = visited_states[0]
        final_state = visited_states[-1]

        if self.is_term_true(final_state):
            positive_states = [start_state] + visited_states[-self.buffer_length:]
            self.positive_examples.append(positive_states)
        else:
            negative_examples = [start_state]
            self.negative_examples.append(negative_examples)

    def fit_initiation_classifier(self):
        if len(self.negative_examples) > 0 and len(self.positive_examples) > 0:
            self.train_two_class_classifier()
        elif len(self.positive_examples) > 0:
            self.train_one_class_svm()

    def construct_feature_matrix(self, examples):
        states = list(itertools.chain.from_iterable(examples))
        positions = [self.mdp.get_position(state) for state in states]
        return np.array(positions)

    def train_one_class_svm(self, nu=0.1):  # TODO: Implement gamma="auto" for thundersvm
        positive_feature_matrix = self.construct_feature_matrix(self.positive_examples)
        self.pessimistic_classifier = OneClassSVM(kernel="rbf", nu=nu)
        self.pessimistic_classifier.fit(positive_feature_matrix)

        self.optimistic_classifier = OneClassSVM(kernel="rbf", nu=nu/10.)
        self.optimistic_classifier.fit(positive_feature_matrix)

    def train_two_class_classifier(self, nu=0.1):
        positive_feature_matrix = self.construct_feature_matrix(self.positive_examples)
        negative_feature_matrix = self.construct_feature_matrix(self.negative_examples)
        positive_labels = [1] * positive_feature_matrix.shape[0]
        negative_labels = [0] * negative_feature_matrix.shape[0]

        X = np.concatenate((positive_feature_matrix, negative_feature_matrix))
        Y = np.concatenate((positive_labels, negative_labels))

        gamma = self.get_gamma("scale", X)

        if negative_feature_matrix.shape[0] >= 10:  # TODO: Implement gamma="auto" for thundersvm
            kwargs = {"kernel": "rbf", "gamma": gamma, "class_weight": "balanced"}
        else:
            kwargs = {"kernel": "rbf", "gamma": gamma}

        self.optimistic_classifier = SVC(**kwargs)
        self.optimistic_classifier.fit(X, Y)

        training_predictions = self.optimistic_classifier.predict(X)
        positive_training_examples = X[training_predictions == 1]

        if positive_training_examples.shape[0] > 0:
            self.pessimistic_classifier = OneClassSVM(kernel="rbf", nu=nu, gamma=gamma)
            self.pessimistic_classifier.fit(positive_training_examples)

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

    def is_eligible_for_off_policy_triggers(self):
        eligible_phase = False  # self.get_training_phase() != "initiation_done"
        return eligible_phase and not self.started_at_target_salient_event

    def trigger_termination_condition_off_policy(self, trajectory, episode):
        trajectory_so_far = []
        states_so_far = []

        if not self.is_eligible_for_off_policy_triggers():
            return False

        for state, action, reward, next_state in trajectory:
            trajectory_so_far.append((state, action, reward, next_state))
            states_so_far.append(state)

            if self.is_term_true(next_state) and episode != self.last_episode_term_triggered and \
                    self.is_valid_init_data(states_so_far):

                print(f"Triggered termination condition for {self}")
                self.effect_set.append(next_state)
                self.last_episode_term_triggered = episode

                if self.parent is None and self.is_term_true(next_state):
                    print(f"[{self}] Adding {next_state.position} to {self.target_salient_event}'s trigger points")
                    self.target_salient_event.trigger_points.append(next_state)

                self.num_goal_hits += 1
                states_so_far.append(next_state)  # We want the terminal state to be part of the initiation classifier
                self.derive_positive_and_negative_examples(visited_states=states_so_far)
                self.fit_initiation_classifier()
                return self.get_training_phase() != "gestation"

        return False

    # ------------------------------------------------------------
    # Distance functions
    # ------------------------------------------------------------

    def get_effective_effect_set(self):
        """ Return the subset of the effect set still in the termination region. """

        positive_goal_states = [state for state in self.effect_set if self.is_term_true(state)]
        return positive_goal_states

    def get_states_inside_pessimistic_classifier_region(self):
        point_array = self.construct_feature_matrix(self.positive_examples)
        point_array_predictions = self.pessimistic_classifier.predict(point_array)
        positive_point_array = point_array[point_array_predictions == 1]
        return positive_point_array

    def distance_to_state(self, state, metric="euclidean"):
        """ Compute the distance between the current option and the input `state`. """

        assert metric in ("euclidean", "value"), metric
        if metric == "euclidean":
            return self._euclidean_distance_to_state(state)
        return self._value_distance_to_state(state)

    def _euclidean_distance_to_state(self, state):
        point = self.mdp.get_position(state)

        assert isinstance(point, np.ndarray)
        assert point.shape == (2,), point.shape

        positive_point_array = self.get_states_inside_pessimistic_classifier_region()

        distances = distance.cdist(point[None, :], positive_point_array)
        return np.median(distances)

    def _value_distance_to_state(self, state):
        features = state.features() if not isinstance(state, np.ndarray) else state
        goals = self.get_states_inside_pessimistic_classifier_region()

        distances = self.value_function(features, goals)
        distances[distances > 0] = 0.
        return np.median(np.abs(distances))

    # ------------------------------------------------------------
    # Convenience functions
    # ------------------------------------------------------------

    def get_option_success_rate(self):
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

    def __hash__(self):
        return hash(self.name)

    def __call__(self, state):
        return self.is_in_effect_set(state)
