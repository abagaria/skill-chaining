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

from simple_rl.agents.func_approx.dqn.DQNAgentClass import DQNAgent


class ModelBasedOption(object):
    def __init__(self, *, name, parent, mdp, buffer_length, global_init,
                 gestation_period, timeout, max_steps, device, global_value_learner,
                 option_idx, gamma, goal_conditioning, target_salient_event=None, 
                 preprocessing=None):
        self.mdp = mdp
        self.name = name
        self.gamma = gamma
        self.parent = parent
        self.device = device
        self.timeout = timeout
        self.max_steps = max_steps
        self.global_init = global_init
        self.buffer_length = buffer_length
        self.preprocessing = preprocessing
        self.goal_conditioning = goal_conditioning
        self.target_salient_event = target_salient_event
        self.global_value_learner = global_value_learner

        # TODO
        self.overall_mdp = mdp
        self.seed = 0
        self.option_idx = option_idx

        self.num_goal_hits = 0
        self.num_executions = 0
        self.gestation_period = gestation_period

        self.positive_examples = []
        self.negative_examples = []
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
        if self.goal_conditioning and not self.global_init:
            return self.global_value_learner

        return DQNAgent(state_size=self.mdp.state_space_size(),
                        action_size=self.mdp.action_space_size(),
                        device=self.device,
                        goal_conditioning=self.goal_conditioning,
                        pixel_observation=True,
                        name=f"DQN-Agent-{self.option_idx}")

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

    def act(self, state, train_mode, goal=None):
        """ Epsilon-greedy action selection. """

        if train_mode and random.random() < self._get_epsilon():
            return self.mdp.sample_random_action()

        if goal:
            augmented_state = self.get_augmented_state(state, goal)
            return self.solver.act(augmented_state, train_mode=train_mode)
        return self.solver.act(state, train_mode=train_mode)

    def rollout(self, step_number, eval_mode=False):
        """ Main option control loop. """

        start_state = deepcopy(self.mdp.cur_state)
        assert self.is_init_true(start_state)

        num_steps = 0
        total_reward = 0
        visited_states = []
        option_transitions = []

        state = deepcopy(self.mdp.cur_state)

        print(f"[Step: {step_number}] Rolling out {self.name} from {self.mdp.get_player_position()}")

        self.num_executions += 1

        while not self.is_term_true(state) and step_number < self.max_steps and num_steps < self.timeout and not state.is_terminal():

            # Control
            action = self.act(state, not eval_mode)
            reward, next_state = self.mdp.execute_agent_action(action)
            
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
            print(f"{self.name} reached termination condition!")
            self.num_goal_hits += 1

        self.derive_positive_and_negative_examples(visited_states)

        if not eval_mode:
            self.experience_replay(option_transitions)

        # Always be refining your initiation classifier
        if not self.global_init and not eval_mode and self.get_training_phase() != "gestation":
            self.fit_initiation_classifier()

        return option_transitions, total_reward

    def experience_replay(self, trajectory):
        for state, action, reward, next_state in trajectory:
            done = self.is_term_true(next_state)
            subgoal_reward = +1. if done else -1.

            self.solver.step(state.features(), action, subgoal_reward, next_state.features(), done)
            if not self.global_init:
                self.global_value_learner.step(state.features(), action, reward, next_state.features(), next_state.is_terminal())
    
    def get_features_for_initiation_classifier(self, state):
        if self.preprocessing == "None":
            return state.features()[-1,:,:].flatten()
        elif self.preprocessing == "go-explore": # TODO scale range from 0-255 to 0-8?
            frame = state.features()[-1,10:,:]
            downsampled = cv2.resize(frame, (6,6), interpolation=cv2.INTER_AREA)
            return downsampled.flatten()
        elif self.preprocessing == "position":
            return state if isinstance(state, np.ndarray) else state.get_position()
        elif self.preprocessing == "dqn":
            return self.solver.feature_extract(state.features())[0]
        else:
            raise RuntimeError(f"{self.preprocessing} not implemented!")

    # ------------------------------------------------------------
    # Hindsight Experience Replay
    # ------------------------------------------------------------

    def initialize_value_function_with_global_value_function(self):
        self.solver.policy_network.load_state_dict(self.global_value_learner.policy_network.state_dict())
        self.solver.target_network.load_state_dict(self.global_value_learner.target_network.state_dict())

    # ------------------------------------------------------------
    # Goal Conditioning
    # ------------------------------------------------------------

    def get_augmented_state(self, state, goal):
        goal_image = goal.features()[-1,:,:]
        return np.concatenate((state.features(), goal_image), axis=0)

    def is_at_local_goal(self, state, goal):
        def is_close(pos1, pos2):
            return abs(pos1[0] - pos2[0]) <= 5 and abs(pos1[1] - pos2[1]) <= 5
        state_pos = state.get_position()
        goal_pos = goal.get_position()
        return is_close(state_pos, goal_pos)

    def get_goal_for_rollout(self):
        if not self.parent:
            pass
        num_positives = len(self.parent.positive_examples)
        idx = random.randint(0, num_positives)
        return self.parent.positive_examples[idx][0]

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
