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
                 option_idx, target_salient_event=None, preprocessing=None):
        self.mdp = mdp
        self.name = name
        self.parent = parent
        self.device = device
        self.timeout = timeout
        self.max_steps = max_steps
        self.global_init = global_init
        self.buffer_length = buffer_length
        self.preprocessing = preprocessing
        self.target_salient_event = target_salient_event
        self.global_value_learner = global_value_learner

        # TODO
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
        self.effect_set = []

        self.global_value_learner = global_value_learner
        self.solver = DQNAgent(state_size=self.mdp.state_space_size(),
                               action_size=self.mdp.action_space_size(),
                               device=self.device,
                               pixel_observation=True,
                               name=f"DQN-Agent-{option_idx}")

        if self.parent is not None:
            self.initialize_value_function_with_global_value_function()

        print(f"Created model-based option {self.name} with option_idx={self.option_idx}")

        self.is_last_option = False

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
        
        if self.is_last_option and self.mdp.get_start_state_salient_event()(state):
            return True

        features = self.get_features_for_initiation_classifier(state)
        return self.optimistic_classifier.predict([features])[0] == 1 or self.pessimistic_is_init_true(state)

    def is_term_true(self, state):
        if self.parent is None:
            return self.mdp.is_goal_state(state)

        return self.parent.pessimistic_is_init_true(state)

    def pessimistic_is_init_true(self, state):
        if self.global_init or self.get_training_phase() == "gestation":
            return True

        features = self.get_features_for_initiation_classifier(state)
        return self.pessimistic_classifier.predict([features])[0] == 1

    # ------------------------------------------------------------
    # Control Loop Methods
    # ------------------------------------------------------------

    def _get_epsilon(self):
        # TODO decrease to 0
        return 0.1

    def act(self, state, train_mode):
        """ Epsilon-greedy action selection. """

        if random.random() < self._get_epsilon():
            return self.mdp.sample_random_action()

        assert isinstance(self.solver, DQNAgent), f"{type(self.solver)}"
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
            done = self.is_term_true(next_state)
            subgoal_reward = +10 if done else 0
            
            if not eval_mode:
                self.solver.step(state.features(), action, subgoal_reward, next_state.features(), done)
                if not self.global_init:
                    self.global_value_learner.step(state.features(), action, reward, next_state.features(), next_state.is_terminal())

            # Logging
            num_steps += 1
            step_number += 1
            total_reward += reward
            visited_states.append(state)
            option_transitions.append((state, action, reward, next_state))
            state = deepcopy(self.mdp.cur_state)

        visited_states.append(state)
        self.success_curve.append(self.is_term_true(state))
        self.effect_set.append(state.features())

        if self.is_term_true(state):
            print(f"{self.name} reached termination condition!")
            self.num_goal_hits += 1

        if not eval_mode:
            self.derive_positive_and_negative_examples(visited_states)

        # Always be refining your initiation classifier
        if not self.global_init and not eval_mode:
            self.fit_initiation_classifier()

        return option_transitions, total_reward

    def get_features_for_initiation_classifier(self, state):
        if self.preprocessing == None:
            return state.features()[-1,:,:].flatten()
        elif self.preprocessing == "go-explore": # TODO scale range from 0-255 to 0-8?
            frame = state.features()[-1,10:,:]
            downsampled = cv2.resize(frame, (11,8), interpolation=cv2.INTER_AREA)
            return downsampled.flatten()
        else:
            raise RuntimeError(f"{self.preprocessing} not implemented!")

    # ------------------------------------------------------------
    # Hindsight Experience Replay
    # ------------------------------------------------------------

    def initialize_value_function_with_global_value_function(self):
        self.solver.policy_network.load_state_dict(self.global_value_learner.policy_network.state_dict())
        self.solver.target_network.load_state_dict(self.global_value_learner.target_network.state_dict())

    # ------------------------------------------------------------
    # Learning Initiation Classifiers
    # ------------------------------------------------------------

    def derive_positive_and_negative_examples(self, visited_states):
        start_state = visited_states[0]
        final_state = visited_states[-1]

        if self.is_term_true(final_state):
            positive_states = visited_states[-self.buffer_length:] # [start_state] + visited_states[-self.buffer_length:]
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
        positions = [self.get_features_for_initiation_classifier(state) for state in states]
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

        if negative_feature_matrix.shape[0] >= 10:  # TODO: Implement gamma="auto" for thundersvm
            kwargs = {"kernel": "rbf", "gamma": "auto", "class_weight": "balanced"}
        else:
            kwargs = {"kernel": "rbf", "gamma": "auto"}

        self.optimistic_classifier = SVC(**kwargs)
        self.optimistic_classifier.fit(X, Y)

        training_predictions = self.optimistic_classifier.predict(X)
        positive_training_examples = X[training_predictions == 1]

        if positive_training_examples.shape[0] > 0:
            self.pessimistic_classifier = OneClassSVM(kernel="rbf", nu=nu)
            self.pessimistic_classifier.fit(positive_training_examples)
    
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
