import ipdb
import random
import itertools
import numpy as np
from copy import deepcopy
from sklearn import svm
from simple_rl.agents.func_approx.dsc.dynamics.mpc import MPC


class ModelBasedOption(object):
    def __init__(self, name, parent, mdp, buffer_length, global_init, gestation_period, initiation_period,
                 timeout, max_steps, device, target_salient_event=None, path_to_model=""):
        self.mdp = mdp
        self.name = name
        self.parent = parent
        self.device = device
        self.timeout = timeout
        self.max_steps = max_steps
        self.global_init = global_init
        self.buffer_length = buffer_length
        self.target_salient_event = target_salient_event

        # TODO
        self.overall_mdp = mdp
        self.seed = 0
        self.option_idx = 1

        self.num_goal_hits = 0
        self.gestation_period = gestation_period
        self.initiation_period = initiation_period

        self.positive_examples = []
        self.negative_examples = []
        self.optimistic_classifier = None
        self.pessimistic_classifier = None

        self.solver = MPC(self.mdp.state_space_size(), self.mdp.action_space_size(), self.device)

        if path_to_model:
            self.solver.load_model(path_to_model)

    # ------------------------------------------------------------
    # Learning Phase Methods
    # ------------------------------------------------------------

    def get_training_phase(self):
        if self.num_goal_hits < self.gestation_period:
            return "gestation"
        if self.num_goal_hits < (self.gestation_period + self.initiation_period):
            return "initiation"
        if self.num_goal_hits >= (self.gestation_period + self.initiation_period):
            return "initiation_done"
        return "trained"

    def is_init_true(self, state):
        if self.global_init or self.get_training_phase() == "gestation":
            return True

        features = self.mdp.get_position(state)
        return self.optimistic_classifier.predict([features])[0] == 1

    def is_term_true(self, state):
        if self.parent is None:
            return self.target_salient_event(state)

        return self.parent.pessimistic_is_init_true(state)

    def pessimistic_is_init_true(self, state):
        if self.global_init or self.get_training_phase() == "gestation":
            return True

        features = self.mdp.get_position(state)
        return self.pessimistic_classifier.predict([features])[0] == 1

    def is_at_local_goal(self, state, goal):
        """ Goal-conditioned termination condition. """

        reached_goal = self.mdp.sparse_gc_reward_function(state, goal, {})[1]
        reached_term = self.is_term_true(state) or state.is_terminal()
        return reached_goal and reached_term

    # ------------------------------------------------------------
    # Control Loop Methods
    # ------------------------------------------------------------

    def act(self, state, goal):
        """ Epsilon-greedy action selection. """

        if random.random() < 0.2:
            return self.mdp.sample_random_action()
        return self.solver.act(state, goal)

    def update(self, state, action, reward, next_state):
        """ Learning update for option model/actor/critic. """
        pass

    def get_goal_for_rollout(self):
        """ Sample goal to pursue for option rollout. """

        if self.parent is None and self.target_salient_event is not None:
            return self.target_salient_event.get_target_position()

        sampled_goal = self.parent.sample_from_initiation_region()
        assert sampled_goal is not None
        return sampled_goal.squeeze()

    def rollout(self, step_number):
        """ Main option control loop. """

        start_state = deepcopy(self.mdp.cur_state)
        assert self.is_init_true(start_state)

        num_steps = 0
        total_reward = 0
        visited_states = []
        option_transitions = []

        state = deepcopy(self.mdp.cur_state)
        goal = self.get_goal_for_rollout()

        print(f"Rolling out {self.name}, from {state.position} targeting {goal}")

        while not self.is_at_local_goal(state, goal) and step_number < self.max_steps and num_steps < self.timeout:

            # Control
            action = self.act(state, goal)
            reward, next_state = self.mdp.execute_agent_action(action)
            self.update(state, action, reward, next_state)

            # Logging
            num_steps += 1
            step_number += 1
            total_reward += reward
            visited_states.append(state)
            option_transitions.append((state, action, reward, next_state))
            state = deepcopy(self.mdp.cur_state)

        visited_states.append(state)

        if self.is_term_true(state):
            self.num_goal_hits += 1

        self.derive_positive_and_negative_examples(visited_states)

        if not self.global_init and self.get_training_phase() == "initiation":
            self.fit_initiation_classifier()

        return option_transitions, total_reward

    # ------------------------------------------------------------
    # Learning Initiation Classifiers
    # ------------------------------------------------------------

    def sample_from_initiation_region(self):
        """ Sample from the pessimistic initiation classifier. """

        positive_feature_matrix = self.construct_feature_matrix(self.positive_examples)
        training_predictions = self.pessimistic_classifier.predict(positive_feature_matrix)
        positive_training_examples = positive_feature_matrix[training_predictions == 1]
        if positive_training_examples.shape[0] > 0:
            idx = random.sample(range(positive_training_examples.shape[0]), k=1)
            return positive_training_examples[idx]

    def derive_positive_and_negative_examples(self, visited_states):
        start_state = visited_states[0]
        final_state = visited_states[-1]

        if self.is_term_true(final_state):
            positive_states = [start_state] + visited_states[-self.buffer_length:]
            positive_examples = [self.mdp.get_position(s) for s in positive_states]
            self.positive_examples.append(positive_examples)
        else:
            negative_examples = [self.mdp.get_position(start_state)]
            self.negative_examples.append(negative_examples)

    def fit_initiation_classifier(self):
        if len(self.negative_examples) > 0 and len(self.positive_examples) > 0:
            self.train_two_class_classifier()
        elif len(self.positive_examples) > 0:
            self.train_one_class_svm()

    @staticmethod
    def construct_feature_matrix(examples):
        states = list(itertools.chain.from_iterable(examples))
        return np.array(states)

    def train_one_class_svm(self, nu=0.1):
        positive_feature_matrix = self.construct_feature_matrix(self.positive_examples)
        self.pessimistic_classifier = svm.OneClassSVM(kernel="rbf", nu=nu, gamma="scale")
        self.pessimistic_classifier.fit(positive_feature_matrix)

    def train_two_class_classifier(self, nu=0.1):
        positive_feature_matrix = self.construct_feature_matrix(self.positive_examples)
        negative_feature_matrix = self.construct_feature_matrix(self.negative_examples)
        positive_labels = [1] * positive_feature_matrix.shape[0]
        negative_labels = [0] * negative_feature_matrix.shape[0]

        X = np.concatenate((positive_feature_matrix, negative_feature_matrix))
        Y = np.concatenate((positive_labels, negative_labels))

        if negative_feature_matrix.shape[0] >= 10:
            kwargs = {"kernel": "rbf", "gamma": "scale", "class_weight": "balanced"}
        else:
            kwargs = {"kernel": "rbf", "gamma": "scale"}

        self.optimistic_classifier = svm.SVC(**kwargs)
        self.optimistic_classifier.fit(X, Y)

        training_predictions = self.optimistic_classifier.predict(X)
        positive_training_examples = X[training_predictions == 1]

        if positive_training_examples.shape[0] > 0:
            self.pessimistic_classifier = svm.OneClassSVM(kernel="rbf", nu=nu, gamma="scale")
            self.pessimistic_classifier.fit(positive_training_examples)
