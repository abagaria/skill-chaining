import ipdb
import random
import itertools
import numpy as np
from copy import deepcopy
from thundersvm import OneClassSVM, SVC
from simple_rl.agents.func_approx.td3.TD3AgentClass import TD3


class ModelFreeOption(object):
    def __init__(self, *, name, parent, mdp, global_solver, buffer_length, global_init,
                 gestation_period, initiation_period, timeout, max_steps, device, dense_reward,
                 option_idx, lr_c, lr_a, use_pessimistic_clf_only=False, target_salient_event=None):
        self.seed = 0
        self.mdp = mdp
        self.name = name
        self.lr_c = lr_c
        self.lr_a = lr_a
        self.parent = parent
        self.device = device
        self.timeout = timeout
        self.max_steps = max_steps
        self.option_idx = option_idx
        self.global_init = global_init
        self.dense_reward = dense_reward
        self.buffer_length = buffer_length
        self.target_salient_event = target_salient_event
        self.overall_mdp = mdp

        self.num_goal_hits = 0
        self.num_executions = 0
        self.gestation_period = gestation_period
        self.initiation_period = initiation_period

        self.positive_examples = []
        self.negative_examples = []
        self.optimistic_classifier = None
        self.pessimistic_classifier = None
        self.use_pessimistic_clf_only = use_pessimistic_clf_only

        self.solver = TD3(state_dim=self.mdp.state_space_size(),
                         action_dim=self.mdp.action_space_size(),
                         max_action=1.,
                         name=f"{name}-td3-agent",
                         device=self.device,
                         lr_c=lr_c, lr_a=lr_a,
                         use_output_normalization=False)

        self.global_solver = global_solver if not self.global_init else None  # type: TD3

        self.children = []
        self.success_curve = []
        self.effect_set = []

        if self.parent is not None:
            self.initialize_value_function_with_global_value_function()

        print(f"Created baseline model-free option {self.name} with option_idx={self.option_idx}")

    # ------------------------------------------------------------
    # Learning Phase Methods
    # ------------------------------------------------------------

    def get_training_phase(self):
        if self.num_goal_hits < self.gestation_period:
            return "gestation"
        if self.num_goal_hits < self.initiation_period:
            return "initiation"
        return "initiation_done"

    def is_init_true(self, state):
        if self.global_init or self.get_training_phase() == "gestation":
            return True

        features = self.mdp.get_position(state)

        if self.use_pessimistic_clf_only:
            return self.pessimistic_is_init_true(state)

        return self.optimistic_classifier.predict([features])[0] == 1 or self.pessimistic_is_init_true(state)

    def is_term_true(self, state):
        if self.parent is None:
            return self.target_salient_event(state)

        return self.parent.pessimistic_is_init_true(state)

    def pessimistic_is_init_true(self, state):
        if self.global_init or self.get_training_phase() == "gestation":
            return True

        features = self.mdp.get_position(state)
        return self.pessimistic_classifier.predict([features])[0] == 1

    # ------------------------------------------------------------
    # Control Loop Methods
    # ------------------------------------------------------------

    def _get_epsilon(self):
        if not self.dense_reward and self.num_goal_hits <= 3:
            return 0.8
        return 0.2

    def act(self, state):
        """ Epsilon-greedy action selection. """

        if random.random() < self._get_epsilon():
            return self.mdp.sample_random_action()

        assert isinstance(self.solver, TD3), f"{type(self.solver)}"
        return self.solver.act(state.features(), evaluation_mode=False)

    def rollout(self, step_number, eval_mode=False):
        """ Main option control loop. """

        start_state = deepcopy(self.mdp.cur_state)
        assert self.is_init_true(start_state)

        num_steps = 0
        total_reward = 0
        visited_states = []
        option_transitions = []

        state = deepcopy(self.mdp.cur_state)
        
        if not self.global_init:
            print(f"[Step: {step_number}] Rolling out {self.name}, from {state.position}")

        self.num_executions += 1

        while not self.is_term_true(state) and step_number < self.max_steps and num_steps < self.timeout:

            # Control
            action = self.act(state)
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
        self.effect_set.append(state.features())

        if self.is_term_true(state):
            self.num_goal_hits += 1

        if not eval_mode:
            self.experience_replay(option_transitions)

        if not self.global_init:
            self.derive_positive_and_negative_examples(visited_states)

        # Always be refining your initiation classifier
        if not self.global_init and not eval_mode:
            self.fit_initiation_classifier()

        return option_transitions, total_reward

    # ------------------------------------------------------------
    # Experience Replay
    # ------------------------------------------------------------

    def distance_to_goal_state(self, state):
        pos = self.mdp.get_position(state)
        goal = self.target_salient_event.get_target_position()
        return np.linalg.norm(pos-goal)

    def distance_to_goal_region(self, state):
        pos = self.mdp.get_position(state)
        if self.parent is None:
            return self.distance_to_goal_state(state)
        # Decision_function returns a negative distance for points not inside the classifier
        return -self.parent.pessimistic_classifier.decision_function(pos.reshape(1, -1))[0]

    def get_local_reward(self, state):
        if self.dense_reward:
            return -self.distance_to_goal_region(state)
        return -1.

    def get_global_reward(self, state):
        if self.dense_reward:
            return -self.distance_to_goal_state(state)
        return -1.

    def initialize_value_function_with_global_value_function(self):
        self.solver.actor.load_state_dict(self.global_solver.actor.state_dict())
        self.solver.critic.load_state_dict(self.global_solver.critic.state_dict())
        self.solver.target_actor.load_state_dict(self.global_solver.target_actor.state_dict())
        self.solver.target_critic.load_state_dict(self.global_solver.target_critic.state_dict())

    def experience_replay(self, trajectory):

        for state, action, reward, next_state in trajectory:
            done = self.is_term_true(next_state)
            global_done = self.target_salient_event(next_state)

            local_reward = +0 if done else self.get_local_reward(next_state)
            global_reward = +0. if global_done else self.get_global_reward(next_state)

            self.solver.step(state.features(), action, local_reward, next_state.features(), done)

            # Off-policy updates to the global option value function
            if not self.global_init:
                assert self.global_solver is not None
                self.global_solver.step(state.features(), action, global_reward, next_state.features(), global_done)

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

    def get_sibling_options(self):
        if self.parent is not None:
            return [option for option in self.parent.children if option != self]
        return []

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
        if isinstance(other, ModelFreeOption):
            return self.name == other.name
        return False
