import ipdb
import torch
import random
import itertools
import numpy as np
from copy import deepcopy
from thundersvm import OneClassSVM, SVC
from simple_rl.agents.func_approx.dsc.dynamics.mpc import MPC
from simple_rl.agents.func_approx.td3.TD3AgentClass import TD3


class ModelBasedOption(object):
    def __init__(self, *, name, parent, mdp, global_solver, global_value_learner, buffer_length, global_init,
                 gestation_period, timeout, max_steps, device, use_vf, use_global_vf, use_model, dense_reward,
                 option_idx, lr_c, lr_a, max_num_children=2, target_salient_event=None, path_to_model=""):
        self.mdp = mdp
        self.name = name
        self.lr_c = lr_c
        self.lr_a = lr_a
        self.parent = parent
        self.device = device
        self.use_vf = use_vf
        self.use_global_vf = use_global_vf
        self.timeout = timeout
        self.use_model = use_model
        self.max_steps = max_steps
        self.global_init = global_init
        self.dense_reward = dense_reward
        self.buffer_length = buffer_length
        self.max_num_children = max_num_children
        self.target_salient_event = target_salient_event

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

        # In the model-free setting, the output norm doesn't seem to work
        # But it seems to stabilize off policy value function learning
        # Therefore, only use output norm if we are using MPC for action selection
        use_output_norm = self.use_model

        if not self.use_global_vf or global_init:
            self.value_learner = TD3(state_dim=self.mdp.state_space_size()+2,
                                    action_dim=self.mdp.action_space_size(),
                                    max_action=1.,
                                    name=f"{name}-td3-agent",
                                    device=self.device,
                                    lr_c=lr_c, lr_a=lr_a,
                                    use_output_normalization=use_output_norm)

        self.global_value_learner = global_value_learner if not self.global_init else None  # type: TD3

        if global_solver is not None and use_model:
            assert not self.global_init, "Assuming local option here"
            self.solver = global_solver
        elif use_model:
            assert self.global_init, "Assuming global option here"
            print(f"Using model-based controller for {name}")
            self.solver = MPC(mdp=self.mdp,
                              state_size=self.mdp.state_space_size(),
                              action_size=self.mdp.action_space_size(),
                              dense_reward=self.dense_reward,
                              device=self.device)
        else:
            print(f"Using model-free controller for {name}")
            self.solver = self.value_learner if not self.use_global_vf else self.global_value_learner

        self.children = []
        self.success_curve = []
        self.effect_set = []

        if path_to_model:
            print(f"Loading model from {path_to_model} for {self.name}")
            self.solver.load_model(path_to_model)

        if self.use_vf and not self.use_global_vf and self.parent is not None:
            self.initialize_value_function_with_global_value_function()

        print(f"Created model-based option {self.name} with option_idx={self.option_idx}")

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

        features = self.mdp.get_position(state)
        return self.optimistic_classifier.predict([features])[0] == 1 or self.pessimistic_is_init_true(state)

    def is_term_true(self, state):
        if self.parent is None:
            return self.target_salient_event(state)

        # TODO change
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

    def _get_epsilon(self):
        if self.use_model:
            return 0.1
        if not self.dense_reward and self.num_goal_hits <= 3:
            return 0.8
        return 0.2

    def act(self, state, goal):
        """ Epsilon-greedy action selection. """

        if random.random() < self._get_epsilon():
            return self.mdp.sample_random_action()

        if self.use_model:
            assert isinstance(self.solver, MPC), f"{type(self.solver)}"
            vf = self.value_function if self.use_vf else None
            return self.solver.act(state, goal, vf=vf)

        assert isinstance(self.solver, TD3), f"{type(self.solver)}"
        augmented_state = self.get_augmented_state(state, goal)
        return self.solver.act(augmented_state, evaluation_mode=False)

    def update_model(self, state, action, reward, next_state):
        """ Learning update for option model/actor/critic. """

        self.solver.step(state.features(), action, reward, next_state.features(), next_state.is_terminal())

    def get_goal_for_rollout(self):
        """ Sample goal to pursue for option rollout. """

        if self.parent is None and self.target_salient_event is not None:
            return self.target_salient_event.get_target_position()

        sampled_goal = self.parent.sample_from_initiation_region_fast_and_epsilon()
        assert sampled_goal is not None

        if isinstance(sampled_goal, np.ndarray):
            return sampled_goal.squeeze()

        return self.extract_goal_dimensions(sampled_goal)

    def rollout(self, step_number, rollout_goal=None, eval_mode=False):
        """ Main option control loop. """

        start_state = deepcopy(self.mdp.cur_state)
        assert self.is_init_true(start_state)

        num_steps = 0
        total_reward = 0
        visited_states = []
        option_transitions = []

        state = deepcopy(self.mdp.cur_state)
        goal = self.get_goal_for_rollout() if rollout_goal is None else rollout_goal

        print(f"[Step: {step_number}] Rolling out {self.name}, from {state.position} targeting {goal}")

        self.num_executions += 1

        while not self.is_at_local_goal(state, goal) and step_number < self.max_steps and num_steps < self.timeout:

            # Control
            action = self.act(state, goal)
            reward, next_state = self.mdp.execute_agent_action(action)

            if self.use_model:
                self.update_model(state, action, reward, next_state)

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

        if self.use_vf and not eval_mode:
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
        goal_features = goal if isinstance(goal, np.ndarray) else goal.features()
        if "ant" in self.mdp.env_name:
            return goal_features[:2]
        raise NotImplementedError(f"{self.mdp.env_name}")

    def get_augmented_state(self, state, goal):
        assert goal is not None and isinstance(goal, np.ndarray)

        goal_position = self.extract_goal_dimensions(goal)
        return np.concatenate((state.features(), goal_position))

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

        goal_positions = goals[:, :2]
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

    def get_first_state_in_classifier(self, trajectory, classifier_type="pessimistic"):
        """ Extract the first state in the trajectory that is inside the initiation classifier. """

        assert classifier_type in ("pessimistic", "optimistic"), classifier_type
        classifier = self.pessimistic_is_init_true if classifier_type == "pessimistic" else self.is_init_true
        for state in trajectory:
            if classifier(state):
                return state
        return None

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

    def should_change_negative_examples(self):
        should_change = []
        for negative_example in self.negative_examples:
            should_change += [self.does_model_rollout_reach_goal(negative_example[0])]
        return should_change

    def does_model_rollout_reach_goal(self, state):
        sampled_goal = self.get_goal_for_rollout()
        final_states, actions, costs = self.solver.simulate(state, sampled_goal, num_rollouts=14000, num_steps=self.timeout)
        farthest_position = final_states[:, :2].max(axis=0)
        return self.is_term_true(farthest_position)

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

    def get_option_success_rate(self):
        """
        TODO: implement success rate at test time as well (Jason)
        """
        if self.num_executions > 0:
            return self.num_goal_hits / self.num_executions
        return 1.

    # ------------------------------------------------------------
    # Convenience functions
    # ------------------------------------------------------------

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
