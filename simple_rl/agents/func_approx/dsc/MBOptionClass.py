import random
from copy import deepcopy
from simple_rl.agents.func_approx.dsc.dynamics.mpc import MPC


class ModelBasedOption(object):
    def __init__(self, parent, mdp, buffer_length, global_init, timeout, device, target_salient_event=None):
        self.mdp = mdp
        self.parent = parent
        self.device = device
        self.timeout = timeout
        self.global_init = global_init
        self.buffer_length = buffer_length
        self.target_salient_event = target_salient_event

        self.positive_examples = []
        self.negative_examples = []
        self.initiation_classifier = None  # TODO

        self.solver = MPC(self.mdp.state_space_size(), self.mdp.action_space_size(), self.device)
        self.solver.load_model("mpc-ant-umaze-model.pkl")

    def get_training_phase(self):
        return ""

    def is_init_true(self, state):
        if self.global_init or self.get_training_phase() == "gestation":
            return True

        features = self.mdp.get_position(state)
        return self.initiation_classifier.predict([features])[0] == 1

    def is_term_true(self, state):
        if self.parent is None:
            return self.target_salient_event(state)
        return self.parent.is_init_true(state)

    def is_at_local_goal(self, state, goal):
        reached_goal = self.mdp.sparse_gc_reward_function(state, goal, {})[1]
        reached_term = self.is_term_true(state) or state.is_terminal()
        return reached_goal and reached_term

    def act(self, state, goal):
        if random.random() < 0.2:
            return self.mdp.sample_random_action()
        return self.solver.act(state, goal)

    def get_goal_for_rollout(self):
        return self.mdp.cur_state

    def rollout(self, step_number):
        start_state = deepcopy(self.mdp.cur_state)
        assert self.is_init_true(start_state)

        goal = self.get_goal_for_rollout()

    def derive_positive_and_negative_examples(self, visited_states):
        start_state = visited_states[0]
        final_state = visited_states[1]

        if self.is_term_true(final_state):
            positive_states = [start_state] + visited_states[-self.buffer_length:]
            positive_examples = [self.mdp.get_position(s) for s in positive_states]
            self.positive_examples.append(positive_examples)
        else:
            negative_examples = [self.mdp.get_position(start_state)]
            self.negative_examples.append(negative_examples)

    def fit_initiation_classifier(self):
        pass
