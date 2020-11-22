import ipdb
import time
import torch
import random
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict


class SubgoalOption(object):
    """ Simple version of option class that we use for picking subgoals. """

    def __init__(self, dsc_option, name, parent=None):
        self.name = name
        self.dsc_option = dsc_option
        self.input_states, self.output_states = self.get_input_output_states()
        self.parent = parent

        super(SubgoalOption, self).__init__()

    def get_input_output_states(self):
        """ Extract input, output states from the DSC Options. """

        features = lambda s: s if isinstance(s, np.ndarray) else s.features()

        input_states = [traj[0] for traj in self.dsc_option.positive_examples]
        input_states = [s for s in input_states if self.dsc_option.is_init_true(s)]
        input_states = [features(s) for s in input_states]

        output_states = self.dsc_option.effect_set
        output_states = [s for s in output_states if self.dsc_option.is_term_true(s)]
        output_states = [features(s) for s in output_states]
        return input_states, output_states

    def __str__(self):
        return self.name

    def __repr__(self):
        return str(self)


class OptimalSubgoalSelector(object):
    """ Implements algorithm for picking subgoals that approximate the hierarchically optimal solution. """

    def __init__(self, options, goal_state, n_state_dims=29, n_goal_dims=2, gamma=0.95, tolerance=0.6):

        # Discount factor
        self.gamma = gamma

        # Number of dimensions in the state vector and the goal vectors
        self.n_state_dims = n_state_dims
        self.n_goal_dims = n_goal_dims

        # Option sequence that leads to the goal state. Reorganize so the
        # first option is the one that leads to the goal (for DP)
        self.options = self.parse_dsc_options(options)

        # Overall goal state of the MDP
        self.overall_goal = goal_state

        # State identity function
        self.is_equal = lambda s1, s2: np.linalg.norm(s1[:2] - s2[:2]) <= tolerance

        # (s, g) -> reward
        self.reward_table = self._create_reward_table()

        # (s, g) -> q-value
        self.Q = defaultdict(lambda: defaultdict(float))

        # Debug: (state, goal, option) pairs for which we individually had to
        # query the option-value function rather than using the cached values in
        # the reward table constructed in the beginning
        self.vf_queries = []

    # ---------------------------------------------------------
    # Q-Table Construction
    # ---------------------------------------------------------

    def _create_reward_table(self):
        """ Query the option-value function in a batched way -- one forward pass of the VF for one option. """

        start_time = time.time()

        reward_table = defaultdict(lambda: defaultdict(float))
        for option in self.options:  # type: SubgoalOption
            goal_conditioned_states, goal_states = self._get_augmented_states_for_option(option)
            states_tensor = torch.as_tensor(goal_conditioned_states).float().to(option.dsc_option.device)
            values = option.dsc_option.value_learner.get_values(states_tensor)

            for i, (augmented_state, goal_state) in enumerate(zip(goal_conditioned_states, goal_states)):
                input_state = augmented_state[:-self.n_goal_dims]
                input_key, output_key = self.get_keys(input_state, goal_state)

                assert len(input_key) == len(output_key) == self.n_state_dims

                reward_table[input_key][output_key] = values[i].item()

        print(f"Took {time.time() - start_time} s to construct reward table.")

        return reward_table

    def _get_augmented_states_for_option(self, option):
        """
        Given an option with N input and M output states, get N x M goal-conditioned states.
        Also return the full goal states corresponding to the augmented states for
        determining the keys in the reward hash table.

        Args:
            option (SubgoalOption)

        Returns:
            augmented_state_matrix (np.ndarray)
            output_states (list)

        """
        def get_output_states(o):
            if o.parent is None:
                return o.output_states
            return o.output_states + o.parent.input_states

        augmented_states = []
        output_states = []
        for input_state in option.input_states:  # type: np.ndarray
            for output_state in get_output_states(option):  # type: np.ndarray
                goal_position = output_state[:self.n_goal_dims]
                augmented_state = np.concatenate((input_state, goal_position), axis=0)
                augmented_states.append(augmented_state)
                output_states.append(output_state)

        # augmented_states is a list of length N x M, where each element has dimension (31,)
        return np.array(augmented_states), output_states

    def R(self, state, goal, option):
        """ Local reward function. """

        key1, key2 = self.get_keys(state, goal)

        if key1 in self.reward_table:
            if key2 in self.reward_table[key1]:
                return self.reward_table[key1][key2]

        self.vf_queries.append((option, state, goal))
        return option.dsc_option.value_function(state, goal)

    @staticmethod
    def get_output_states(option):
        if option.parent is None:
            return option.output_states
        return option.parent.input_states

    def construct_table_for_option(self, option):
        print(f"Constructing table for option {option.name}")
        output_states = self.get_output_states(option)

        for input_state in option.input_states:
            for output_state in output_states:

                if self.is_equal(output_state, self.overall_goal):
                    value = self.R(input_state, output_state, option)
                else:
                    parent_output_states = self.get_output_states(option.parent)
                    Q_next = max([self.get_value(output_state, sg_prime) for sg_prime in parent_output_states])
                    value = self.R(input_state, output_state, option) + (self.gamma * Q_next)

                self.set_value(input_state, output_state, value)

    def construct_table(self, options):
        for option in options:
            self.construct_table_for_option(option)

    # ---------------------------------------------------------
    # Subgoal interface
    # ---------------------------------------------------------

    def pick_subgoal(self, state, option):
        output_states = self.get_output_states(option)
        values = [(goal, self.get_value(state, goal)) for goal in output_states]
        subgoal = sorted(values, key=lambda x: x[1], reverse=True)[0][0]
        return subgoal

    def get_all_subgoals(self, current_state, current_option):
        selected_subgoals = []
        while not self.is_equal(current_state, self.overall_goal) and not current_option is None:
            subgoal = self.pick_subgoal(current_state, current_option)
            print(f"Picked {subgoal[:2]} for {current_option}")
            current_state = subgoal
            current_option = current_option.parent
            selected_subgoals.append(subgoal)
        return selected_subgoals

    # ---------------------------------------------------------
    # Q-Table Management
    # ---------------------------------------------------------

    def get_value(self, state, goal):

        if self.is_equal(state, goal):
            return 0.

        key1, key2 = self.get_keys(state, goal)

        if key1 in self.Q:
            if key2 in self.Q[key1]:
                return self.Q[key1][key2]

        # It is possible that (s, g) is not in the parent option's
        # Q-table. If so, use the parent option's reward function as a stand in
        option = self.get_option(state, goal)
        assert option is not None, f"{state, goal}"

        return self.R(state, goal, option)

    def set_value(self, state, goal, value):
        key1, key2 = self.get_keys(state, goal)
        self.Q[key1][key2] = value

    def get_keys(self, state, goal):

        # If the goal state is roughly equal to the overall goal,
        # we collapse the goal to the overall goal. This helps with
        # (a) grounding the terminal values in the Q-table and
        # (b) speeding up the process of constructing the Q-table
        if self.is_equal(goal, self.overall_goal):
            goal = self._get_expanded_goal_state()

        # We also round the number of decimal places of the keys in the
        # Q-table & R-table to avoid any funny numerical rounding issues
        # while querying the respective hash tables
        key1 = tuple(np.round(state, decimals=2))
        key2 = tuple(np.round(goal, decimals=2))

        return key1, key2

    def _get_expanded_goal_state(self):
        """ Pad the goal-position with zeros to match the dimensionality of the state vector. """

        goal_state = np.zeros((self.n_state_dims,))
        goal_state[0] = self.overall_goal[0]
        goal_state[1] = self.overall_goal[1]

        return goal_state

    # ---------------------------------------------------------
    # Options Management
    # ---------------------------------------------------------

    def parse_dsc_options(self, options):

        subgoal_options = []
        options = options if options[0].parent is None else list(reversed(options))
        parent = None

        for option in options:
            local_option = SubgoalOption(option, name=option.name, parent=parent)
            subgoal_options.append(local_option)
            parent = local_option

        return subgoal_options

    def get_option(self, state, goal):
        cond1 = lambda s, o: o.is_init_true(s) and not o.is_term_true(s)
        cond2 = lambda g, o: o.is_term_true(g)

        for option in self.options:
            if cond1(state, option.dsc_option) and cond2(goal, option.dsc_option):
                return option

        return None

    # ---------------------------------------------------------
    # Options Management
    # ---------------------------------------------------------

    def visualize_values(self, options):
        xx, yy, cc = [], [], []

        for option in options:
            pairs = []
            for s_in in option.input_states:
                values = [self.get_value(s_in, s_out) for s_out in option.output_states]
                pairs.append((s_in, max(values)))
            x = [pair[0][0] for pair in pairs]
            y = [pair[0][1] for pair in pairs]
            c = [pair[1] for pair in pairs]
            xx += x
            yy += y
            cc += c

        plt.scatter(xx, yy, c=cc, cmap=plt.cm.Dark2)
        plt.colorbar()

        init_state = random.choice(options[-1].input_states)
        subgoals = self.get_all_subgoals(init_state, options[-1])

        for subgoal in subgoals:
            plt.scatter(subgoal[0], subgoal[1], s=200, marker="x")
