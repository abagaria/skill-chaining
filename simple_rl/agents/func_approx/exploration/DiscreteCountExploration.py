import numpy as np
from collections import defaultdict
import pdb


class CountBasedDensityModel(object):
    def __init__(self, state_rounding_decimals=None, action_rounding_decimals=None, use_position_only=False):
        """
        Create a count based exploration module that will discretize the state-action
        space if it is not already discrete. It will then maintain a visitation count
        dictionary, based on which it will assign exploration bonuses.
        Args:
            state_rounding_decimals (int)
            action_rounding_decimals (int)
            use_position_only (bool)
        """
        self.state_rounding_decimals = state_rounding_decimals
        self.action_rounding_decimals = action_rounding_decimals
        self.use_position_only = use_position_only

        # Map (state, action) pairs to visitation counts and exploration bonuses
        self.s_a_counts = defaultdict(lambda : defaultdict(lambda: 0))
        self.s_a_bonus  = defaultdict(lambda : defaultdict(lambda: 0))

    def _round_state_action(self, state, action):
        if self.use_position_only:
            state = state[:2]
        if self.state_rounding_decimals is not None:
            state = np.round(state, self.state_rounding_decimals)
        if self.action_rounding_decimals is not None:
            action = np.round(action, self.action_rounding_decimals)
        return state, action

    def get_single_count(self, state, action):
        state, action = self._round_state_action(state, action)
        return self.s_a_counts[tuple(state)][action]

    def _update_single_count(self, state, action):
        state, action = self._round_state_action(state, action)
        self.s_a_counts[tuple(state)][action] += 1

    def _update_single_bonus(self, state, action, bonus):
        state, action = self._round_state_action(state, action)
        self.s_a_bonus[tuple(state)][action] = bonus

    def get_online_exploration_bonus(self, state, action, beta=5e-3):
        count = self.get_single_count(state, action)
        bonus = beta * np.power(count + 0.01, -0.5)
        self._update_single_count(state, action)
        self._update_single_bonus(state, action, bonus)
        return bonus

    def reset(self):
        self.s_a_counts = defaultdict(lambda: defaultdict(lambda: 0))
        self.s_a_bonus = defaultdict(lambda: defaultdict(lambda: 0))
