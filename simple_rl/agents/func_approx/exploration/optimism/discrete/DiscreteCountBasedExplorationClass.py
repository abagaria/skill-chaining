# Python imports.
from collections import defaultdict
import numpy as np
import pdb

# Other imports.
from simple_rl.agents.func_approx.exploration.optimism.ExplorationBonusClass import ExplorationBonus


class DiscreteCountBasedExploration(ExplorationBonus):
    def __init__(self, action_size):
        self.actions = list(range(0, action_size))

        # action -> state -> count (float)
        self.action_state_counts = defaultdict(lambda : defaultdict(float))
        super(DiscreteCountBasedExploration, self).__init__()

    def add_transition(self, state, action, next_state=None):
        assert isinstance(state, tuple), type(state)

        self.action_state_counts[action][state] += 1

    def get_exploration_bonus(self, state, action=None):
        assert state.shape == (2,)
        assert isinstance(state, np.ndarray)

        if action is not None:
            return 1. / np.sqrt(self.action_state_counts[action][tuple(state)] + 1e-2)

        # If `action` is not specified, return the exploration bonuses
        # corresponding to all the actions in the MDP
        bonuses = []
        for action in self.actions:
            bonus = 1. / np.sqrt(self.action_state_counts[action][tuple(state)] + 1e-2)
            bonuses.append(bonus)
        return np.array(bonuses)[None, ...]

    def get_batched_exploration_bonus(self, states):
        """
        Args:
            states (np.ndarray): List of tuples representing ground truth states.

        Returns:
            bonus_array (np.ndarray): array of shape (num_states, num_actions) with exploration bonuses.
        """
        assert isinstance(states, np.ndarray), type(states)

        counts = []
        for state in states:
            action_counts = []
            for action in self.actions:
                count = self.action_state_counts[action][tuple(state)]
                action_counts.append(count)
            counts.append(action_counts)
        count_array = np.array(counts)
        return 1. / np.sqrt(count_array + 1e-2)
