import numpy as np
from simple_rl.mdp.StateClass import State
from scipy.spatial import distance
import pdb


class BaseSalientEvent(object):
    def __init__(self, target_state, event_idx, name=None, tolerance=0.6,
            intersection_event=False, get_relevant_position=None):
            
        self.event_idx = event_idx
        self.tolerance = tolerance
        self.target_state = target_state
        self.name = name

        if get_relevant_position is None:
            self.get_relevant_position = self._get_position
        else:
            self.get_relevant_position = get_relevant_position

        assert isinstance(tolerance, float)

    def __call__(self, states):
        """
        Args:
            states: this can either be an array representing a single state, or an array
                    representing a batch of states or a State object
        Returns:
            is_satisfied: bool or bool array depending on the shape of states.
        """
        if isinstance(states, State) or len(states.shape) == 1:
            return self.is_init_true(states)
        return self.batched_is_init_true(states)

    def __eq__(self, other):
        if isinstance(other, BaseSalientEvent):
            return self.event_idx == other.event_idx
        else:
            return False

    def __hash__(self):
        return self.event_idx

    def is_init_true(self, state):
        position = self.get_relevant_position(state)
        target_position = self.get_relevant_position(self.target_state)
        return np.linalg.norm(position - target_position) <= self.tolerance

    def batched_is_init_true(self, position_matrix):
        assert isinstance(position_matrix, np.ndarray), type(position_matrix)
        goal_position = self.get_relevant_position(self.target_state)
        in_goal_position = distance.cdist(self.get_relevant_position(position_matrix),
                                          goal_position[None, :]) <= self.tolerance
        return in_goal_position.squeeze(1)

    @staticmethod
    def _get_position(state):
        position = state.position if isinstance(state, State) else state[:2]
        assert isinstance(position, np.ndarray), type(position)
        return position

    def __repr__(self):
        if self.name is not None:
            return f"SalientEvent targeting {self.target_state} {self.name}"
        else:
            return f"SalientEvent targeting {self.target_state}"
