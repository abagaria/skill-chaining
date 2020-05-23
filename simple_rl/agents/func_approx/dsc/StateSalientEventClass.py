import numpy as np

from simple_rl.agents.func_approx.dsc.BaseSalientEventClass import BaseSalientEvent
from simple_rl.mdp.StateClass import State
from scipy.spatial import distance
import pdb


class StateSalientEvent(BaseSalientEvent):
    def __init__(self, target_state, event_idx, name='', tolerance=0.6,
                 intersection_event=False, get_relevant_position=None):
        self.tolerance = tolerance
        self.name = name

        if get_relevant_position is None:
            self.get_relevant_position = self._get_position
        else:
            self.get_relevant_position = get_relevant_position

        assert isinstance(tolerance, float)

        BaseSalientEvent.__init__(
            self,
            is_init_true=self.is_init_true,
            event_idx=event_idx,
            batched_is_init_true=self.batched_is_init_true,
            intersection_event=intersection_event,
            target_state=target_state
        )

    def __eq__(self, other):
        def _state_eq(s1, s2):
            s1 = self.get_relevant_position(s1)
            s2 = self.get_relevant_position(s2)
            return (s1 == s2).all()

        if not isinstance(other, StateSalientEvent):
            return False

        return _state_eq(self.target_state, other.target_state) and \
               self.tolerance == other.tolerance and \
               self.event_idx == other.event_idx

    def is_init_true(self, state):
        position = self.get_relevant_position(state)
        target_position = self.get_relevant_position(self.target_state)
        return np.linalg.norm(position - target_position) <= self.tolerance

    def batched_is_init_true(self, position_matrix):
        pdb.set_trace()
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
        return f"SalientEvent targeting {self.target_state} {self.name}"
