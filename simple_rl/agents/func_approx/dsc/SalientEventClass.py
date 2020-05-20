import numpy as np
from simple_rl.mdp.StateClass import State
from scipy.spatial import distance
import pdb


class SalientEvent(object):
    def __init__(self, target_state, event_idx, tolerance=0.6, intersection_event=False, get_relevant_position=None):
        self.target_state = target_state
        self.event_idx = event_idx
        self.tolerance = tolerance
        self.intersection_event = intersection_event

        if get_relevant_position is None:
            self.get_relevant_position = self._get_position
        else:
            self.get_relevant_position = get_relevant_position

        assert isinstance(event_idx, int)
        assert isinstance(tolerance, float)

    def __call__(self, states):
        """

        Args:
            states: this can either be an array representing a single state, or an array
                    representing a batch of states or a State object

        Returns:
            is_satisfied: bool or bool array depending on the shape of states.
        """
        if isinstance(states, State):
            return self.is_init_true(states)
        if len(states.shape) == 1:
            return self.is_init_true(states)
        return self.batched_is_init_true(states)

    def __eq__(self, other):
        def _state_eq(s1, s2):
            s1 = self.get_relevant_position(s1)
            s2 = self.get_relevant_position(s2)
            return (s1 == s2).all()

        if not isinstance(other, SalientEvent):
            return False

        return _state_eq(self.target_state, other.target_state) and \
               self.tolerance == other.tolerance and \
               self.event_idx == other.event_idx

    def __hash__(self):
        return self.event_idx

    def is_init_true(self, state):
        try:
            position = self.get_relevant_position(state)
            target_position = self.get_relevant_position(self.target_state)
            return np.linalg.norm(position - target_position) <= self.tolerance
        except:
            pdb.set_trace()

    def batched_is_init_true(self, position_matrix):
        pdb.set_trace()
        assert isinstance(position_matrix, np.ndarray), type(position_matrix)
        goal_position = self.get_relevant_position(self.target_state)
        in_goal_position = distance.cdist(self.get_relevant_position(position_matrix), goal_position[None, :]) <= self.tolerance
        return in_goal_position.squeeze(1)

    @staticmethod
    def _get_position(state):
        position = state.position if isinstance(state, State) else state[:2]
        assert isinstance(position, np.ndarray), type(position)
        return position

    def __repr__(self):
        return f"SalientEvent targeting {self.target_state}"


class UnionSalientEvent(SalientEvent):
    def __init__(self, event1, event2, event_idx):
        """

        Args:
            event1 (SalientEvent)
            event2 (SalientEvent)
            event_idx (int)
        """
        self.event1 = event1
        self.event2 = event2
        
        super(UnionSalientEvent, self).__init__(None, event_idx)

    def __eq__(self, other):
        if not isinstance(other, UnionSalientEvent):
            return False

        other_events = [other.event1, other.event2]
        return self.event1 in other_events and self.event2 in other_events

    def __repr__(self):
        return f"{self.event1} U {self.event2}"

    def is_init_true(self, state):
        return self.event1(state) or self.event2(state)

    def batched_is_init_true(self, position_matrix):
        inits1 = self.event1.batched_is_init_true(position_matrix)
        inits2 = self.event2.batched_is_init_true(position_matrix)
        return np.logical_or(inits1, inits2)

