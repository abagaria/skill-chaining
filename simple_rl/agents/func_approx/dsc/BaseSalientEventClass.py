import numpy as np
from simple_rl.mdp.StateClass import State
from scipy.spatial import distance
import pdb


class BaseSalientEvent(object):
    def __init__(self, is_init_true, event_idx, batched_is_init_true=None, name=None, intersection_event=False, target_state=None):
        self._is_init_true = is_init_true
        self._batched_is_init_true = batched_is_init_true
        self.event_idx = event_idx
        self.intersection_event = intersection_event
        self.name = name
        self.target_state = None

        assert isinstance(event_idx, int)

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
        if not isinstance(other, BaseSalientEvent):
            return False

        return self.event_idx == other.event_idx  # TODO: probably a better way to check if equal

    def __hash__(self):
        return self.event_idx

    def is_init_true(self, state):
        return self._is_init_true(state)

    def batched_is_init_true(self, position_matrix):
        if self._batched_is_init_true is not None:
            return self._batched_is_init_true(position_matrix)
        else:
            return [self.is_init_true(state) for state in position_matrix]

    def __repr__(self):
        return f"SalientEvent: {self.name}"


class UnionSalientEvent(BaseSalientEvent):
    def __init__(self, event1, event2, event_idx):
        """

        Args:
            event1 (BaseSalientEvent)
            event2 (BaseSalientEvent)
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

