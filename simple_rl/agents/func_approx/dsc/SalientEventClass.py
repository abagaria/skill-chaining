import numpy as np
from simple_rl.mdp.StateClass import State
from scipy.spatial import distance
from sklearn.svm import OneClassSVM
import ipdb


class SalientEvent(object):
    def __init__(self, target_state, event_idx, tolerance=0.6, intersection_event=False):
        self.target_state = target_state
        self.event_idx = event_idx
        self.tolerance = tolerance
        self.intersection_event = intersection_event

        # This is the union of the effect set of all the options targeting this salient event
        self.trigger_points = []

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
            s1 = self._get_position(s1)
            s2 = self._get_position(s2)
            return (s1 == s2).all()

        if not isinstance(other, SalientEvent):
            return False

        return _state_eq(self.target_state, other.target_state) and \
               self.tolerance == other.tolerance and \
               self.event_idx == other.event_idx

    def __hash__(self):
        return self.event_idx

    def is_init_true(self, state):
        position = self._get_position(state)
        target_position = self._get_position(self.target_state)
        return np.linalg.norm(position - target_position) <= self.tolerance

    def batched_is_init_true(self, position_matrix):
        assert isinstance(position_matrix, np.ndarray), type(position_matrix)
        goal_position = self._get_position(self.target_state)
        in_goal_position = distance.cdist(position_matrix, goal_position[None, :]) <= self.tolerance
        return in_goal_position.squeeze(1)

    def is_intersecting(self, option):
        """
        Is this salient event "intersecting" with another option? To intersect, we have to make sure
        that `option`'s initiation set includes all the current salient event. For this we have to
        have access to the effect sets of all the options that are targeting this salient event.
        Alternatively, we can keep track of all the states which were considered to successfully
        trigger the current salient event and make sure that they are all inside `option`'s initiation set.

        Args:
            option (Option)

        Returns:
            is_intersecting (bool)
        """
        if len(self.trigger_points) > 0 and option.get_training_phase() == "initiation_done":
            return all([option.is_init_true(s) for s in self.trigger_points])
        return False

    @staticmethod
    def _get_position(state):
        position = state.position if isinstance(state, State) else state[:2]
        assert isinstance(position, np.ndarray), type(position)
        return position

    def __repr__(self):
        return f"SalientEvent targeting {self.target_state}"


class LearnedSalientEvent(SalientEvent):
    def __init__(self, state_set, event_idx, tolerance=0.6, intersection_event=False):
        self.state_set = state_set
        self.classifier = self._classifier_on_state_set()

        SalientEvent.__init__(self, target_state=None, event_idx=event_idx,
                              tolerance=tolerance, intersection_event=intersection_event)

    def is_init_true(self, state):
        position = self._get_position(state)
        return self.classifier.predict(position.reshape(1, -1))

    def batched_is_init_true(self, position_matrix):
        assert isinstance(position_matrix, np.ndarray), type(position_matrix)
        return self.classifier.predict(position_matrix)

    def __eq__(self, other):
        if not isinstance(other, SalientEvent):
            return False
        return self.event_idx == other.event_idx and self.tolerance == other.tolerance

    def _classifier_on_state_set(self):
        positions = np.array([state.position for state in self.state_set])
        classifier = OneClassSVM(nu=0.01, gamma="scale")
        classifier.fit(positions)
        return classifier


class DCOSalientEvent(SalientEvent):
    def __init__(self, covering_option, event_idx, replay_buffer, is_low, tolerance=1.0, intersection_event=False):
        self.covering_option = covering_option
        self.is_low = is_low

        states = replay_buffer.sample(len(replay_buffer))[0]

        goal_states = [s for s in states if not covering_option.is_init_true(s, is_low)]

        goal_values = covering_option.initiation_classifier(covering_option.states_to_tensor(goal_states)).flatten()

        if is_low:
            target_state = goal_states[np.argmin(goal_values)]
        else:
            target_state = goal_states[np.argmax(goal_values)]
        SalientEvent.__init__(self, target_state=target_state, event_idx=event_idx,
                              tolerance=tolerance, intersection_event=intersection_event)

    def is_init_true(self, state):
        return not self.covering_option.is_init_true(state, self.is_low) and SalientEvent.is_init_true(self, state)

    def batched_is_init_true(self, position_matrix):
        return np.logical_and(np.logical_not(self.covering_option.batched_is_init_true(position_matrix, self.is_low)), SalientEvent.batched_is_init_true(self, position_matrix))

    def __eq__(self, other):
        if not isinstance(other, SalientEvent):
            return False
        return self is other

    def __repr__(self):
        return f"DCOSalientEvent with event_idx={self.event_idx}"

    def __hash__(self):
        return self.event_idx
