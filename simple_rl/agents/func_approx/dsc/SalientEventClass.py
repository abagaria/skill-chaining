import numpy as np
from simple_rl.mdp.StateClass import State
from scipy.spatial import distance
from sklearn.svm import OneClassSVM
import ipdb


class SalientEvent(object):
    def __init__(self, target_state, event_idx, tolerance=0.6, intersection_event=False, is_init_event=False):
        self.target_state = target_state
        self.event_idx = event_idx
        self.tolerance = tolerance
        self.intersection_event = intersection_event
        self.is_init_event = is_init_event

        # This is the union of the effect set of all the options targeting this salient event
        self.trigger_points = []
        self._initialize_trigger_points()

        self.revised_by_mpc = False

        assert isinstance(event_idx, int)
        assert isinstance(tolerance, float)

    def _initialize_trigger_points(self):
        self.trigger_points = [self.target_state]

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
               self.tolerance == other.tolerance # and \
               # self.event_idx == other.event_idx

    def __hash__(self):
        return hash(self.event_idx)

    def is_subset(self, other_event):
        """ I am a subset of `other_event` if all my trigger points are inside `other_event`. """
        assert isinstance(other_event, SalientEvent)
        trigger_point_array = self._batched_get_position(self.trigger_points)
        return other_event.batched_is_init_true(trigger_point_array).all()

    def distance_to_other_event(self, other):
        """
        Method to compute the distance between two salient events.
        Overload with more sophisticated metric if needed.

        Args:
            other (SalientEvent)

        Returns:
            distance (float)
        """
        point1 = self.get_target_position()

        if other.get_target_position() is not None:
            point2 = other.get_target_position()
            return self.point_to_point_distance(point1, point2)

        return self.point_to_set_distance(point1, other.trigger_points)

    def distance_to_effect_set(self, effect_set):
        """
        To compute the distance between a point (eg a goal state) and graph vertices,
        we use a conservative distance measure: Measure the distance from that point
        to all the states in the effect set of each option and return the max
        euclidean distance.

        Args:
            effect_set (list): List of State objects

        Returns:
            distance (float)
        """
        point = self._get_position(self.target_state)
        return self.point_to_set_distance(point, effect_set)

    @staticmethod
    def point_to_point_distance(point1, point2):
        return np.linalg.norm(point1 - point2)

    @staticmethod
    def point_to_set_distance(point, state_set):
        assert isinstance(state_set, list)
        assert isinstance(point, np.ndarray)

        point_set = [SalientEvent._get_position(state) for state in state_set]
        point_array = np.array(point_set)
        distances = distance.cdist(point[None, :], point_array)

        return distances.max()

    @staticmethod
    def set_to_set_distance(set1, set2):
        assert isinstance(set1, list)
        assert isinstance(set2, list)

        positions1 = np.array([SalientEvent._get_position(state) for state in set1])
        positions2 = np.array([SalientEvent._get_position(state) for state in set2])
        distances = distance.cdist(positions1, positions2)
        assert distances.shape == (len(set1), len(set2)), distances.shape

        return distances.max()

    def is_init_true(self, state):
        position = self._get_position(state)
        target_position = self._get_position(self.target_state)
        dist = np.linalg.norm(position - target_position)
        return np.round(dist, 8) <= self.tolerance

    def batched_is_init_true(self, position_matrix):
        assert isinstance(position_matrix, np.ndarray), type(position_matrix)
        goal_position = self._get_position(self.target_state)
        distances = distance.cdist(position_matrix, goal_position[None, :])
        in_goal_position = np.round(distances, 8) <= self.tolerance
        return in_goal_position.squeeze(1)

    def get_target_position(self):
        return self._get_position(self.target_state)

    @staticmethod
    def _get_position(state):
        position = state.position if isinstance(state, State) else state[:2]
        assert isinstance(position, np.ndarray), type(position)
        return position

    @staticmethod
    def _batched_get_position(states):
        positions = [SalientEvent._get_position(state) for state in states]
        positions = np.array(positions)
        assert isinstance(positions, np.ndarray), type(positions)
        return positions

    def __repr__(self):
        return f"SalientEvent targeting {self.target_state}"


class LearnedSalientEvent(SalientEvent):
    def __init__(self, state_set, event_idx, tolerance=0.6, intersection_event=False):
        self.state_set = state_set
        self.classifier = self._classifier_on_state_set()

        SalientEvent.__init__(self, target_state=None, event_idx=event_idx,
                              tolerance=tolerance, intersection_event=intersection_event)

    def __repr__(self):
        return f"LearnedSalientEvent {self.event_idx}"

    def __hash__(self):
        return hash(self.event_idx)

    def get_target_position(self):
        return None

    def _initialize_trigger_points(self):
        self.trigger_points = [state.position for state in self.state_set]

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

    def distance_to_effect_set(self, effect_set):
        """ Compute the max distance from `state_set` to `effect_set`. """
        return self.set_to_set_distance(self.state_set, effect_set)

    def distance_to_other_event(self, other):
        if other.get_target_position() is None:
            return self.set_to_set_distance(self.state_set, other.trigger_points)
        return self.point_to_set_distance(other.get_target_position(), self.state_set)


class DCOSalientEvent(SalientEvent):
    def __init__(self, covering_option, event_idx, is_low, tolerance=0.6, intersection_event=False):
        self.covering_option = covering_option
        self.is_low = is_low

        target_state = self.covering_option.min_f_value_state if is_low else self.covering_option.max_f_value_state

        SalientEvent.__init__(self, target_state=target_state, event_idx=event_idx,
                              tolerance=tolerance, intersection_event=intersection_event)

    def __eq__(self, other):
        if not isinstance(other, SalientEvent):
            return False
        return self is other

    def __repr__(self):
        return f"DCOSalientEvent {self.event_idx} targeting {self.target_state}"

    def __hash__(self):
        return self.event_idx

    def distance_to_other_event(self, other):
        if other.get_target_position() is not None:
            return super(DCOSalientEvent, self).distance_to_other_event(other)
        return self.point_to_set_distance(self.get_target_position(), other.trigger_points)


class DSCOptionSalientEvent(SalientEvent):
    def __init__(self, option, event_idx, tolerance=0.6):
        """

        Args:
            option (Option)
        """
        self.option = option
        self._initialize_trigger_points()

        SalientEvent.__init__(self,
                              target_state=None,
                              event_idx=event_idx,
                              tolerance=tolerance,
                              intersection_event=False)

    def _initialize_trigger_points(self):
        self.trigger_points = self.option.effect_set

    def is_init_true(self, state):
        return self.option.is_init_true(state)

    def batched_is_init_true(self, position_matrix):
        self.option.batched_is_init_true(position_matrix)

    def __eq__(self, other):
        return self is other

    def __hash__(self):
        return self.event_idx

    def __repr__(self):
        return f"SalientEvent corresponding to {self.option}"

    def get_target_position(self):
        return None

    @staticmethod
    def _get_position(state):
        return None

    def distance_to_effect_set(self, effect_set):
        return self.set_to_set_distance(self.trigger_points, effect_set)

    def distance_to_other_event(self, other):
        if other.get_target_position() is not None:
            return self.point_to_set_distance(other.get_target_position(), self.trigger_points)
        return self.set_to_set_distance(self.trigger_points, other.trigger_points)

