import cv2
import time
import numpy as np
from simple_rl.mdp.StateClass import State
from scipy.spatial import distance
from sklearn.svm import OneClassSVM
import ipdb
import random
from collections import deque


class SalientEvent(object):
    def __init__(self, target_state, event_idx, predicate=None, tolerance=2.,
                 use_position=False, is_init_event=False):
        self.predicate = predicate
        self.target_state = target_state
        self.event_idx = event_idx
        self.tolerance = tolerance
        self.use_position = use_position
        self.is_init_event = is_init_event

        # This is the union of the effect set of all the options targeting this salient event
        self.trigger_points = deque(maxlen=50)
        self.revised_by_mpc = False

        assert isinstance(event_idx, int)
        assert isinstance(tolerance, float)

    def add_trigger_point(self, trigger_point):
        assert isinstance(trigger_point, State)
        self.trigger_points.append(trigger_point)

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
            return np.allclose(s1, s2)

        if not isinstance(other, SalientEvent):
            return False

        if self.use_position:
            return _state_eq(self.target_state, other.target_state) \
                    and self.tolerance == other.tolerance

        return self.event_idx == other.event_idx

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
        if self.use_position:
            return self._position_based_distance(other)
        return self._emd_based_distance(other)

    def _position_based_distance(self, other):
        point1 = self.get_target_position()

        if other.get_target_position() is not None:
            point2 = other.get_target_position()
            return self.point_to_point_distance(point1, point2)

        return self.point_to_set_distance(point1, other.trigger_points)

    def _emd_based_distance(self, e2):
        g = e2.trigger_points[0].features()
        s = random.choice(self.trigger_points).features()
        d = self.emd_between_images(self.extract(s), self.extract(g))
        return d

    def distance_to_state(self, state):
        return self.distance_to_effect_set([state])

    def sample_from_initiation_region_fast_and_epsilon(self):
        return self.get_target_position()

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
        if self.use_position or len(self.trigger_points) == 0:
            point = self._get_position(self.target_state)
            return self.point_to_set_distance(point, effect_set)

        effect_point = random.choice(effect_set)
        trigger_point = random.choice(self.trigger_points)

        effect_image = self.extract(effect_point.features())
        trigger_image = self.extract(trigger_point.features())

        emd = self.emd_between_images(effect_image, trigger_image)
        return emd

    @staticmethod
    def emd_between_images(image1, image2):
        image1 = cv2.resize(image1, (30, 30), interpolation=cv2.INTER_AREA)
        image2 = cv2.resize(image2, (30, 30), interpolation=cv2.INTER_AREA)
        sig1 = SalientEvent.img_to_sig(image1)
        sig2 = SalientEvent.img_to_sig(image2)
        dist, _, _ = cv2.EMD(sig1, sig2, cv2.DIST_L2)
        return dist

    @staticmethod
    def img_to_sig(arr):
        """ Create a signature of the input image for EMD calculation. """
        sig = np.empty((arr.size, 3), dtype=np.float32)
        count = 0
        for i in range(arr.shape[0]):
            for j in range(arr.shape[1]):
                sig[count] = np.array([arr[i,j], i, j])
                count += 1
        return sig

    @staticmethod
    def extract(i):
        return np.array(i)[-1, ...]

    @staticmethod
    def point_to_point_distance(point1, point2):
        return np.linalg.norm(point1 - point2)

    @staticmethod
    def point_to_set_distance(point, state_set):
        assert isinstance(state_set, (list, deque))
        assert isinstance(point, np.ndarray)

        point_set = [SalientEvent._get_position(state) for state in state_set]
        point_array = np.array(point_set)
        distances = distance.cdist(point[None, :], point_array)

        return distances.max()

    @staticmethod
    def set_to_set_distance(set1, set2):
        assert isinstance(set1, (list, deque))
        assert isinstance(set2, (list, deque))

        positions1 = np.array([SalientEvent._get_position(state) for state in set1])
        positions2 = np.array([SalientEvent._get_position(state) for state in set2])
        distances = distance.cdist(positions1, positions2)
        assert distances.shape == (len(set1), len(set2)), distances.shape

        return distances.max()

    def is_init_true(self, state):
        if self.predicate is not None:
            return self.predicate(state)

        position = self._get_position(state)
        target_position = self._get_position(self.target_state)
        dist = np.linalg.norm(position - target_position)
        return np.round(dist, 8) <= self.tolerance

    def pessimistic_is_init_true(self, state):
        if self.predicate is not None:
            return self.predicate(state)
        return self.is_init_true(state)

    def batched_is_init_true(self, state_set):

        if self.predicate is not None:
            return np.array([self.is_init_true(state) for state in state_set])
        
        position_matrix = self._batched_get_position(state_set)
        assert isinstance(position_matrix, np.ndarray), type(position_matrix)
        goal_position = self._get_position(self.target_state)
        distances = distance.cdist(position_matrix, goal_position[None, :])
        in_goal_position = np.round(distances, 8) <= self.tolerance
        return in_goal_position.squeeze(1)

    def get_target_position(self):
        return self._get_position(self.target_state)

    @staticmethod
    def _get_position(state):
        position = state.get_position() if isinstance(state, State) else state[:2]  # TODO: state[:2] for ant --- automate!
        if not isinstance(position, np.ndarray):
            position = np.array(position)
        return position

    @staticmethod
    def _batched_get_position(states):
        positions = [SalientEvent._get_position(state) for state in states]
        positions = np.array(positions)
        assert isinstance(positions, np.ndarray), type(positions)
        return positions

    def __repr__(self):
        return f"SalientEvent targeting {self.target_state}"
