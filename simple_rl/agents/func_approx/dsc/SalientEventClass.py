import random
import numpy as np
from simple_rl.mdp.StateClass import State
from scipy.spatial import distance
from sklearn.svm import OneClassSVM
import ipdb


class SalientEvent(object):
    tolerance = None
    state_size = None
    factor_indices = None

    def __init__(self, target_state, event_idx, intersection_event=False, name=None, is_init_event=False):
        """

        Args:
            target_state (np.ndarray): state that is being targeted (full state, not the salient event factors)
            event_idx (int):
            intersection_event (bool):
            get_relevant_position (lambda):
            name (str):
            is_init_event (bool):
        """
        assert self.tolerance is not None
        assert self.state_size is not None
        assert self.factor_indices is not None
        assert isinstance(event_idx, int)

        if isinstance(target_state, State):
            target_state = target_state.features()
        assert isinstance(target_state, np.ndarray)

        self.target_state = target_state
        self.event_idx = event_idx
        self.intersection_event = intersection_event
        self.name = name
        self.is_init_event = is_init_event

        # This is the union of the effect set of all the options targeting this salient event
        self.trigger_points = []
        self._initialize_trigger_points()

    def _initialize_trigger_points(self):
        # TODO: We don't need this after we have our graph checking daemon
        additive_constants = [self.target_state]

        if not self.is_init_event:
            for dim in range(self.state_size):
                offset = np.zeros(self.state_size)
                offset[dim] = self.tolerance
                additive_constants.append(self.target_state + offset)
                additive_constants.append(self.target_state - offset)

        return additive_constants

    def __call__(self, states):
        """

        Args:
            states (np.ndarray or State): this can either be an array representing a single state, or an array
                    representing a batch of states or a State object

        Returns:
            is_satisfied: bool or bool array depending on the shape of states.
        """
        if isinstance(states, State):
            states = states.features()

        if len(states.shape) == 1:
            return self.is_init_true(states)
        else:
            return self.batched_is_init_true(states)

    def __eq__(self, other):
        if not isinstance(other, SalientEvent):
            return False

        return (self.target_state == other.target_state).all() and self.tolerance == other.tolerance
        # self.event_idx == other.event_idx

    def __hash__(self):
        return hash(tuple(self.target_state))

    def is_subset(self, other_event):
        """ I am a subset of `other_event` if all my trigger points are inside `other_event`. """
        assert isinstance(other_event, SalientEvent)
        return other_event.batched_is_init_true(self.trigger_points).all()

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

    def is_init_true(self, state):
        return self.distance_from_goal(state) <= self.tolerance

    def batched_is_init_true(self, position_matrix):
        assert isinstance(position_matrix, np.ndarray), type(position_matrix)
        in_goal_position = self.distance_from_goal(position_matrix) <= self.tolerance
        return in_goal_position.squeeze(1)

    def get_target_position(self):
        return self.target_state

    def __repr__(self):
        if self.name is None:
            return f"SalientEvent targeting {self.target_state}"
        else:
            return f"SalientEvent targeting {self.target_state} | {self.name}"

    def _get_relevant_factors(self, state):
        if isinstance(state, list):
            return [self._get_relevant_factors(x) for x in state]
        elif isinstance(state, State):
            return self._get_relevant_factors(state.features())
        elif isinstance(state, np.ndarray):
            return state[..., self.factor_indices]
        else:
            raise TypeError(f"state was of type {type(state)} but must be a State, np.ndarray, or list")

    def distance_from_goal(self, state):
        factors = self._get_relevant_factors(state)
        target_factors = self._get_relevant_factors(self.target_state)
        if len(state.shape) == 1:
            return np.linalg.norm(target_factors - factors)
        else:
            return distance.cdist(factors, target_factors[None, :])


class LearnedSalientEvent(SalientEvent):
    def __init__(
            self, state_set, event_idx, intersection_event=False, name=None
    ):
        self.state_set = state_set
        self.classifier = self._classifier_on_state_set()

        SalientEvent.__init__(self, target_state=None, event_idx=event_idx,
                              intersection_event=intersection_event, name=name)

    def is_init_true(self, state):
        factors = self._get_relevant_factors(state)
        return self.classifier.predict(factors.reshape(1, -1))

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

    def __repr__(self):
        if self.name is None:
            return f"LearnedSalientEvent targeting {self.target_state}"
        else:
            return f"LearnedSalientEvent targeting {self.target_state} | {self.name}"


class DCOSalientEvent(SalientEvent):
    def __init__(self, covering_option, event_idx, replay_buffer, is_low, intersection_event=False, name=None):
        self.covering_option = covering_option
        self.is_low = is_low

        assert len(replay_buffer), "replay_buffer was empty"
        states = replay_buffer.sample(len(replay_buffer), get_tensor=False)[0]
        values = covering_option.initiation_classifier(states).flatten()

        target_state = states[np.argmin(values) if is_low else np.argmax(values)]

        SalientEvent.__init__(self, target_state=target_state, event_idx=event_idx, intersection_event=intersection_event, name=name)

    def __eq__(self, other):
        if not isinstance(other, SalientEvent):
            return False
        return self is other

    def __repr__(self):
        if self.name is None:
            return f"DCOSalientEvent {self.event_idx} targeting {self.target_state}"
        else:
            return f"DCOSalientEvent {self.event_idx} targeting {self.target_state} | {self.name}"

    def __hash__(self):
        return self.event_idx


class DSCOptionSalientEvent(SalientEvent):
    def __init__(self, option, event_idx):
        """
        Args:
            option (Option)
        """
        self.option = option
        self._initialize_trigger_points()
        SalientEvent.__init__(self,
                              target_state=None,
                              event_idx=event_idx,
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
