import numpy as np
from scipy.spatial import distance
from simple_rl.agents.func_approx.dsc.SalientEventClass import SalientEvent
from simple_rl.mdp import MDP, State
from simple_rl.agents.func_approx.dsc.OptionClass import Option

from copy import copy


class GoalDirectedMDP(MDP):
    def __init__(self, actions, transition_func, reward_func, init_state, salient_tolerance,
                 dense_reward, salient_events, task_agnostic, salient_event_factor_indices, init_classifier_factor_indices,
                 goal_state=None, start_salient_event=None):

        """
        :params:
            actions (int) : action space dimension
            transition_func : the fundamental transion of the mdp
            reward_func : the mdp reward function
            init_state (np.ndarray) : the start state of the mdp
            salient_tolerance (float) : the tolerance of the goal and for all salients (global variable)
            dense_reward (bool) : True if we want dense reward, False otherwise
            salient_events ([SalientEvent]) : hard-coded salient events we are targeting (if any)
            task_agnostic (bool) : True if we don't care about the MDP's reward function while training, False otherwise
            salient_event_factor_indices ([int]) : the relevant indices for classifying if a state is in the effect set of a salient event
            init_classifier_factor_indices ([int]) : the relevant indices for classifying if a state is in the initiation set of an Option
            goal_state (np.ndarray) : the center of the effect set of the final salient event that we are shooting towards, after training
            start_salient_event (SalientEvent) : The salient event with an effect set centered around the starting position of the MDP
        """

        self._salient_events = salient_events
        self.task_agnostic = task_agnostic
        self.goal_state = goal_state
        self.dense_reward = dense_reward

        self.salient_tolerance = salient_tolerance
        SalientEvent.tolerance = salient_tolerance
        GoalDirectedMDP.get_salient_event_factors = lambda state: GoalDirectedMDP._get_state_factors(state, salient_event_factor_indices)
        GoalDirectedMDP.get_init_classifier_factors = lambda state: GoalDirectedMDP._get_state_factors(state, init_classifier_factor_indices)

        if not task_agnostic:
            assert self.goal_state is not None, self.goal_state

        self._initialize_salient_events(start_salient_event)

        MDP.__init__(self, actions, transition_func, reward_func, init_state)

    def _initialize_salient_events(self, start_salient_event):
        # Set the current target events in the MDP
        self.current_salient_events = copy(self._salient_events)

        # Set an ever expanding list of salient events - we need to keep this around to call is_term_true on trained options
        self.original_salient_events = copy(self._salient_events)

        # In some MDPs, we use a predicate to determine if we are at the start state of the MDP
        self.start_state_salient_event = start_salient_event if start_salient_event is not None else \
            SalientEvent(target_state=self.init_state.position, event_idx=0, name="Start State Salient")

        # Keep track of all the salient events ever created in this MDP
        self.all_salient_events_ever = copy(self._salient_events)

        # Make sure that we didn't create multiple copies of the same events
        self._ensure_all_events_are_the_same()

    def _ensure_all_events_are_the_same(self):
        for e1, e2, e3 in zip(self.current_salient_events, self.original_salient_events, self.all_salient_events_ever):
            assert id(e1) == id(e2) == id(e3)

    def get_current_target_events(self):
        """ Return list of predicate functions that indicate salience in this MDP. """
        return self.current_salient_events

    def get_original_target_events(self):
        return self.original_salient_events

    def get_all_target_events_ever(self):
        return self.all_salient_events_ever

    def add_new_target_event(self, new_event):
        if new_event not in self.current_salient_events:
            self.current_salient_events.append(new_event)

        if new_event not in self.all_salient_events_ever:
            self.all_salient_events_ever.append(new_event)

    def is_start_state(self, state):
        pos = self._get_position(state)
        s0 = self.init_state.position
        return np.linalg.norm(pos - s0) <= self.salient_tolerance

    def batched_is_start_state(self, position_matrix):
        s0 = self.init_state.position
        in_start_pos = distance.cdist(position_matrix, s0[None, :]) <= self.salient_tolerance
        return in_start_pos.squeeze(1)

    def get_start_state_salient_event(self):
        return self.start_state_salient_event

    def satisfy_target_event(self, chains):
        """
        Once a salient event has both forward and backward options related to it,
        we no longer need to maintain it as a target_event. This function will find
        the salient event that corresponds to the input state and will remove that
        event from the list of target_events. Additionally, we have to ensure that
        the chain corresponding to `option` is "completed".

        A target event is satisfied when all chains targeting it are completed.

        Args:
            chains (list)

        Returns:

        """
        for salient_event in self.get_current_target_events():  # type: SalientEvent
            satisfied_salience = self.should_remove_salient_event_from_mdp(salient_event, chains)
            if satisfied_salience and (salient_event in self.current_salient_events):
                self.current_salient_events.remove(salient_event)

    @staticmethod
    def should_remove_salient_event_from_mdp(salient_event, chains):
        incoming_chains = [chain for chain in chains if chain.target_salient_event == salient_event]
        outgoing_chains = [chain for chain in chains if chain.init_salient_event == salient_event]

        event_chains = incoming_chains + outgoing_chains
        event_chains_completed = all([chain.is_chain_completed(chains) for chain in event_chains])
        satisfied = len(incoming_chains) > 0 and len(outgoing_chains) and event_chains_completed

        return satisfied

    def is_goal_state(self, state):
        if self.task_agnostic:
            return False
        raise NotImplementedError(self.task_agnostic)

    def execute_agent_action(self, action, option_idx=None):
        reward, next_state = super(GoalDirectedMDP, self).execute_agent_action(action)
        return reward, next_state

    @staticmethod
    def _get_position(state):
        position = state.position if isinstance(state, State) else state
        # position = state.position if isinstance(state, State) else state[:2]
        return position

    @staticmethod
    def _get_state_factors(state, factors):
        if isinstance(state, list):
            return [GoalDirectedMDP._get_state_factors(x, factors) for x in state]
        elif isinstance(state, State):
            return GoalDirectedMDP._get_state_factors(state.features(), factors)
        elif isinstance(state, np.ndarray):
            return state[..., factors]
        else:
            raise TypeError(f"state was of type {type(state)} but must be a State, np.ndarray, or list")
