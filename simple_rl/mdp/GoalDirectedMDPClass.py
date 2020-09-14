import numpy as np
from copy import copy
import abc

from scipy.spatial import distance
from simple_rl.agents.func_approx.dsc.SalientEventClass import SalientEvent
from simple_rl.agents.func_approx.dsc.InitiationSetClass import InitiationSet
from simple_rl.mdp import MDP, State


class GoalDirectedMDP(MDP):
    def __init__(self, actions, transition_func, reward_func, init_state, salient_tolerance, dense_reward, salient_states,
                 goal_state, task_agnostic, salient_event_factor_idxs, init_set_factor_idxs):

        """
        Implements MDP functions required for Skill Chaining and Deep Skill Graphs.

        :params:
            actions (int) : action space dimension
            transition_func : the fundamental transition of the mdp
            reward_func : the mdp reward function
            init_state (State) : the start state of the mdp
            salient_tolerance (float) : the tolerance of the goal and for all salients (global variable)
            dense_reward (bool) : True if we want dense reward, False otherwise
            salient_states ([np.ndarray]) : hard-coded states we are targeting (if any) that will be turned into salient events
            goal_state (np.ndarray) : goal state for MDP
            task_agnostic (bool) : True if task_agnostic
            salient_event_factor_idxs (List[int]) : indices of the dimensions of the state that are used to define salient events
            init_set_factor_idxs (List[int]) : indices of the dimensions of the state that are used to define initiation sets
        """
        self.task_agnostic = task_agnostic
        self.goal_state = goal_state
        self.dense_reward = dense_reward
        self._initialize_salient_events(salient_tolerance, init_state, salient_states, salient_event_factor_idxs)
        self._setup_initiation_sets(init_state, init_set_factor_idxs)

        MDP.__init__(self, actions, transition_func, reward_func, init_state)

    # ----------------------
    # -- Abstract methods --
    # ----------------------
    @abc.abstractmethod
    def sample_goal_state(self):
        """
        Returns a random goal state that will become a new salient target for Deep Skill Graphs.
        :return: State
        """
        pass

    @abc.abstractmethod
    def sample_start_state(self):
        """
        Returns a random valid start state that will be used at test-time.
        :return: State
        """
        pass

    @abc.abstractmethod
    def reset_to_state(self, start_state):
        """
        Reset the MDP to the specified start state.
        Args:
            start_state (np.ndarray)
        """
        pass

    def _initialize_salient_events(self, salient_tolerance, init_state, salient_states, salient_event_factor_idxs):
        # setup salient event static variables
        state_size = len(init_state.features())
        SalientEvent.state_size = state_size
        SalientEvent.tolerance = salient_tolerance
        SalientEvent.factor_indices = salient_event_factor_idxs

        salient_events = [SalientEvent(state, i + 1) for i, state in enumerate(salient_states)]

        # Set the current target events in the MDP
        self.current_salient_events = salient_events

        # Set an ever expanding list of salient events - we need to keep this around to call is_term_true on trained options
        self.original_salient_events = copy(salient_events)

        # In some MDPs, we use a predicate to determine if we are at the start state of the MDP
        self.start_state_salient_event = SalientEvent(target_state=init_state,
                                                      event_idx=0,
                                                      name="Start State Salient",
                                                      is_init_event=True)

        # Keep track of all the salient events ever created in this MDP
        self.all_salient_events_ever = copy(salient_events)

        # Make sure that we didn't create multiple copies of the same events
        self._ensure_all_events_are_the_same()

    def _ensure_all_events_are_the_same(self):
        for e1, e2, e3 in zip(self.current_salient_events, self.original_salient_events, self.all_salient_events_ever):
            assert id(e1) == id(e2) == id(e3)

    @staticmethod
    def _setup_initiation_sets(init_state, init_set_factor_idxs):
        state_size = len(init_state.features())
        InitiationSet.state_size = state_size
        InitiationSet.factor_indices = init_set_factor_idxs

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

    def sample_random_action(self):
        size = (self.action_space_size,)
        return np.random.uniform(-1., 1., size=size)
