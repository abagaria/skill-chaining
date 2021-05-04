import ipdb
import numpy as np
from scipy.spatial import distance
from simple_rl.agents.func_approx.dsc.SalientEventClass import SalientEvent
from simple_rl.mdp import MDP, State


class GoalDirectedMDP(MDP):
    def __init__(self, actions, transition_func, reward_func, init_state,
                 salient_positions, task_agnostic, dense_reward, goal_state=None, goal_tolerance=0.6):

        self.task_agnostic = task_agnostic
        self.goal_tolerance = goal_tolerance
        self.goal_state = goal_state
        self.dense_reward = dense_reward
        self.salient_positions = salient_positions + [goal_state] if goal_state is not None else salient_positions

        if not task_agnostic:
            assert self.goal_state is not None, self.goal_state

        self._initialize_salient_events()

        MDP.__init__(self, actions, transition_func, reward_func, init_state)

    def sparse_gc_reward_function(self, state, goal, info):
        curr_pos = self.get_position(state)
        goal_pos = self.get_position(goal)
        done = np.linalg.norm(curr_pos - goal_pos) <= self.goal_tolerance
        time_limit_truncated = info.get('TimeLimit.truncated', False)
        is_terminal = done and not time_limit_truncated
        reward = +100. if is_terminal else -1.
        return reward, is_terminal

    def dense_gc_reward_function(self, state, goal, info={}):
        time_limit_truncated = info.get('TimeLimit.truncated', False)
        curr_pos = self.get_position(state)
        goal_pos = self.get_position(goal)
        distance_to_goal = np.linalg.norm(curr_pos - goal_pos)
        done = distance_to_goal <= self.goal_tolerance
        is_terminal = done and not time_limit_truncated
        reward = +100. if is_terminal else -distance_to_goal / 13.
        return reward, is_terminal

    def batched_sparse_gc_reward_function(self, states, goals):
        assert isinstance(states, np.ndarray)
        assert isinstance(goals, np.ndarray)

        current_positions = states[:, :2]
        goal_positions = goals[:, :2]
        distances = np.linalg.norm(current_positions-goal_positions, axis=1)
        dones = distances <= self.goal_tolerance

        rewards = np.zeros_like(distances)
        rewards[dones==1] = +0.
        rewards[dones==0] = -1.

        return rewards, dones

    def batched_dense_gc_reward_function(self, states, goals):

        current_positions = states[:, :2]
        goal_positions = goals[:, :2]

        distances = np.linalg.norm(current_positions-goal_positions, axis=1)
        dones = distances <= self.goal_tolerance

        assert distances.shape == dones.shape == (states.shape[0],) == (goals.shape[0],)

        rewards = -distances
        rewards[dones==1] = 0.
        
        return rewards, dones

    def _initialize_salient_events(self):
        # Set the current target events in the MDP
        self.current_salient_events = [SalientEvent(pos, event_idx=i + 1) for i, pos in
                                       enumerate(self.salient_positions)]

        # Set an ever expanding list of salient events - we need to keep this around to call is_term_true on trained options
        self.original_salient_events = [event for event in self.current_salient_events]

        # In some MDPs, we use a predicate to determine if we are at the start state of the MDP
        self.start_state_salient_event = SalientEvent(target_state=self.init_state.position, event_idx=0, is_init_event=True)

        # Keep track of all the salient events ever created in this MDP
        self.all_salient_events_ever = [event for event in self.current_salient_events]

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
        pos = self.get_position(state)
        s0 = self.get_start_state_salient_event().get_target_position()
        return np.linalg.norm(pos - s0) <= self.goal_tolerance

    def batched_is_start_state(self, position_matrix):
        s0 = self.get_start_state_salient_event().get_target_position()
        in_start_pos = distance.cdist(position_matrix, s0[None, :]) <= self.goal_tolerance
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
        event_chains_completed = all([chain.is_chain_completed() for chain in event_chains])
        satisfied = len(incoming_chains) > 0 and len(outgoing_chains) and event_chains_completed

        return satisfied

    def get_current_goal(self):
        return self.get_position(self.goal_state)

    def set_current_goal(self, goal):
        self.goal_state = goal

    def is_goal_state(self, state):
        if self.task_agnostic:
            return False

        assert self.goal_state is not None
        state_position = self.get_position(state)
        goal_position = self.get_position(self.goal_state)
        return np.linalg.norm(goal_position - state_position) <= self.goal_tolerance

    def execute_agent_action(self, action, option_idx=None):
        reward, next_state = super(GoalDirectedMDP, self).execute_agent_action(action)
        return reward, next_state

    @staticmethod
    def get_position(state):
        if state is not None:
            position = state.position if isinstance(state, State) else state[:2]
            return position
        return None
