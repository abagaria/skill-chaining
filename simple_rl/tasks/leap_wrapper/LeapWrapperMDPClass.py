"""
LeapWrapperMDPClass.py: Contains implementation for MDPs of the Leap Environments.
https://github.com/vitchyr/multiworld
"""

import ipdb
import numpy as np
import gym
import multiworld

from simple_rl.mdp.GoalDirectedMDPClass import GoalDirectedMDP
from simple_rl.tasks.leap_wrapper.LeapWrapperStateClass import LeapWrapperState
from simple_rl.agents.func_approx.dsc.SalientEventClass import SalientEvent


class LeapWrapperMDP(GoalDirectedMDP):
    """ Class for Leap Wrapper MDPs """

    def __init__(self, goal_type, dense_reward, task_agnostic):
        self.goal_type = goal_type
        self.env_name = "sawyer"
        self.dense_reward = dense_reward
        salient_tolerance = 0.06

        # Configure env
        multiworld.register_all_envs()
        self.env = gym.make('SawyerPushAndReachArenaEnv-v0', goal_type=goal_type, dense_reward=dense_reward,
                            goal_tolerance=salient_tolerance, task_agnostic=task_agnostic, goal=(0.15, 0.6, 0.02, -0.2, 0.6))
        self.goal_state = self.env.get_goal()['state_desired_goal']

        # Sets the initial state
        self.reset()

        # salient_events = []
        #
        # if use_hard_coded_events:
        #     # endeff position is ignored by these salient events - just used when plotting initiation_sets
        #     salient_event_1 = np.zeros(5)
        #     salient_event_2 = np.zeros(5)
        #     salient_event_1[3:] = [-0.11, 0.6]
        #     salient_event_2[3:] = [-0.15, 0.6]
        #
        #     salient_events = [
        #         SalientEvent(salient_event_1, 1, name='Puck to goal 1/3', get_relevant_position=get_puck_pos),
        #         SalientEvent(salient_event_2, 2, name='Puck to goal 2/3', get_relevant_position=get_puck_pos)
        #     ]

        action_dims = range(self.env.action_space.shape[0])

        # Needs to be defined inside of LeapWrapperMDP because it has a `get_relevant_position` because the start state is only defined
        # by the puck position
        # start_state_salient_event = SalientEvent(target_state=self.init_state.position,
        #                                          event_idx=0)
        GoalDirectedMDP.__init__(self,
                                 actions=action_dims,
                                 transition_func=self._transition_func,
                                 reward_func=self._reward_func,
                                 init_state=self.init_state,
                                 task_agnostic=task_agnostic,
                                 goal_state=self.goal_state,
                                 salient_positions=[],
                                 goal_tolerance = salient_tolerance
                                 )

    def _reward_func(self, state, action):
        assert isinstance(action, np.ndarray), type(action)
        next_state_dict = self.env.step(action)
        self.next_state = self._get_state(next_state_dict)
        if self.dense_reward:
            reward = self.dense_gc_reward_function(self.next_state, self.goal_state)
        else:
            raise NotImplementedError
        return reward

    def _transition_func(self, state, action):
        # References the next state calculated in the reward function
        return self.next_state

    def _get_state(self, observation_dict):
        """ Convert observation dict from gym into a State object. """
        np_state = observation_dict['observation']
        done = self.is_goal_state(np_state)
        endeff_pos = np_state[:3]
        puck_pos = np_state[3:]
        state = LeapWrapperState(endeff_pos, puck_pos, done)
        return state

    def execute_agent_action(self, action, option_idx=None):
        reward, next_state = super(LeapWrapperMDP, self).execute_agent_action(action)
        return reward, next_state

    def set_goal(self, goal):
        self.goal_state = goal
        self.env.set_goal(goal)

    def is_goal_state(self, state):
        if self.task_agnostic:
            return False
        if isinstance(state, LeapWrapperState):
            return state.is_terminal()
        return self.distance_to_goal(state, self.goal_state) < self.goal_tolerance

    @staticmethod
    def state_space_size():
        return 5

    @staticmethod
    def action_space_size():
        return 2

    @staticmethod
    def is_primitive_action(action):
        return -1. <= action.all() <= 1.

    def reset(self):
        init_state_array = self.env.reset()
        self.init_state = self._get_state(init_state_array, done=False)
        super(LeapWrapperMDP, self).reset()

    def __str__(self):
        return self.env_name

    def sample_random_action(self):
        size = (self.action_space_size(),)
        return np.random.uniform(-1., 1., size=size)

    def sample_salient_event(self, episode):
        event_idx = len(self.all_salient_events_ever) + 1
        target_state = np.random.uniform(self.env.goal_low, self.env.goal_high)

        return SalientEvent(target_state=target_state,
                            event_idx=event_idx,
                            get_relevant_position=get_puck_pos,
                            name=f"RRT Salient Episode {episode}")

    def reset_to_start_state(self, start_state):
        self.env.reset_to_new_start_state(start_pos=start_state)
        self.cur_state = LeapWrapperState(endeff_pos=start_state[:3], puck_pos=start_state[3:], done=False)

    def sample_random_state(self):
        return np.random.uniform(self.env.goal_low, self.env.goal_high)

    def get_low_lims(self):
        return self.env.goal_low

    def get_high_lims(self):
        return self.env.goal_high

    def distance_to_goal(self, state, goal):
        state = state.features()
        curr_arm_pos = state[:2]
        curr_puck_pos = state[3:]
        goal_arm_pos = goal[:2]
        goal_puck_pos = goal[3:]
        if self.goal_type == 'puck':
            distance = np.linalg.norm(curr_puck_pos - goal_puck_pos)
        elif self.goal_type == 'hand':
            distance = np.linalg.norm(curr_arm_pos - goal_arm_pos)
        elif self.goal_type == 'complex_puck':
            touch_distance = np.linalg.norm(curr_puck_pos - curr_arm_pos[:2])
            puck_distance = np.linalg.norm(curr_puck_pos - goal_puck_pos)
            TOUCH_THRESHOLD = 0.1
            if touch_distance > TOUCH_THRESHOLD:
                distance = touch_distance
            else:
                distance = puck_distance
        else:
            raise NotImplementedError
        return distance

    def dense_gc_reward_function(self, state, goal):
        distance = self.distance_to_goal(state, goal)
        return -distance if distance > self.goal_tolerance else 0

    def batched_dense_gc_reward_function(self, states, goals):
        curr_puck_pos = states[:, 3:]
        goal_puck_pos = goals[:, 3:]
        curr_arm_pos = states[:, :2]
        goal_arm_pos = goals[:, :2]
        if self.goal_type == 'puck':
            distances = np.linalg.norm(curr_puck_pos - goal_puck_pos, axis=1)
        elif self.goal_type == 'hand':
            distances = np.linalg.norm(curr_arm_pos - goal_arm_pos, axis=1)
        else:
            raise NotImplementedError
        dones = distances <= self.goal_tolerance
        rewards = -distances
        rewards[dones == 1] = 0.
        return rewards, dones
    def sparse_gc_reward_function(self, states, goals):
        raise NotImplementedError


def get_endeff_pos(state):
    if isinstance(state, LeapWrapperState):
        return state.endeff_pos
    elif state.ndim == 2:
        return state[:, :3]
    elif state.ndim == 1:
        return state[:3]
    else:
        ipdb.set_trace()


def get_xy_endeff_pos(state):
    if isinstance(state, LeapWrapperState):
        return state.endeff_pos[:2]
    elif state.ndim == 2:
        return state[:, :2]
    elif state.ndim == 1:
        return state[:2]
    else:
        ipdb.set_trace()


def get_puck_pos(state):
    if isinstance(state, LeapWrapperState):
        return state.puck_pos
    elif state.ndim == 2:
        return state[:, 3:]
    elif state.ndim == 1:
        return state[3:]
    else:
        ipdb.set_trace()
