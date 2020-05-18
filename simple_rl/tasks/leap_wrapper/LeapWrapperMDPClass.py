"""
LeapWrapperMDPClass.py: Contains implementation for MDPs of the Leap Environments.
https://github.com/vitchyr/multiworld
"""

# Python imports.
import random
import sys
import os
import numpy as np
from scipy.spatial import distance

# Other imports.
import gym
from simple_rl.mdp.MDPClass import MDP
from simple_rl.tasks.leap_wrapper.LeapWrapperStateClass import LeapWrapperState

# Kiran and Kshitij edit
import multiworld
import pdb

class LeapWrapperMDP(MDP):
    ''' Class for Leap Wrapper MDPs '''

    def __init__(self, dense_reward=False, render=False):
        self.env_name = "sawyer"
        self.dense_reward = dense_reward
        self.render = render

        # Configure env
        multiworld.register_all_envs()
        # Assuming that dsc works with gym environments
        self.env = gym.make('SawyerPushAndReachArenaEnv-v0')
        self.puck_goal_pos = self.env._state_goal[3:]
        self.endeff_goal_pos = self.env._state_goal[:3]

        # Will this exist in all gym environments??
        #self.threshold = self.env.indicator_threshold # Default is 0.06
        self.threshold = 0.2

        # Not sure why we do a reset here
        self.reset()

        self.current_target_events = [self.endeff_in_goal_pos, self.puck_in_goal_pos]
        self.current_batched_target_events = [self.batched_endeff_in_goal_pos, self.batched_puck_in_goal_pos]

        # Record the originals, because the currents may expand arbitrarily
        self.original_target_events = [self.endeff_in_goal_pos, self.puck_in_goal_pos]
        self.original_batched_target_events = [self.batched_endeff_in_goal_pos, self.batched_puck_in_goal_pos]

        #MDP.__init__(self, [1, 2], self._transition_func, self._reward_func, self.init_state)
        action_dims = range(self.env.action_space.shape[0])
        MDP.__init__(self, action_dims, self._transition_func, self._reward_func, self.init_state)

    def endeff_in_goal_pos(self, state):
        endeff_pos = state.endeff_pos if isinstance(state, LeapWrapperState) else state[:3]
        return self.pos_dist(endeff_pos, self.endeff_goal_pos) < self.threshold

    def batched_endeff_in_goal_pos(self, position_matrix):
        end_goal_pos = distance.cdist(position_matrix[:, :3], self.endeff_goal_pos[None, :]) < self.threshold
        return end_goal_pos.squeeze(1)

    def puck_in_goal_pos(self, state):
        puck_pos = state.puck_pos if isinstance(state, LeapWrapperState) else state[3:]
        return self.pos_dist(puck_pos, self.puck_goal_pos) < self.threshold

    def batched_puck_in_goal_pos(self, position_matrix):
        puck_goal_pos = distance.cdist(position_matrix[:, 3:], self.puck_goal_pos[None, :]) < self.threshold
        return puck_goal_pos.squeeze(1)

    def get_current_salient_events(self):
        return [self.endeff_goal_pos, self.puck_goal_pos]

    @staticmethod
    def pos_dist(pos1, pos2):
        """ Calculates the norm1 distance between two np.ndarray positions"""
        return np.linalg.norm(pos1 - pos2, ord=1)

    def _reward_func(self, state, action):
        next_state, reward, done, _ = self.env.step(action)
        if self.render:
            self.env.render()
        self.next_state = self._get_state(next_state, done)
        if self.dense_reward:
            # Doesn't this assume we know where the goal state is??
            # TODO: Ask Akhil about how/why this works
            return -0.1 * self.distance_to_goal(next_state)
        return reward + 1.  # TODO: Changing the reward function to return 0 step penalty and 1 reward

    def _transition_func(self, state, action):
        # References the next state calculated in the reward function
        return self.next_state

    @staticmethod
    def _get_state(observation_dict, done):
        """ Convert observation dict from gym into a State object. """
        observation = observation_dict['observation']
        obs = np.copy(observation)
        endeff_pos = obs[:3]
        puck_pos = obs[3:]

        state = LeapWrapperState(endeff_pos, puck_pos, done)
        return state

    def execute_agent_action(self, action, option_idx=None):
        reward, next_state = super(LeapWrapperMDP, self).execute_agent_action(action)
        return reward, next_state

    def is_goal_state(self, state):
        if isinstance(state, LeapWrapperState):
            return state.is_terminal()
        return self.endeff_in_goal_pos(state) and self.puck_in_goal_pos(state)

    def distance_to_goal(self, state):
        return self.pos_dist(state.endeff_pos, self.endeff_goal_pos)

    def get_current_target_events(self):
        """ Return list of predicate functions that indicate salience in this MDP. """
        return self.current_target_events

    def get_current_batched_target_events(self):
        # TODO: Implement batching
        return self.current_batched_target_events

    def get_original_target_events(self):
        return self.original_target_events

    def get_original_batched_target_events(self):
        # TODO: Implement batching
        return self.original_batched_target_events

    def satisfy_target_event(self, option):
        """
        Once a salient event has both forward and backward options related to it,
        we no longer need to maintain it as a target_event. This function will find
        the salient event that corresponds to the input state and will remove that
        event from the list of target_events.

        :param option: trained option which has potentially satisfied the target event

        """

        if option.chain_id == 3:
            satisfied_endeff_salience = option.is_init_true(self.endeff_goal_pos)
            satisfied_puck_salience = option.is_init_true(self.puck_goal_pos)

            if satisfied_endeff_salience and (self.endeff_in_goal_pos in self.current_target_events):
                self.current_target_events.remove(self.endeff_in_goal_pos)
                #self.current_batched_target_events.remove(self.env.batched_endeff_in_goal_pos)

            if satisfied_puck_salience and (self.puck_in_goal_pos in self.current_target_events):
                self.current_target_events.remove(self.puck_in_goal_pos)
                #self.current_batched_target_events.remove(self.env.batched_puck_in_goal_pos)

    @staticmethod
    def state_space_size():
        # Should this reference something instead of hardcoding?
        return 5

    @staticmethod
    def action_space_size():
        # Should this reference something instead of hardcoding?
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

# from multiworld.core.flat_goal_env import FlatGoalEnv

# class LeapWrapperMDP(MDP):
#     ''' Class for Leap Wrapper MDPs '''

#     def __init__(self, env_name='SawyerPushAndReachArenaEnv-v0', render=False):
#         '''
#         Args:
#             env_name (str)
#         '''
#         self.env_name = env_name
#         multiworld.register_all_envs()
#         self.base_env = gym.make(self.env_name)
#         self.env = FlatGoalEnv(self.base_env)
#         self.render = render
#         MDP.__init__(self, range(self.env.action_space.shape[0]), self._transition_func, self._reward_func, init_state=LeapWrapperState(self.env.reset()))

#     def _reward_func(self, state, action):
#         '''
#         Args:
#             state (AtariState)
#             action (str)

#         Returns
#             (float)
#         '''
#         obs, reward, is_terminal, info = self.env.step(action)

#         if self.render:
#             self.env.render()

#         self.next_state = LeapWrapperState(obs, is_terminal=is_terminal)

#         return reward

#     def _transition_func(self, state, action):
#         '''
#         Args:
#             state (AtariState)
#             action (str)

#         Returns
#             (State)
#         '''
#         return self.next_state

#     def reset(self):
#         self.env.reset()

#     def __str__(self):
#         return "leap-wrapper-" + str(self.env_name)