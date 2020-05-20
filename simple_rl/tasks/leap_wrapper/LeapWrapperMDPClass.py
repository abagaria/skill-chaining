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
from simple_rl.mdp.GoalDirectedMDPClass import GoalDirectedMDP
from simple_rl.tasks.leap_wrapper.LeapWrapperStateClass import LeapWrapperState

# Kiran and Kshitij edit
import multiworld
import pdb

class LeapWrapperMDP(GoalDirectedMDP):
    ''' Class for Leap Wrapper MDPs '''

    def __init__(self, dense_reward=False, render=False):
        self.env_name = "sawyer"
        self.dense_reward = dense_reward
        self.render = render

        # Configure env
        multiworld.register_all_envs()
        # Assuming that dsc works with gym environments
        self.env = gym.make('SawyerPushAndReachArenaEnv-v0')
        self.goal_state = self.env._state_goal

        # Will this exist in all gym environments??
        #self.threshold = self.env.indicator_threshold # Default is 0.06
        self.threshold = 0.08

        # Not sure why we do a reset here
        self.reset()

        self.current_target_events = [self.endeff_in_goal_pos, self.puck_in_goal_pos]
        self.current_batched_target_events = [self.batched_endeff_in_goal_pos, self.batched_puck_in_goal_pos]

        # Record the originals, because the currents may expand arbitrarily
        self.original_target_events = [self.endeff_in_goal_pos, self.puck_in_goal_pos]
        self.original_batched_target_events = [self.batched_endeff_in_goal_pos, self.batched_puck_in_goal_pos]

        salient_positions = [
            (self.goal_state, self.get_endeff_pos),
            (self.goal_state, self.get_puck_pos)
        ]

        action_dims = range(self.env.action_space.shape[0])
        GoalDirectedMDP.__init__(self, 
            action_dims, 
            self._transition_func, 
            self._reward_func, 
            self.init_state,
            salient_positions, 
            False, 
            goal_state= self.goal_state, 
            goal_tolerance=self.env.indicator_threshold
        )
    
    @staticmethod
    def get_endeff_pos(state):
        return state.endeff_pos if isinstance(state, LeapWrapperState) else state[:3]

    @staticmethod
    def get_puck_pos(state):
        return state.puck_pos if isinstance(state, LeapWrapperState) else state[3:]

    def _reward_func(self, state, action):
        next_state, reward, done, _ = self.env.step(action)
        if self.render:
            self.env.render()
        self.next_state = self._get_state(next_state, done)
        if self.dense_reward:
            # TODO: Ask Akhil about how/why this works
            return -0.1 * np.linalg.norm(self.next_state - self.goal_state)
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
        return np.linalg.norm(state - self.goal_state) < self.threshold

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