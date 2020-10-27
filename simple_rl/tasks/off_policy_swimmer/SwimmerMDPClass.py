'''
GymMDPClass.py: Contains implementation for MDPs of the Gym Environments.
'''

# Python imports.
import random
import sys
import os
import random
import numpy as np
import ipdb

# Other imports.
import gym
from simple_rl.mdp.MDPClass import MDP
from simple_rl.tasks.off_policy_swimmer.SwimmerMDPStateClass import SwimmerMDPState


class SwimmerMDP(MDP):
    ''' Class for Gym MDPs '''

    def __init__(self, goal_pos, seed, tolerance, render=False, dense_reward=False):
        '''
        Args:
            env_name (str)
        '''
        random.seed(seed)
        np.random.seed(seed)

        self.env = gym.make("Swimmer-v2")
        self.env_name = "swimmer"
        self.goal_pos = goal_pos
        self.tolerance = tolerance
        self.render = render
        self.dense_reward = dense_reward
        init_state = self.env.reset()
        pos = self.env.sim.data.qpos[:2]
        MDP.__init__(self, range(self.env.action_space.shape[0]), self._transition_func, self._reward_func,
                     init_state=SwimmerMDPState(pos, init_state, False))

    def _reward_func(self, state, action):
        '''
        Args:
            state (AtariState)
            action (str)

        Returns
            (float)
        '''
        obs, _, _, _ = self.env.step(action)
        pos = self.env.sim.data.qpos[:2]
        is_terminal = self.is_goal_state(pos)
        if self.dense_reward:
            reward = np.linalg.norm(self._get_position(state) - self.goal_pos) * -1
        else:
            reward = 10 if is_terminal else -1

        if self.render:
            self.env.render()

        self.next_state = SwimmerMDPState(pos, obs, is_terminal)
        return reward

    def is_goal_state(self, state):
        return np.linalg.norm(state - self.goal_pos) < self.tolerance
        # execute agent action

    @staticmethod
    def state_space_size():
        return 10
    
    @staticmethod
    def action_space_size():
        return 2

    def _transition_func(self, state, action):
        '''
        Args:
            state (AtariState)
            action (str)

        Returns
            (State)
        '''
        return self.next_state

    def reset(self):
        self.env.reset()

    def __str__(self):
        return "gym-" + str(self.env_name)

    @staticmethod
    def _get_position(state):
        position = state.position if isinstance(state, SwimmerMDPState) else state[:2]
        return position
