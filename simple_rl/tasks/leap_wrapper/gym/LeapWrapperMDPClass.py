'''
LeapWrapperMDPClass.py: Contains implementation for MDPs of the Leap Environments.
https://github.com/vitchyr/multiworld
'''

# Python imports.
import random
import sys
import os
import random

# Other imports.
import gym
from simple_rl.mdp.MDPClass import MDP
from simple_rl.tasks.leap_wrapper.LeapWrapperStateClass import LeapWrapperState

# Kiran and Kshitij edit
import multiworld
from multiworld.core.flat_goal_env import FlatGoalEnv

class LeapWrapperMDP(MDP):
    ''' Class for Leap Wrapper MDPs '''

    def __init__(self, env_name='SawyerPushAndReachArenaEnv-v0', render=False):
        '''
        Args:
            env_name (str)
        '''
        self.env_name = env_name
        self.base_env = gym.make(self.env_name)
        self.env = FlatGoalEnv(self.base_env)
        self.render = render
        MDP.__init__(self, range(self.env.action_space.n), self._transition_func, self._reward_func, init_state=LeapWrapperState(self.env.reset()))

    def _reward_func(self, state, action):
        '''
        Args:
            state (AtariState)
            action (str)

        Returns
            (float)
        '''
        obs, reward, is_terminal, info = self.env.step(action)

        if self.render:
            self.env.render()

        self.next_state = LeapWrapperState(obs, is_terminal=is_terminal)

        return reward

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
        return "leap-wrapper-" + str(self.env_name)
