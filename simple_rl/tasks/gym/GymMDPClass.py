'''
GymMDPClass.py: Contains implementation for MDPs of the Gym Environments.
'''

# Python imports.
import random
import sys
import os
import random

# Other imports.
import gym
from simple_rl.mdp.MDPClass import MDP
from simple_rl.tasks.gym.GymStateClass import GymState
from simple_rl.tasks.gym.wrappers import *


class GymMDP(MDP):
    ''' Class for Gym MDPs '''

    def __init__(self, env_name='CartPole-v0', render=False):
        '''
        Args:
            env_name (str)
        '''
        self.env_name = env_name
        self.env = FrameStack(AtariPreprocessing(gym.make(env_name)), num_stack=4)
        self.render = render

        init_obs = self.env.reset()

        MDP.__init__(self, range(self.env.action_space.n), self._transition_func, self._reward_func,
                     init_state=GymState(init_obs))

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

        self.next_state = GymState(obs, is_terminal=is_terminal)

        if reward < 0:
            return -1.
        if reward > 0:
            return 1.
        return 0.

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
