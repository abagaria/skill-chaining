'''
GymMDPClass.py: Contains implementation for MDPs of the Gym Environments.
'''

# Python imports.
import random
import sys
import os
import random
import numpy as np

# Other imports.
import gym
from simple_rl.mdp.MDPClass import MDP
from simple_rl.tasks.gym.GymStateClass import GymState


class SwimmerMDP(MDP):
    ''' Class for Gym MDPs '''

    def __init__(self, goal_state, tolerance, render=False):
        '''
        Args:
            env_name (str)
        '''
        self.env = gym.make("Swimmer-v2")
        self.render = render
        MDP.__init__(self, range(self.env.action_space.shape[0]), self._transition_func, self._reward_func, init_state=GymState(self.env.reset()))

    def _reward_func(self, state, action):
        '''
        Args:
            state (AtariState)
            action (str)

        Returns
            (float)
        '''
        obs, _, _, info = self.env.step(action)
        pos = self.env.sim.data.qpos[:2]
        done = self.is_goal_state(pos)



        if self.render:
            self.env.render()

        self.next_state = GymState(obs, is_terminal=is_terminal)
        return 1. if done else -1

    def is_goal_state(self, state):
        return np.linalg.norm(state - self.goal_pos) < self.tolerance

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
