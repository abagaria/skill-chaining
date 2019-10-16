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


class NormalizedEnv(gym.ActionWrapper):
    """ Wrap action """

    def action(self, action):
        act_k = (self.action_space.high - self.action_space.low)/ 2.
        act_b = (self.action_space.high + self.action_space.low)/ 2.
        return act_k * action + act_b

    def reverse_action(self, action):
        act_k_inv = 2./(self.action_space.high - self.action_space.low)
        act_b = (self.action_space.high + self.action_space.low)/ 2.
        return act_k_inv * (action - act_b)

class GymMDP(MDP):
    ''' Class for Gym MDPs '''

    def __init__(self, env_name='CartPole-v0', render=False):
        '''
        Args:
            env_name (str)
        '''
        self.env_name = env_name
        self.env = NormalizedEnv(gym.make(env_name))
        self.render = render

        if self.env.action_space.shape == ():
            # If the action space is discrete
            MDP.__init__(self, self.env.action_space, self._transition_func, self._reward_func, init_state=GymState(self.env.reset()))
        else:
            # action space is continuous
            MDP.__init__(self, range(self.env.action_space.shape[0]), self._transition_func, self._reward_func, init_state=GymState(self.env.reset()))

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

    # These functions are required for Skill Chaining.
    def state_space_size(self):
        return self.env.observation_space.shape[0]
    
    def action_space_size(self):
        return self.env.action_space.shape[0]

    def is_goal_state(self, state):
        return state.is_terminal()

    def reset(self):
        self.env.reset()

    def __str__(self):
        return "gym-" + str(self.env_name)
