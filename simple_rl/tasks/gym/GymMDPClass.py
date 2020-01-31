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
# import gym_puddle
from simple_rl.mdp.MDPClass import MDP
from simple_rl.tasks.gym.GymStateClass import GymState
from simple_rl.tasks.gym.wrappers import *
from gym_minigrid.wrappers import *


class GymMDP(MDP):
    ''' Class for Gym MDPs '''

    def __init__(self, env_name='CartPole-v0', pixel_observation=False,
                 clip_rewards=False, step_reward=False, render=False, seed=0):
        '''
        Args:
            env_name (str)
        '''
        self.env_name = env_name

        if "mini" in env_name.lower():  # Visual Grid World from https://github.com/maximecb/gym-minigrid
            assert pixel_observation, "set pixel_observation to true"
            self.env = FrameStack(ImgObsWrapper(RGBImgObsWrapper(gym.make(env_name))), num_stack=1)
        elif pixel_observation:
            # self.env = FrameStack(AtariPreprocessing(gym.make(env_name)), num_stack=4)
            self.env = FrameStack(gym.make(env_name), num_stack=4)
        else:
            self.env = gym.make(env_name)

        self.env.seed(seed)

        self.clip_rewards = clip_rewards
        self.step_reward = step_reward
        self.render = render
        self.pixel_observation = pixel_observation

        init_obs = self.env.reset()
        action_dim = range(self.env.action_space.n) if hasattr(self.env.action_space, "n") else self.env.action_space.shape[0]

        MDP.__init__(self, action_dim, self._transition_func, self._reward_func,
                     init_state=GymState(init_obs))

    def _reward_func(self, state, action):
        '''
        Args:
            state (AtariState)
            action (str)

        Returns
            (float)
        '''
        obs, reward, done, info = self.env.step(action)

        if self.render:
            self.env.render()

        self.next_state = GymState(obs, is_terminal=done)

        if self.clip_rewards:
            if reward < 0:
                return -1.
            if reward > 0:
                return 1.
            return 0.
        elif self.step_reward:  # Mountain-car style reward function
            if reward > 0:
                return 0.
            return -1.
        else:
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
        return "gym-" + str(self.env_name)

