'''
GymMDPClass.py: Contains implementation for MDPs of the Gym Environments.
'''

# Python imports.
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

    def __init__(self, env_name='CartPole-v0', pixel_observation=False,
                 clip_rewards=False, term_func=None, render=False, seed=0):
        '''
        Args:
            env_name (str)
        '''
        self.env_name = env_name

        if pixel_observation:
            self.env = FrameStack(AtariPreprocessing(gym.make(env_name)), num_stack=4)
        else:
            self.env = gym.make(env_name)

        self.env.seed(seed)

        self.clip_rewards = clip_rewards
        self.term_func = term_func
        self.render = render
        self.dense_reward = False

        init_obs = self.env.reset()

        self.game_over = False

        MDP.__init__(self, range(self.env.action_space.n), self._transition_func, self._reward_func,
                     init_state=GymState(init_obs))

    @staticmethod
    def is_goal_state(state):
        return state.is_terminal()

    def is_action_space_discrete(self):
        return hasattr(self.env.action_space, 'n')

    def state_space_size(self):
        return self.env.observation_space.shape

    def action_space_size(self):
        return len(self.actions)

    def _reward_func(self, state, action):
        '''
        Args:
            state (AtariState)
            action (str)

        Returns
            (float)
        '''
        obs, reward, done, info = self.env.step(action)

        self.game_over = done

        if self.render:
            self.env.render()

        if "Monte" in self.env_name:
            position = self.get_player_position()
            goal_cond = 120 <= position[0] <= 140 and position[1] <= 150
            is_terminal = goal_cond
            reward = +10. if goal_cond else 0.
        else:
            is_terminal = self.term_func(obs, reward) if self.term_func is not None else done

        self.next_state = GymState(obs, is_terminal=is_terminal)

        if self.clip_rewards:
            if reward < 0:
                return -1.
            if reward > 0:
                return 1.
            return 0.
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
        init_state_array = self.env.reset()
        self.init_state = GymState(init_state_array, is_terminal=False)
        super(GymMDP, self).reset()

    def __str__(self):
        return "gym-" + str(self.env_name)

    @staticmethod
    def _getIndex(address):
        assert type(address) == str and len(address) == 2
        row, col = tuple(address)
        row = int(row, 16) - 8
        col = int(col, 16)
        return row * 16 + col

    @staticmethod
    def getByte(ram, address):
        # Return the byte at the specified emulator RAM location
        idx = GymMDP._getIndex(address)
        return ram[idx]

    def get_player_position(self):
        ram = self.env.env.ale.getRAM()
        x = int(self.getByte(ram, 'aa'))
        y = int(self.getByte(ram, 'ab'))
        return x, y

    def is_primitive_action(self, action):
        return action in self.actions
