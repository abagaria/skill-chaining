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

    def __init__(self, env_name='CartPole-v0', pixel_observation=False, clip_rewards=False, render=False, seed=0):
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
        self.render = render
        self.dense_reward = False

        init_obs = self.env.reset()
        init_pos = self.get_player_position()

        self.game_over = False

        MDP.__init__(self, range(self.env.action_space.n), self._transition_func, self._reward_func,
                     init_state=GymState(init_obs, position=init_pos))

    def is_action_space_discrete(self):
        return hasattr(self.env.action_space, 'n')

    def state_space_size(self):
        return self.env.observation_space.shape

    def action_space_size(self):
        return len(self.actions)

    @staticmethod
    def term_function_1(state):
        position = state.position
        goal_cond = 120 <= position[0] <= 140 and position[1] <= 150
        return goal_cond

    # TODO: Determine the position bounds for bottom_right
    @staticmethod
    def term_function_2(state):
        position = state.position
        goal_cond = GymMDP.key_condition(position)
        return goal_cond

    @staticmethod
    def key_condition(pos):
        return pos[0] <= 25 and 200 < pos[1] < 220

    @staticmethod
    def bottom_right_condition(position):
        return position[0] <= 25 and position[1] <= 150

    def _reward_func(self, state, action):
        '''
        Args:
            state (AtariState)
            action (str)

        Returns
            (float)
        '''
        obs, _, done, info = self.env.step(action)
        pos = self.get_player_position()

        reward = +10. if self.key_condition(pos) else 0.

        self.game_over = done

        if self.render:
            self.env.render()

        # Goal of Monte for now is to pick up the key
        if "Monte" in self.env_name:
            is_terminal = done or self.key_condition(pos)
        else:
            raise Warning(self.env_name)

        self.next_state = GymState(obs, position=pos, is_terminal=is_terminal)

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
        init_position = self.get_player_position()
        self.init_state = GymState(init_state_array, position=init_position, is_terminal=False)
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
