'''TreasureGameMDPClass.py: Contains implementation for MDPs of the Treasure Game environment'''

# Python imports
import numpy as np
import sys
import os
import random
from collections import defaultdict

# Other imports
import gym
import gym_treasure_game
from simple_rl.mdp.MDPClass import MDP
from simple_rl.tasks.treasure_game.TreasureGameStateClass import TreasureGameState


class TreasureGameMDP(MDP):
    ''' Class for Treasure Game MDPs '''

    def __init__(self, seed, env_name='treasure_game-v0', dense_reward=False, render=False, render_every_n_episodes=0):
        '''
        Args:
            seed (int): Random seed for run.
            env_name (str): Name of Gym environment
            render (bool): If True, renders the screen every time step.
            render_every_n_epsiodes (int): @render must be True, then renders the screen every n episodes.
        '''
        
        self.env_name = env_name
        self.env = gym.make(env_name)
        self.render = render
        self.dense_reward = dense_reward
        self.seed = seed

        self.episode = 0
        self.goal_tolerance = 0.02
        self.prev_reward = 0

        # Set random seed
        random.seed(seed)
        np.random.seed(seed)

        self.reset()
        MDP.__init__(self, range(self.env.action_space.n), self._transition_func, self._reward_func, self.init_state)


    def _reward_func(self, state, action):
        print("TreasureGameMDPClass::_reward_func: state = {}".format(state))
        print("TreasureGameMDPClass::_reward_func: action = {}".format(action))
        # exit()
        next_state, reward, done, _ = self.env.step(action)
        if self.render:
            self.env.render()
        return reward

    def _transition_func(self, state, action):
        return self.next_state

    @staticmethod
    def _get_state(observation, done):
        """ Convert np obs array from gym into a State object. """
        obs = np.copy(observation)
        agent_x, agent_y = obs[:2]
        handle_1_angle, handle_2_angle = obs[2:4]
        key_x, key_y = obs[4:6]
        bolt_locked = obs[6]
        coin_x, coin_y = obs[7:9]

        state = TreasureGameState(agent_x, agent_y, handle_1_angle, handle_2_angle, key_x, key_y, bolt_locked, coin_x, coin_y, done)
        return state

    def execute_agent_action(self, action, option_idx=None):
        reward, next_state = super(TreasureGameMDP, self).execute_agent_action(action)
        return reward, next_state

    def is_goal_state(self, state):
        if isinstance(state, TreasureGameState):
            return state.is_terminal()
        position = state[:2]
        return self.env.is_in_goal_position(position)

    def distance_to_goal(self, position):
        return self.env.distance_to_goal_position(position)

    def state_space_size(self):
        return self.init_state.features().shape[0]

    def action_space_size(self):
        return self.env.action_space.n

    @staticmethod
    def is_primitive_action(action):
        return -1. <= action.all() <= 1.

    def reset(self):
        init_state_array = self.env.reset()
        self.init_state = self._get_state(init_state_array, done=False)
        super(TreasureGameMDP, self).reset()
        self.episode += 1

    def __str__(self):
        return str(self.env_name)