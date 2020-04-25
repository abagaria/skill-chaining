'''
GymMDPClass.py: Contains implementation for MDPs of the Gym Environments.
'''

# Python imports.
import numpy as np
import random
import sys
import os
import random
from collections import defaultdict

# Other imports.
import gym
from simple_rl.mdp.MDPClass import MDP
import gym_treasure_game

from simple_rl.tasks.gym.GymStateClass import GymState
from simple_rl.tasks.treasure_game.TreasureGameStateClass import TreasureGameState

class TreasureGameMDP(MDP):
    ''' Class for Gym MDPs '''

    ACTIONS = ["go_left",
               "go_right",
               "up_ladder",
               "down_ladder",
               "down_left",
               "down_right",
               "jump_left",
               "jump_right",
               "interact"]

    def __init__(self, seed, env_name='treasure_game-v0', dense_reward=False, render=False, render_every_n_episodes=0):
        '''
        Args:

        '''
        # self.render_every_n_steps = render_every_n_steps
        self.render_every_n_episodes = render_every_n_episodes
        # self.episode = 0
        # self.env_name = env_name
        # self.env = gym.make(env_name)
        # self.render = render
        # MDP.__init__(self, range(self.env.action_space.n), self._transition_func, self._reward_func, init_state=GymState(self.env.reset()))

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


    def get_parameters(self):
        '''
        Returns:
            (dict) key=param_name (str) --> val=param_val (object).
        '''
        param_dict = defaultdict(int)
        param_dict["env_name"] = self.env_name
   
        return param_dict

    def _reward_func(self, state, action):
        '''
        Args:
            state (AtariState)
            action (str)

        Returns
            (float)
        '''
        return self.prev_reward

    def _transition_func(self, state, action):
        '''
        Args:
            state (AtariState)
            action (str)

        Returns
            (State)
        '''
        # TODO: action needs to be index of the 9 possible actions
        obs, reward, is_terminal, info = self.env.step(action)

        if self.render and (self.render_every_n_episodes == 0 or self.episode % self.render_every_n_episodes == 0):
            self.env.render()

        self.prev_reward = reward
        self.next_state = TreasureGameState(*self.deconstruct_obs(obs), done=is_terminal)

        return self.next_state

    def reset(self):
        init_state_array = self.env.reset()
        self.init_state = self._get_state(init_state_array, done=False)
        super(TreasureGameMDP, self).reset()
        self.episode += 1


    def deconstruct_obs(self, obs):
        obs = np.copy(obs)
        agent_x, agent_y = obs[:2]
        handle_1_angle, handle_2_angle = obs[2:4]
        key_x, key_y = obs[4:6]
        bolt_locked = obs[6]
        coin_x, coin_y = obs[7:9]

        return agent_x, agent_y, handle_1_angle, handle_2_angle, key_x, key_y, bolt_locked, coin_x, coin_y

    def _get_state(self, observation, done):
        """ Convert np obs array from gym into a State object. """
        state = TreasureGameState(*self.deconstruct_obs(observation), done)
        return state

    def state_space_size(self):
        return self.init_state.features().shape[0]

    def action_space_size(self):
        return self.env.action_space.n

    def is_goal_state(self, state):
        if isinstance(state, TreasureGameState):
            return state.is_terminal()
        return self.has_returned_to_start() and self.has_coin()

    def has_coin(self):
        return self.env._env.player_got_goldcoin()
    
    def has_returned_to_start(self):
        return self.env._env.get_player_cell()[1] == 0

    def execute_agent_action(self, action, option_idx=None):
        reward, next_state = super(TreasureGameMDP, self).execute_agent_action(action)
        
        if reward is None:
            reward = 0

        return reward, next_state

    def is_primitive_action(self, action):
        return True if action in list(range(self.env.action_space.n)) else False

    def __str__(self):
        return str(self.env_name)