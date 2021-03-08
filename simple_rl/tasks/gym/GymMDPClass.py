'''
GymMDPClass.py: Contains implementation for MDPs of the Gym Environments.
'''

# Python imports.
import sys
import os
import random

# Other imports.
import gym
import cv2
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
            self.env = make_atari("MontezumaRevengeNoFrameskip-v4")
            self.env = wrap_deepmind(self.env, episode_life=False, clip_rewards=False, frame_stack=True)
        else:
            self.env = gym.make(env_name)

        self.env.seed(seed)

        self.clip_rewards = clip_rewards
        self.term_func = term_func
        self.render = render
        self.dense_reward = False

        init_obs = self.env.reset()

        self.game_over = False

        # self.spawn_states = [(77, 250), (88, 235), (77, 215), (75, 192), (84, 192), (109,192), (109,199), (132,192),
        #                      (138,192), (139,192), (130,192), (133,173), (133,162), (133,151), (133,148), (123,148),
        #                      (114,148), (99, 148), (75, 148), (62, 148), (50, 148), (38, 148), (25, 148), (20, 148),
        #                      (21, 152), (21, 164), (21, 177), (21, 192), (10, 192)]
        self.spawn_states = [(77, 235), (130, 192), (123, 148), (20, 148), (21, 192), (114,148), (99, 148), (62, 148), (50, 148), (38, 148), (25, 148)]
        # self.spawn_states = [(99, 148)]  # Just to the right of the skull (when fixed)

        MDP.__init__(self, range(self.env.action_space.n), self._transition_func, self._reward_func,
                     init_state=GymState(image=init_obs, position=self.get_player_position(), ram=self.env.env.ale.getRAM()))

    def is_goal_state(self, state):
        return self.has_key(state) and not self.is_dead(state.ram)

    def has_key(self, state):
        return int(self.getByte(state.ram, 'c1')) != 0

    def falling(self, state):
        return int(self.getByte(state.ram, 'd8')) != 0

    def is_skull_moving(self, state):
        return int(self.getByte(state.ram, 'c3')) != 0

    def is_action_space_discrete(self):
        return hasattr(self.env.action_space, 'n')

    def state_space_size(self):
        return self.env.observation_space.shape

    def action_space_size(self):
        return len(self.actions)

    def sparse_gc_reward_function(self, state, goal, info={}, tol=2):
        assert isinstance(state, GymState), f"{type(state)}"
        assert isinstance(goal, GymState), f"{type(goal)}"

        def is_close(pos1, pos2):
            return abs(pos1[0] - pos2[0]) <= tol and abs(pos1[1] - pos2[1]) <= tol

        state_pos = state.get_position()
        goal_pos = goal.get_position()

        done = is_close(state_pos, goal_pos)
        reward = +1. if done else 0.

        return reward, done

    def _reward_func(self, state, action):
        '''
        Args:
            state (AtariState)
            action (str)
        Returns
            (float)
        '''
        obs, reward, done, info = self.env.step(action)
        ram = self.env.env.ale.getRAM()

        self.game_over = self.get_player_lives(ram) != 5

        if self.render:
            self.env.render()

        position = self.get_player_position()
        goal_cond = int(self.getByte(ram, 'c1')) != 0  # TODO: Restore for key
        # goal_cond = position[0] <= 62 and position[1] <= 148  # Position just to the left of the skull
        is_terminal = goal_cond or self.game_over

        # reward = -1. if self.game_over else reward  # Negative rewards for death
        reward = +1. if goal_cond else reward
        reward = np.sign(reward)  # Reward clipping

        self.next_state = GymState(image=obs, position=position, ram=ram, is_terminal=is_terminal)

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
        self.remove_skull()

        for _ in range(4):
            obs, _, _, _ = self.env.step(0) # no-op to get agent onto ground

        ram = self.env.env.ale.getRAM()
        self.init_state = GymState(image=obs, position=self.get_player_position(), ram=ram, is_terminal=False)
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

    def get_player_lives(self, ram):
        return int(self.getByte(ram, 'ba'))

    def is_dead(self, ram):
        return int(self.getByte(ram, 'ba')) != 5

    def get_player_position(self):
        ram = self.env.env.ale.getRAM()
        x = int(self.getByte(ram, 'aa'))
        y = int(self.getByte(ram, 'ab'))
        return x, y

    @staticmethod
    def get_skull_position(ram):
        skull_x = int(GymMDP.getByte(ram, 'af')) + 33
        return skull_x

    def set_player_position(self, x, y):
        state_ref = self.env.env.ale.cloneState()
        state = self.env.env.ale.encodeState(state_ref)
        self.env.env.ale.deleteState(state_ref)

        state[331] = x
        state[335] = y

        new_state_ref = self.env.env.ale.decodeState(state)
        self.env.env.ale.restoreState(new_state_ref)
        self.env.env.ale.deleteState(new_state_ref)
        self.execute_agent_action(0) # NO-OP action to update the RAM state

    def remove_skull(self):
        print("Setting skull position")
        state_ref = self.env.env.ale.cloneState()
        state = self.env.env.ale.encodeState(state_ref)
        self.env.env.ale.deleteState(state_ref)
        
        state[431] = 1
        state[351] = 40

        new_state_ref = self.env.env.ale.decodeState(state)
        self.env.env.ale.restoreState(new_state_ref)
        self.env.env.ale.deleteState(new_state_ref)
        self.execute_agent_action(0) # NO-OP action to update the RAM state

    def saveImage(self, path):
        cv2.imwrite(f"{path}.png", self.cur_state.image[-1,:,:])

    def is_primitive_action(self, action):
        return action in self.actions

    def sample_random_action(self):
        return random.choice(self.actions)

    def get_x_y_low_lims(self):
        return 0, 100

    def get_x_y_high_lims(self):
        return 140, 300