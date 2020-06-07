"""
LeapWrapperMDPClass.py: Contains implementation for MDPs of the Leap Environments.
https://github.com/vitchyr/multiworld
"""

# Python imports.
import ipdb
import numpy as np

# Other imports.
import gym

from simple_rl.mdp.GoalDirectedMDPClass import GoalDirectedMDP
from simple_rl.tasks.leap_wrapper.LeapWrapperStateClass import LeapWrapperState
from simple_rl.agents.func_approx.dsc.SalientEventClass import SalientEvent

import multiworld
import pdb

# Rendering tests -Kiran
import imageio


class LeapWrapperMDP(GoalDirectedMDP):
    """ Class for Leap Wrapper MDPs """

    def __init__(self, dense_reward=False, render=False):
        self.env_name = "sawyer"
        self.dense_reward = dense_reward
        self.render = render

        if self.render:
            self.movie_width = 600
            self.movie_height = 600
            self.movie_framerate = 240.
            self.movie_timestep = 0
            self.movie_timestep_start = 2000000
            self.movie_timestep_stop = 50000
            self.save_every = 5000

            movie_duration = self.movie_timestep_start - self.movie_timestep_stop
            assert(self.save_every < movie_duration)

            self.empty_movie = np.zeros((
                self.save_every,
                self.movie_width,
                self.movie_height,
                3), dtype=np.uint8)
            self.movie = self.empty_movie.copy()

        self.goal_tolerance = 0.06
        self.salient_tolerance = 0.06

        # Configure env
        multiworld.register_all_envs()
        self.env = gym.make('SawyerPushAndReachArenaEnv-v0', goal_type='puck', dense_reward=False, goal_tolerance=self.goal_tolerance)
        self.goal_state = self.env.get_goal()['state_desired_goal']

        # Sets the initial state
        self.reset()

        # endeff position is ignored by these salient events - just used when plotting initiation_sets
        salient_event_1 = np.zeros(5)
        salient_event_2 = np.zeros(5)

        salient_event_1[3:] = [-0.1, 0.6]
        salient_event_2[3:] = [-0.18, 0.6]

        salient_events = [
            SalientEvent(salient_event_1, 1, name='Puck to goal 1/3',
                         tolerance=self.salient_tolerance, get_relevant_position=get_puck_pos),
            SalientEvent(salient_event_2, 2, name='Puck to goal 2/3',
                         tolerance=self.salient_tolerance, get_relevant_position=get_puck_pos)
        ]

        action_dims = range(self.env.action_space.shape[0])
        GoalDirectedMDP.__init__(self,
                                 action_dims,
                                 self._transition_func,
                                 self._reward_func,
                                 self.init_state,
                                 salient_events,
                                 False,
                                 goal_state=self.goal_state,
                                 goal_tolerance=self.goal_tolerance,
                                 start_tolerance=self.goal_tolerance
                                 )

    def add_frame_to_movie(self):
        if self.movie_timestep_start <= self.movie_timestep < self.movie_timestep_stop:
            if self.movie_timestep == self.movie_timestep_start:
                print("Starting recording")
            frame = self.env.sim.render(camera_name='topview', width=self.movie_width, height=self.movie_height)
            self.movie[self.movie_timestep - self.movie_timestep_start, :, :, :] = frame
        
            if self.movie_timestep == self.movie_timestep_stop:
                clip_number = np.int(np.ceil(self.movie_timestep_stop / self.save_every))
                print(f"Saving clip {clip_number}")
                imageio.mimwrite(f'movie_{clip_number}.mp4', self.movie, fps = self.movie_framerate)

                print("Finishing recording")
                self.render = False

            elif self.movie_timestep > 0 and self.movie_timestep % self.save_every == 0:
                clip_number = self.movie_timestep // self.save_every
                print(f"Saving clip {clip_number}")
                imageio.mimwrite(f'movie_{clip_number}.mp4', self.movie, fps = self.movie_framerate)
                self.movie = self.empty_movie.copy()

        self.movie_timestep += 1

    def _reward_func(self, state, action):
        assert isinstance(action, np.ndarray), type(action)
        next_state, reward, done, _ = self.env.step(action)
        self.next_state = self._get_state(next_state, done)
        ipdb.set_trace()
        if self.render:
            self.add_frame_to_movie()
        return reward

    def _transition_func(self, state, action):
        # References the next state calculated in the reward function
        return self.next_state

    @staticmethod
    def _get_state(observation_dict, done):
        """ Convert observation dict from gym into a State object. """
        observation = observation_dict['observation']
        obs = np.copy(observation)
        endeff_pos = obs[:3]
        puck_pos = obs[3:]

        state = LeapWrapperState(endeff_pos, puck_pos, done)
        return state

    def execute_agent_action(self, action, option_idx=None):
        reward, next_state = super(LeapWrapperMDP, self).execute_agent_action(action)
        return reward, next_state

    def is_goal_state(self, state):
        if isinstance(state, LeapWrapperState):
            return state.is_terminal()
        return self.env.is_goal_state(state)

    @staticmethod
    def state_space_size():
        return 5

    @staticmethod
    def action_space_size():
        return 2

    @staticmethod
    def is_primitive_action(action):
        return -1. <= action.all() <= 1.

    def reset(self):
        init_state_array = self.env.reset()
        self.init_state = self._get_state(init_state_array, done=False)
        super(LeapWrapperMDP, self).reset()

    def __str__(self):
        return self.env_name


def get_endeff_pos(state):
    if isinstance(state, LeapWrapperState):
        return state.endeff_pos
    elif state.ndim == 2:
        return state[:, :3]
    elif state.ndim == 1:
        return state[:3]
    else:
        pdb.set_trace()


def get_xy_endeff_pos(state):
    if isinstance(state, LeapWrapperState):
        return state.endeff_pos[:2]
    elif state.ndim == 2:
        return state[:, :2]
    elif state.ndim == 1:
        return state[:2]
    else:
        pdb.set_trace()


def get_puck_pos(state):
    if isinstance(state, LeapWrapperState):
        return state.puck_pos
    elif state.ndim == 2:
        return state[:, 3:]
    elif state.ndim == 1:
        return state[3:]
    else:
        pdb.set_trace()
