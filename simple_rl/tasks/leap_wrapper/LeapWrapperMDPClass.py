"""
LeapWrapperMDPClass.py: Contains implementation for MDPs of the Leap Environments.
https://github.com/vitchyr/multiworld
"""

import ipdb
import numpy as np
import gym
import multiworld

from simple_rl.mdp.GoalDirectedMDPClass import GoalDirectedMDP
from simple_rl.tasks.leap_wrapper.LeapWrapperStateClass import LeapWrapperState
from simple_rl.tasks.leap_wrapper.MovieRendererClass import MovieRenderer
from simple_rl.agents.func_approx.dsc.SalientEventClass import SalientEvent


class LeapWrapperMDP(GoalDirectedMDP):
    """ Class for Leap Wrapper MDPs """

    def __init__(self, episode_length, dense_reward=False, render=False, generate_n_clips=10,
                 wait_n_episodes_between_clips=0):
        self.env_name = "sawyer"
        self.dense_reward = dense_reward
        self.render = render

        if self.render:
            self.movie_width = 512
            self.movie_height = 512
            self.movie_renderer = MovieRenderer(
                episode_length,
                self.movie_width,
                self.movie_height,
                3,
                output_folder="movies",
                num_clips=generate_n_clips,
                wait_between_clips=episode_length * wait_n_episodes_between_clips)

        self.goal_tolerance = 0.06
        self.salient_tolerance = 0.06

        # Configure env
        multiworld.register_all_envs()
        self.env = gym.make('SawyerPushAndReachArenaEnv-v0', goal_type='puck', dense_reward=False,
                            goal_tolerance=self.goal_tolerance, goal=(0.15, 0.6, 0.02, -0.25, 0.6))
        self.goal_state = self.env.get_goal()['state_desired_goal']

        # Sets the initial state
        self.reset()

        # self.puck_start = (0., 0.6)
        # self.endeff_x_range = (-0.28, 0.28)
        # self.endeff_y_range = (0.3, 0.9)

        # endeff position is ignored by these salient events - just used when plotting initiation_sets
        salient_event_1 = np.zeros(5)
        salient_event_2 = np.zeros(5)
        salient_event_3 = np.zeros(5)
        salient_event_4 = np.zeros(5)

        salient_event_1[3:] = [-.22, .4]
        salient_event_2[3:] = [.02, .8]
        salient_event_3[3:] = [-.03, 0.35]
        salient_event_4[3:] = [.25, .6]

        salient_events = [
            SalientEvent(salient_event_1, 1, name='Puck goal 1',
                         tolerance=self.salient_tolerance, get_relevant_position=get_puck_pos),
            SalientEvent(salient_event_2, 2, name='Puck goal 2',
                         tolerance=self.salient_tolerance, get_relevant_position=get_puck_pos),
            SalientEvent(salient_event_3, 2, name='Puck goal 3',
                         tolerance=self.salient_tolerance, get_relevant_position=get_puck_pos),
            SalientEvent(salient_event_4, 2, name='Puck goal 4',
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

    def _reward_func(self, state, action):
        assert isinstance(action, np.ndarray), type(action)
        next_state, reward, done, _ = self.env.step(action)
        self.next_state = self._get_state(next_state, done)
        if self.render:
            frame = self.env.sim.render(
                camera_name='topview',
                width=self.movie_width,
                height=self.movie_height)
            self.movie_renderer.add_frame(frame)
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

    def sample_random_action(self):
        size = (self.action_space_size(),)
        return np.random.uniform(-1., 1., size=size)


def get_endeff_pos(state):
    if isinstance(state, LeapWrapperState):
        return state.endeff_pos
    elif state.ndim == 2:
        return state[:, :3]
    elif state.ndim == 1:
        return state[:3]
    else:
        ipdb.set_trace()


def get_xy_endeff_pos(state):
    if isinstance(state, LeapWrapperState):
        return state.endeff_pos[:2]
    elif state.ndim == 2:
        return state[:, :2]
    elif state.ndim == 1:
        return state[:2]
    else:
        ipdb.set_trace()


def get_puck_pos(state):
    if isinstance(state, LeapWrapperState):
        return state.puck_pos
    elif state.ndim == 2:
        return state[:, 3:]
    elif state.ndim == 1:
        return state[3:]
    else:
        ipdb.set_trace()
