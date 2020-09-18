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

    def __init__(self, episode_length, use_hard_coded_events, render, dense_reward, generate_n_clips,
                 wait_n_episodes_between_clips, movie_output_folder, task_agnostic):
        self.env_name = "sawyer"
        self.render = render
        dense_reward = False
        salient_tolerance = 0.10

        if self.render:
            self.movie_width = 512
            self.movie_height = 512
            self.movie_renderer = MovieRenderer(
                episode_length,
                self.movie_width,
                self.movie_height,
                3,
                output_folder=f"{movie_output_folder}/movies",
                num_clips=generate_n_clips,
                wait_between_clips=episode_length * wait_n_episodes_between_clips)

        # Configure env
        multiworld.register_all_envs()
        self.env = gym.make('SawyerPushAndReachArenaEnv-v0', goal_type='puck', dense_reward=dense_reward,
                            goal_tolerance=salient_tolerance, task_agnostic=task_agnostic, goal=(0.15, 0.6, 0.02, -0.2, 0.6))
        goal_state = self.env.get_goal()['state_desired_goal']

        # Sets the initial state
        self.reset()

        salient_states = []
        if use_hard_coded_events:
            # endeff position is ignored by these salient events - just used when plotting initiation_sets
            salient_state_1 = np.zeros(5)
            salient_state_2 = np.zeros(5)
            salient_state_1[3:] = [-0.11, 0.6]
            salient_state_2[3:] = [-0.15, 0.6]

            salient_states = [salient_state_1, salient_state_2]

        GoalDirectedMDP.__init__(self,
                                 actions=range(self.env.action_space.shape[0]),  # 2 dimensional: arm delta x and arm delta y
                                 transition_func=self._transition_func,
                                 reward_func=self._reward_func,
                                 init_state=self.init_state,
                                 salient_tolerance=salient_tolerance,
                                 dense_reward=dense_reward,
                                 salient_states=salient_states,
                                 goal_state=goal_state,
                                 task_agnostic=task_agnostic,
                                 init_set_factor_idxs=list(range(5)),
                                 salient_event_factor_idxs=[3, 4]
                                 )

    def __str__(self):
        return self.env_name

    def _reward_func(self, state, action):
        assert isinstance(action, np.ndarray), type(action)
        next_state, reward, done, _ = self.env.step(action)
        self.next_state = self._get_state(next_state, done)
        if self.render and not self.movie_renderer.should_wait():
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

    def is_goal_state(self, state):
        if self.task_agnostic:
            return False
        if isinstance(state, LeapWrapperState):
            return state.is_terminal()
        return self.env.is_goal_state(state)

    def reset(self):
        init_state_array = self.env.reset()
        self.init_state = self._get_state(init_state_array, done=False)
        super(LeapWrapperMDP, self).reset()

    # ---------------------------------
    # Used during testing only
    # ---------------------------------
    def reset_to_state(self, start_state):
        self.env.reset_to_new_start_state(start_pos=start_state)
        self.cur_state = LeapWrapperState(endeff_pos=start_state[:3], puck_pos=start_state[3:], done=False)

    def _sample_random_state(self):
        """The start and goal states are the same for Sawyer, so this function will be used for both."""
        return np.random.uniform(self.env.goal_low, self.env.goal_high)

    def sample_goal_state(self):
        return self._sample_random_state()

    def sample_start_state(self):
        return self._sample_random_state()
