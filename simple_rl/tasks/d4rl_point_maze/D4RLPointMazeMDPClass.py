import gym
import numpy as np
import random

from simple_rl.mdp.GoalDirectedMDPClass import GoalDirectedMDP
from simple_rl.tasks.point_reacher.PointReacherStateClass import PointReacherState
from simple_rl.tasks.point_maze.environments.point_maze_env import PointMazeEnv


class D4RLPointMazeMDP(GoalDirectedMDP):
    def __init__(self, seed, render=False):
        self.env_name = "d4rl-point-maze"
        self.seed = seed
        self.render = render

        random.seed(seed)
        np.random.seed(seed)

        # Configure env
        gym_mujoco_kwargs = {
            'maze_id': 'd4rl-maze',
            'n_bins': 0,
            'observe_blocks': False,
            'put_spin_near_agent': False,
            'top_down_view': False,
            'manual_collision': True,
            'maze_size_scaling': 3,
        }
        self.env = PointMazeEnv(**gym_mujoco_kwargs)
        self.reset()

        salient_positions = [np.array((6, 8)),
                             np.array((5, -5)),
                             np.array((-7.5, -5)),
                             np.array((-8.5, 8))]

        self._determine_x_y_lims()

        GoalDirectedMDP.__init__(self, range(self.env.action_space.shape[0]),
                                 self._transition_func,
                                 self._reward_func, self.init_state,
                                 salient_positions, task_agnostic=True, goal_tolerance=0.6)

    def _reward_func(self, state, action):
        next_state, _, done, info = self.env.step(action)

        time_limit_truncated = info.get('TimeLimit.truncated', False)
        is_terminal = done and not time_limit_truncated

        if self.task_agnostic:  # No reward function => no rewards and no terminations
            reward = 0.
            is_terminal = False
        else:
            reward = +1. if is_terminal else 0.

        if self.render:
            self.env.render()

        self.next_state = self._get_state(next_state, is_terminal)

        return reward

    def _transition_func(self, state, action):
        return self.next_state

    def _get_state(self, observation, done):
        """ Convert np obs array from gym into a State object. """  # TODO: Adapt has_key
        obs = np.copy(observation)
        position = obs[:2]
        has_key = obs[2]
        theta = obs[3]
        velocity = obs[4:6]
        theta_dot = obs[6]
        # Ignoring obs[7] which corresponds to time elapsed in seconds
        state = PointReacherState(position, theta, velocity, theta_dot, done)
        return state

    def execute_agent_action(self, action, option_idx=None):
        reward, next_state = super(D4RLPointMazeMDP, self).execute_agent_action(action)
        return reward, next_state

    @staticmethod
    def state_space_size():
        return 6

    @staticmethod
    def action_space_size():
        return 2

    def reset(self):
        init_state_array = self.env.reset()
        self.init_state = self._get_state(init_state_array, done=False)
        super(D4RLPointMazeMDP, self).reset()

    def set_xy(self, position):
        self.env.wrapped_env.set_xy(position)
        self.cur_state = self._get_state(np.array((position[0], position[1], 0, 0, 0, 0, 0)), done=False)

    def get_init_positions(self):
        return [self.init_state.position]

    def __str__(self):
        return self.env_name

    def _determine_x_y_lims(self):
        xlow, xhigh = -10., 7.5
        ylow, yhigh = -7.5, 10.
        self.xlims = (xlow, xhigh)
        self.ylims = (ylow, yhigh)

    def get_x_y_low_lims(self):
        return self.xlims[0], self.ylims[0]

    def get_x_y_high_lims(self):
        return self.xlims[1], self.ylims[1]