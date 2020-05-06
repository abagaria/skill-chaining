import gym
import d4rl
import numpy as np
import random

from simple_rl.mdp.GoalDirectedMDPClass import GoalDirectedMDP
from simple_rl.tasks.d4rl_point_maze.D4RLPointMazeStateClass import D4RLPointMazeState


class D4RLPointMazeMDP(GoalDirectedMDP):
    def __init__(self, maze_size, seed, render=False):
        assert maze_size in ("medium", "large"), maze_size
        self.env_name = f"maze2d-{maze_size}-v0"
        self.env = gym.make(self.env_name)
        self.reset()

        self.render = render
        self.seed = seed

        random.seed(seed)
        np.random.seed(seed)
        self.env.seed(seed)

        salient_positions = [np.array((1, 6)), np.array((6, 5)), np.array((1, 1)), np.array((6, 1))]

        GoalDirectedMDP.__init__(self, [1, 2], self._transition_func, self._reward_func, self.init_state,
                                 salient_positions, task_agnostic=True, goal_tolerance=0.6)

    def _reward_func(self, state, action):
        next_state, _, done, info = self.env.step(action)

        time_limit_truncated = info.get('TimeLimit.truncated', False)
        is_terminal = done and not time_limit_truncated

        if self.task_agnostic:
            reward = 0.
        else:
            reward = +1. if is_terminal else 0.

        if self.render:
            self.env.render()

        self.next_state = self._get_state(next_state, is_terminal)

        return reward

    def _transition_func(self, state, action):
        return self.next_state

    def _get_state(self, observation, done):
        """ Convert np obs array from gym into a State object. """
        obs = np.copy(observation)
        position = obs[:2]
        velocity = obs[2:4]
        state = D4RLPointMazeState(position, velocity, done)
        return state

    def state_space_size(self):
        return self.env.observation_space.shape[0]

    def action_space_size(self):
        return self.env.action_space.shape[0]

    @staticmethod
    def is_primitive_action(action):
        return -1. <= action.all() <= 1.

    def get_init_positions(self):
        return [self.init_state.position]

    def reset(self):
        init_state_array = self.env.reset()
        self.init_state = self._get_state(init_state_array, done=False)
        super(D4RLPointMazeMDP, self).reset()

    def set_xy(self, position):
        position = tuple(position)  # `maze_model.py` expects a tuple
        self.env.env.reset_to_location(position)
        self.cur_state = self._get_state(np.array((position[0], position[1], 0, 0, 0, 0, 0)), done=False)

    # Used for visualizations only:
    @staticmethod
    def get_x_y_low_lims():
        return 0., 0.

    @staticmethod
    def get_x_y_high_lims():
        return 6.5, 6.5
