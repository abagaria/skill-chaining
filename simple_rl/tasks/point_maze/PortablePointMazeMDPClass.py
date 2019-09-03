# Python imports.
import numpy as np
import random
import pdb

# Other imports.
from simple_rl.mdp.MDPClass import MDP
from simple_rl.tasks.point_maze.PortablePointMazeStateClass import PortablePointMazeState
from simple_rl.tasks.point_maze.environments.point_maze_env import PointMazeEnv

class PortablePointMazeMDP(MDP):
    def __init__(self, seed, train_mode=True, test_mode=False, dense_reward=False, render=False):
        self.env_name = "point_maze"
        self.seed = seed
        self.dense_reward = dense_reward
        self.render = render
        self.train_mode = train_mode
        self.test_mode = test_mode

        # Set random seed
        random.seed(seed)
        np.random.seed(seed)

        # Configure env
        gym_mujoco_kwargs = {
            'maze_id': 'Maze',
            'n_bins': 6,
            'observe_blocks': False,
            'put_spin_near_agent': False,
            'top_down_view': False,
            'manual_collision': True,
            'maze_size_scaling': 2,
            'sensor_range': 3,
            'train_mode': train_mode,
            'test_mode': test_mode
        }
        self.env = PointMazeEnv(**gym_mujoco_kwargs)
        self.goal_position = self.env.goal_xy
        self.reset()

        MDP.__init__(self, [1, 2], self._transition_func, self._reward_func, self.init_state)

    def _reward_func(self, state, action):
        next_global_obs, next_egocentric_obs, reward, done, _ = self.env.step(action)
        if self.render:
            self.env.render()
        self.next_state = self._get_state(next_global_obs, next_egocentric_obs, done)
        return reward

    def _transition_func(self, state, action):
        return self.next_state

    @staticmethod
    def _get_state(pspace_obs, aspace_obs, done):
        """ Convert np obs array from gym into a State object. """
        # Ignoring obs[7] which corresponds to time elapsed in seconds
        state = PortablePointMazeState(pspace_obs[:7], aspace_obs, done)
        return state

    def execute_agent_action(self, action, option_idx=None):
        reward, next_state = super(PortablePointMazeMDP, self).execute_agent_action(action)
        return reward, next_state

    # TODO: This needs to be designed for Agent-Space
    def is_goal_state(self, state):
        if isinstance(state, PortablePointMazeState):
            return state.is_terminal()
        position = state[:2]
        key = state[2]
        return self.env.is_in_goal_position(position) and bool(key)

    def state_space_size(self):
        return 7

    @staticmethod
    def action_space_size():
        return 2

    def action_space_bound(self):
        return 1.

    @staticmethod
    def is_primitive_action(action):
        return -1. <= action.all() <= 1.

    def reset(self):
        init_pspace_obs, init_aspace_obs = self.env.reset()
        self.init_state = self._get_state(init_pspace_obs, init_aspace_obs, done=False)
        super(PortablePointMazeMDP, self).reset()

    def __str__(self):
        return self.env_name
