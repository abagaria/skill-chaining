# Python imports.
import numpy as np
import random
import pdb

# Other imports.
from simple_rl.mdp.MDPClass import MDP
from simple_rl.tasks.point_maze.environments.ant_maze_env import AntMazeEnv
from simple_rl.tasks.ant_maze.AntMazeStateClass import AntMazeState

class AntMazeMDP(MDP):
    def __init__(self, seed, dense_reward=False, render=False):
        self.env_name = "ant_maze"
        self.seed = seed
        self.dense_reward = dense_reward
        self.render = render

        # Set random seed
        random.seed(seed)
        np.random.seed(seed)

        # Configure env
        gym_mujoco_kwargs = {
            'maze_id': 'Maze',
            'n_bins': 0,
            'observe_blocks': False,
            'put_spin_near_agent': False,
            'top_down_view': False,
            'manual_collision': True,
            'maze_size_scaling': 2,
            "expose_body_coms": ["torso"]
        }

        self.env = AntMazeEnv(**gym_mujoco_kwargs)
        self.goal_position = self.env.goal_xy
        self.reset()

        # The XML file defines actuation range b/w [-30, 30]
        self.torque_multiplier = 35.

        MDP.__init__(self, range(8), self._transition_func, self._reward_func, self.init_state)

    def _reward_func(self, state, action):
        assert self.is_primitive_action(action)
        next_state, reward, done, _ = self.env.step(self.torque_multiplier * action)
        if self.render:
            self.env.render()
        self.next_state = self._get_state(next_state, done)
        if self.dense_reward:
            return reward + (-0.1 * self.distance_to_goal(self.next_state.position))
        return reward

    def _transition_func(self, state, action):
        return self.next_state

    @staticmethod
    def _get_state(observation, done):
        """ Convert np obs array from gym into a State object. """
        obs = np.copy(observation)
        position = obs[-4:-2]
        other_features = obs[:-4]
        state = AntMazeState(position, other_features, done)
        return state

    def execute_agent_action(self, action, option_idx=None):
        reward, next_state = super(AntMazeMDP, self).execute_agent_action(action)
        return reward, next_state

    def is_goal_state(self, state):
        if isinstance(state, AntMazeState):
            return state.is_terminal()
        position = state[:2]
        return self.env.is_in_goal_position(position)

    def distance_to_goal(self, position):
        return self.env.distance_to_goal_position(position)

    def state_space_size(self):
        return self.init_state.features().shape[0]

    @staticmethod
    def action_space_size():
        return 8

    @staticmethod
    def is_primitive_action(action):
        return -1. <= action.all() <= 1.

    def reset(self):
        init_state_array = self.env.reset()
        self.init_state = self._get_state(init_state_array, done=False)
        super(AntMazeMDP, self).reset()

    def __str__(self):
        return self.env_name
