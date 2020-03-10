# Python imports.
import numpy as np
import random
import pdb

# Other imports.
from simple_rl.mdp.MDPClass import MDP
from simple_rl.tasks.point_maze.PointMazeStateClass import PointMazeState
from simple_rl.tasks.point_maze.environments.point_maze_env import PointMazeEnv

class PointMazeMDP(MDP):
    def __init__(self, seed, color_str="", dense_reward=False, render=False):
        self.env_name = "point_maze"
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
            'maze_size_scaling': 3,
            'color_str': color_str
        }
        self.env = PointMazeEnv(**gym_mujoco_kwargs)
        self.goal_position = self.env.goal_xy
        self.key_position = self.env.key_xy
        self.reset()

        # Set the current target events in the MDP
        self.current_target_events = [self.is_in_goal_position, self.is_in_key_position]
        self.current_batched_target_events = [self.env.batched_is_in_goal_position, self.env.batched_is_in_key_position]

        # Set an ever expanding list of salient events - we need to keep this around to call is_term_true on trained options
        self.original_target_events = [self.is_in_goal_position, self.is_in_key_position]
        self.original_batched_target_events = [self.env.batched_is_in_goal_position, self.env.batched_is_in_key_position]

        MDP.__init__(self, [1, 2], self._transition_func, self._reward_func, self.init_state)

    def _reward_func(self, state, action):
        next_state, reward, done, _ = self.env.step(action)
        if self.render:
            self.env.render()
        self.next_state = self._get_state(next_state, done)
        if self.dense_reward:
            return -0.1 * self.distance_to_goal(self.next_state.position)
        return reward + 1.  # TODO: Changing the reward function to return 0 step penalty and 1 reward

    def _transition_func(self, state, action):
        return self.next_state

    @staticmethod
    def _get_state(observation, done):
        """ Convert np obs array from gym into a State object. """
        obs = np.copy(observation)
        position = obs[:2]
        has_key = obs[2]
        theta = obs[3]
        velocity = obs[4:6]
        theta_dot = obs[6]
        # Ignoring obs[7] which corresponds to time elapsed in seconds
        state = PointMazeState(position, has_key, theta, velocity, theta_dot, done)
        return state

    def execute_agent_action(self, action, option_idx=None):
        reward, next_state = super(PointMazeMDP, self).execute_agent_action(action)
        return reward, next_state

    def is_goal_state(self, state):
        if isinstance(state, PointMazeState):
            return state.is_terminal()
        position = state[:2]
        key = state[2]
        return self.env.is_in_goal_position(position) and bool(key)

    def is_in_goal_position(self, state):
        position = state.position if isinstance(state, PointMazeState) else state[:2]
        return self.env.is_in_goal_position(position)

    def is_in_key_position(self, state):
        position = state.position if isinstance(state, PointMazeState) else state[:2]
        return self.env.is_in_key_position(position)

    def distance_to_goal(self, position):
        return self.env.distance_to_goal_position(position)

    def get_current_target_events(self):
        """ Return list of predicate functions that indicate salience in this MDP. """
        return self.current_target_events

    def get_current_batched_target_events(self):
        return self.current_batched_target_events

    def get_original_target_events(self):
        return self.original_target_events

    def get_original_batched_target_events(self):
        return self.original_batched_target_events

    def satisfy_target_event(self, option):
        """
        Once a salient event has both forward and backward options related to it,
        we no longer need to maintain it as a target_event. This function will find
        the salient event that corresponds to the input state and will remove that
        event from the list of target_events.

        :param option: trained option which has potentially satisfied the target event

        """

        if option.chain_id == 3:
            satisfied_goal_salience = option.is_init_true(self.goal_position)
            satisfied_key_salience = option.is_init_true(self.key_position)

            if satisfied_goal_salience and (self.is_in_goal_position in self.current_target_events):
                self.current_target_events.remove(self.is_in_goal_position)
                self.current_batched_target_events.remove(self.env.batched_is_in_goal_position)

            if satisfied_key_salience and (self.is_in_key_position in self.current_target_events):
                self.current_target_events.remove(self.is_in_key_position)
                self.current_batched_target_events.remove(self.env.batched_is_in_key_position)

    @staticmethod
    def state_space_size():
        return 7

    @staticmethod
    def action_space_size():
        return 2

    @staticmethod
    def is_primitive_action(action):
        return -1. <= action.all() <= 1.

    def reset(self):
        init_state_array = self.env.reset()
        self.init_state = self._get_state(init_state_array, done=False)
        super(PointMazeMDP, self).reset()

    def __str__(self):
        return self.env_name
