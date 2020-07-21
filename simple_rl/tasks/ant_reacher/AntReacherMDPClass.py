import random
import numpy as np
from copy import deepcopy

from simple_rl.mdp.GoalDirectedMDPClass import GoalDirectedMDP
from simple_rl.tasks.point_maze.environments.ant_maze_env import AntMazeEnv
from simple_rl.tasks.ant_reacher.AntReacherStateClass import AntReacherState


class AntReacherMDP(GoalDirectedMDP):
    def __init__(self, goal_pos, use_hard_coded_events=False, seed=0, render=False, tolerance=0.6, dense_reward=False):
        self.tolerance = tolerance
        self.goal_pos = goal_pos
        self.env_name = "ant-reacher"
        self.use_hard_coded_events = use_hard_coded_events
        self.seed = seed
        self.render = render
        self.dense_reward = dense_reward

        random.seed(seed)
        np.random.seed(seed)

        # Configure env
        gym_mujoco_kwargs = {
            'maze_id': 'Reacher',
            'n_bins': 0,
            'observe_blocks': False,
            'put_spin_near_agent': False,
            'top_down_view': False,
            'manual_collision': True,
            'maze_size_scaling': 2,
            'color_str': ""
        }
        self.env = AntMazeEnv(**gym_mujoco_kwargs)
        self.reset()

        self._determine_x_y_lims()

        salient_positions = []

        if use_hard_coded_events:
            salient_positions = [np.array((4, 4)),
                                 np.array((8, 8)),
                                 np.array((-4, 4)),
                                 np.array((-8, 8)),
                                 np.array((-4, -4)),
                                 np.array((-8, -8)),
                                 np.array((4, -4)),
                                 np.array((8, -8))]

        GoalDirectedMDP.__init__(self, range(self.env.action_space.shape[0]),
                                 self._transition_func,
                                 self._reward_func,
                                 self.init_state,
                                 tolerance,
                                 False,
                                 salient_positions,
                                 task_agnostic=True)

    def _reward_func(self, state, action):

        assert (np.abs(action) <= 1).all()

        action = 30. * action  # Need to scale the actions so they lie between -30 and 30 and not just -1 and 1

        next_state, _, _, _ = self.env.step(action)
        done = self.is_goal_state(next_state)
        if self.render:
            self.env.render()
        self.next_state = self._get_state(next_state, done)

        if self.dense_reward:
            reward = np.linalg.norm(state - self.goal_pos) * -1
        else:
            reward = 10 if done else -1

        return reward

    def is_goal_state(self, state):
        state = state.position if isinstance(state, AntReacherState) else state[:2]
        return np.linalg.norm(state - self.goal_pos) < self.tolerance

    def _transition_func(self, state, action):
        return self.next_state

    def _get_state(self, observation, done):
        """ Convert np obs array from gym into a State object. """
        obs = np.copy(observation)
        position = obs[:2]
        others = obs[2:]
        state = AntReacherState(position, others, done)
        return state

    def state_space_size(self):
        return self.env.observation_space.shape[0]

    def action_space_size(self):
        return self.env.action_space.shape[0]

    def get_init_positions(self):
        return [self.init_state.position]

    def reset(self):
        init_state_array = self.env.reset()
        self.init_state = self._get_state(init_state_array, done=False)
        super(AntReacherMDP, self).reset()

    def set_xy(self, position):
        """ Used at test-time only. """
        position = tuple(position)  # `maze_model.py` expects a tuple
        self.env.wrapped_env.set_xy(position)

        # TODO: obs = self.env._get_obs()
        obs = np.concatenate((np.array(position), self.init_state.features()[2:]), axis=0)
        self.cur_state = self._get_state(obs, done=False)
        self.init_state = deepcopy(self.cur_state)

    def _determine_x_y_lims(self):
        self.xlims = (-5, 5)
        self.ylims = (-5, 5)

    def get_x_y_low_lims(self):
        return self.xlims[0], self.ylims[0]

    def get_x_y_high_lims(self):
        return self.xlims[1], self.ylims[1]

    def sample_random_state(self):
        """ This function only samples a random position from the MDP. """
        low = np.array((self.xlims[0], self.ylims[0]))
        high = np.array((self.xlims[1], self.ylims[1]))
        return np.random.uniform(low=low, high=high)

    def sample_random_action(self):
        size = (self.action_space_size(),)
        return np.random.uniform(-1., 1., size=size)
