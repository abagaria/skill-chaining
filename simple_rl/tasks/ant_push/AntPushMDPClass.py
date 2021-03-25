import random
import numpy as np
from copy import deepcopy

from simple_rl.mdp.GoalDirectedMDPClass import GoalDirectedMDP
from simple_rl.tasks.point_maze.environments.ant_maze_env import AntMazeEnv
from simple_rl.tasks.ant_reacher.AntReacherStateClass import AntReacherState


class AntPushMDP(GoalDirectedMDP):
    def __init__(self, goal_state=np.array((0, 13)), seed=0, render=False):
        self.env_name = "ant-four-rooms"
        self.seed = seed
        self.render = render

        random.seed(seed)
        np.random.seed(seed)

        # Configure env
        gym_mujoco_kwargs = {
            'maze_id': 'Push',
            'n_bins': 0,
            'observe_blocks': True,
            'put_spin_near_agent': False,
            'top_down_view': False,
            'manual_collision': True,
            'maze_size_scaling': 6,
            'color_str': ""
        }
        self.env = AntMazeEnv(**gym_mujoco_kwargs)
        self.reset()

        self._determine_x_y_lims()

        GoalDirectedMDP.__init__(self, range(self.env.action_space.shape[0]),
                                 self._transition_func,
                                 self._reward_func,
                                 self.init_state,
                                 salient_positions=[],
                                 task_agnostic=False,
                                 goal_tolerance=0.6,
                                 goal_state=goal_state)

    def _reward_func(self, state, action):

        assert (np.abs(action) <= 1).all()

        action = 30. * action  # Need to scale the actions so they lie between -30 and 30 and not just -1 and 1

        next_state, _, done, info = self.env.step(action)

        if self.dense_reward:
            reward, is_terminal = self.dense_gc_reward_function(next_state, self.get_current_goal(), info)
        else:
            reward, is_terminal = self.sparse_gc_reward_function(next_state, self.get_current_goal(), info)

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
        others = obs[2:]
        state = AntReacherState(position, others, done)
        return state

    def state_space_size(self):
        return self.env.observation_space.shape[0]

    def action_space_size(self):
        return self.env.action_space.shape[0]

    def get_init_positions(self):
        return [self.get_start_state_salient_event().get_target_position()]

    def reset(self):
        init_state_array = self.env.reset()
        self.init_state = self._get_state(init_state_array, done=False)
        super(AntPushMDP, self).reset()

    def set_xy(self, position):
        """ Used at test-time only. """
        position = tuple(position)  # `maze_model.py` expects a tuple
        self.env.wrapped_env.set_xy(position)

        # TODO: obs = self.env._get_obs()
        obs = np.concatenate((np.array(position), self.init_state.features()[2:]), axis=0)
        self.cur_state = self._get_state(obs, done=False)
        self.init_state = deepcopy(self.cur_state)

    def _determine_x_y_lims(self):
        self.xlims = (-8, 2)
        self.ylims = (-1, 15)

    def get_x_y_low_lims(self):
        return self.xlims[0], self.ylims[0]

    def get_x_y_high_lims(self):
        return self.xlims[1], self.ylims[1]

    def in_collision(self, point0):
        def get_n_copies(x, n):
            return [x.copy() for _ in range(n)]
        point1, point2, point3, point4, point5, point6, point7, point8 = get_n_copies(point0, 8)
        point1[0] += 1.
        point2[0] -= 1.
        point3[1] += 1.
        point4[1] += 1.

        # Diagonal points
        point5[0] += 1.; point5[1] += 1.
        point6[0] += 1.; point6[1] -= 1.
        point7[0] -= 1.; point7[1] += 1.
        point8[0] -= 1.; point8[1] -= 1.

        points = (point0, point1, point2, point3, point4, point5, point6, point7, point8)
        for point in points:
            if self.env._is_in_collision(point) or self.env._is_in_collision_with_block(point):
                return True
        return False

    def sample_random_state(self):
        """ This function only samples a random position from the MDP. """
        low = np.array((self.xlims[0], self.ylims[0]))
        high = np.array((self.xlims[1], self.ylims[1]))
        point = np.random.uniform(low=low, high=high)
        while self.is_goal_state(point) or self.in_collision(point):
            point = np.random.uniform(low=low, high=high)
        return point

    def sample_random_action(self):
        size = (self.action_space_size(),)
        return np.random.uniform(-1., 1., size=size)
