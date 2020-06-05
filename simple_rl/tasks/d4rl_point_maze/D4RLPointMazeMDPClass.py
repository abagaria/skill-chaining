import gym
import numpy as np
import random

from simple_rl.mdp.GoalDirectedMDPClass import GoalDirectedMDP
from simple_rl.tasks.point_reacher.PointReacherStateClass import PointReacherState
from simple_rl.tasks.point_maze.environments.point_maze_env import PointMazeEnv


class D4RLPointMazeMDP(GoalDirectedMDP):
    def __init__(self, difficulty, use_hard_coded_events=False,
                 goal_directed=False, init_goal_pos=None, seed=0, render=False):
        assert difficulty in ("medium", "hard")

        self.env_name = f"d4rl-{difficulty}-point-maze"
        self.seed = seed
        self.render = render
        self.difficulty = difficulty
        self.goal_directed = goal_directed

        random.seed(seed)
        np.random.seed(seed)

        if goal_directed: assert init_goal_pos is not None
        self.current_goal_position = init_goal_pos

        maze_id = 'd4rl-maze' if difficulty == "medium" else "d4rl-hard-maze"

        # Configure env
        gym_mujoco_kwargs = {
            'maze_id': maze_id,
            'n_bins': 0,
            'observe_blocks': False,
            'put_spin_near_agent': False,
            'top_down_view': False,
            'manual_collision': True,
            'maze_size_scaling': 3,
        }
        self.env = PointMazeEnv(**gym_mujoco_kwargs)
        self.reset()

        salient_positions = []
        if use_hard_coded_events:
            salient_positions = self._determine_salient_positions()

        self._determine_x_y_lims()
        self.hard_coded_salient_positions = np.copy(salient_positions)

        GoalDirectedMDP.__init__(self, range(self.env.action_space.shape[0]),
                                 self._transition_func,
                                 self._reward_func, self.init_state,
                                 salient_positions, task_agnostic=True, goal_tolerance=0.6)

    def _determine_salient_positions(self):
        if self.difficulty == "medium":
            salient_positions = [np.array((6, 8)),
                                 np.array((5, -5)),
                                 np.array((-7.5, -5)),
                                 np.array((-8.5, 8))]
        elif self.difficulty == "hard":
            salient_positions = [np.array((-15, -10)),
                                 np.array((-15, +10)),
                                 np.array((10, -10)),
                                 np.array((10, 9)),
                                 np.array((-6, +5))]
        else:
            raise NotImplementedError(self.difficulty)

        return salient_positions

    def set_current_goal(self, position):
        self.current_goal_position = position

    def _reward_func(self, state, action):
        next_state, _, done, _ = self.env.step(action)
        if self.render:
            self.env.render()

        # If we are in the goal-directed case, done will be set internally in _get_state
        self.next_state = self._get_state(next_state, done)

        if np.linalg.norm(self.next_state.position - self.current_goal_position) <= 0.6:
            return 1.

        return 0.

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
        goal_dist = self.current_goal_position - position

        if self.goal_directed:
            done = np.linalg.norm(goal_dist) <= 0.6

        state = PointReacherState(position, theta, velocity, theta_dot, done, goal_component=goal_dist)
        return state

    def execute_agent_action(self, action, option_idx=None):
        reward, next_state = super(D4RLPointMazeMDP, self).execute_agent_action(action)
        return reward, next_state

    def state_space_size(self):
        if self.goal_directed:
            return 8
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

        if self.difficulty == "medium":
            xlow, xhigh = -10., 7.5
            ylow, yhigh = -7.5, 10.
        elif self.difficulty == "hard":
            xlow, xhigh = -16, +12
            ylow, yhigh = -10, +10
        else:
            raise NotImplementedError(self.difficulty)

        self.xlims = (xlow, xhigh)
        self.ylims = (ylow, yhigh)

    def get_x_y_low_lims(self):
        return self.xlims[0], self.ylims[0]

    def get_x_y_high_lims(self):
        return self.xlims[1], self.ylims[1]

    def sample_random_state(self):
        """ Rejection sampling from the set of feasible states in the maze. """
        default_choices = self._determine_salient_positions()
        num_tries = 0
        rejected = True
        while rejected and num_tries < 200:
            low = np.array((self.xlims[0], self.ylims[0]))
            high = np.array((self.xlims[1], self.ylims[1]))
            sampled_point = np.random.uniform(low=low, high=high)
            rejected = self.env._is_in_collision(sampled_point)
            num_tries += 1

            if not rejected:
                return sampled_point

        return random.choice(default_choices)

    def sample_random_action(self):
        size = (self.action_space_size(),)
        return np.random.uniform(-1., 1., size=size)
