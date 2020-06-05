import gym
import ipdb
import d4rl
import numpy as np
import random
from copy import deepcopy

from simple_rl.mdp.GoalDirectedMDPClass import GoalDirectedMDP
from simple_rl.tasks.d4rl_ant_maze.D4RLAntMazeStateClass import D4RLAntMazeState


class D4RLAntMazeMDP(GoalDirectedMDP):
    def __init__(self, maze_size, use_hard_coded_events=False,
                 goal_directed=False, init_goal_pos=None, seed=0, render=False):
        assert maze_size in ("medium", "large"), maze_size
        # self.env_name = f"maze2d-{maze_size}-v0"
        self.env_name = 'antmaze-umaze-v0'
        self.use_hard_coded_events = use_hard_coded_events
        self.goal_directed = goal_directed

        if goal_directed: assert init_goal_pos is not None
        self.current_goal_position = init_goal_pos

        self.env = gym.make(self.env_name)
        self.reset()

        self.render = render
        self.seed = seed

        random.seed(seed)
        np.random.seed(seed)
        self.env.seed(seed)

        salient_positions = []

        if use_hard_coded_events:
            salient_positions = [np.array((4, 0)),
                                 np.array((8, 0)),
                                 np.array((8, 4)),
                                 np.array((8, 8)),
                                 np.array((4, 8)),
                                 np.array((0, 8))]

        self.dataset = self.env.get_dataset()["observations"]
        self._determine_x_y_lims()

        GoalDirectedMDP.__init__(self, range(self.env.action_space.shape[0]),
                                 self._transition_func,
                                 self._reward_func, self.init_state,
                                 salient_positions, task_agnostic=True, goal_tolerance=0.6)

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
        """ Convert np obs array from gym into a State object. """
        ipdb.set_trace()

        obs = np.copy(observation)
        position = obs[:2]
        others = obs[2:]

        goal_dist = self.current_goal_position - position

        if self.goal_directed:
            done = np.linalg.norm(goal_dist) <= 0.6

        state = D4RLAntMazeState(position, others, done, goal_component=goal_dist)
        return state

    def state_space_size(self):
        if self.goal_directed:
            return self.env.observation_space.shape[0] + 2
        return self.env.observation_space.shape[0]

    def action_space_size(self):
        return self.env.action_space.shape[0]

    def get_init_positions(self):
        return [self.init_state.position]

    def reset(self):
        init_state_array = self.env.reset()
        self.init_state = self._get_state(init_state_array, done=False)
        super(D4RLAntMazeMDP, self).reset()

    def set_xy(self, position):
        """ Used at test-time only. """
        position = tuple(position)  # `maze_model.py` expects a tuple
        self.env.env.set_xy(position)
        internal_state = self.env.get_obs()
        obs = np.concatenate((np.array(position), internal_state[2:]), axis=0)
        self.cur_state = self._get_state(obs, done=False)
        self.init_state = deepcopy(self.cur_state)

    # --------------------------------
    # Used for visualizations only
    # --------------------------------

    def _determine_x_y_lims(self):
        observations = self.dataset
        x = [obs[0] for obs in observations]
        y = [obs[1] for obs in observations]
        xlow, xhigh = min(x), max(x)
        ylow, yhigh = min(y), max(y)
        self.xlims = (xlow, xhigh)
        self.ylims = (ylow, yhigh)

    def get_x_y_low_lims(self):
        return self.xlims[0], self.ylims[0]

    def get_x_y_high_lims(self):
        return self.xlims[1], self.ylims[1]

    # ---------------------------------
    # Used during testing only
    # ---------------------------------

    def sample_random_state(self):
        data = self.dataset
        idx = np.random.choice(data.shape[0])
        return data[idx, :][:2]

    def sample_random_action(self):
        size = (self.action_space_size(),)
        return np.random.uniform(-1., 1., size=size)
