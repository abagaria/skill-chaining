import gym
import d4rl
import numpy as np
import random
from copy import deepcopy

from simple_rl.mdp.GoalDirectedMDPClass import GoalDirectedMDP
from simple_rl.tasks.d4rl_ant_maze.D4RLAntMazeStateClass import D4RLAntMazeState


class D4RLAntMazeMDP(GoalDirectedMDP):
    def __init__(self, maze_size, seed, render=False):
        assert maze_size in ("medium", "large"), maze_size
        # self.env_name = f"maze2d-{maze_size}-v0"
        self.env_name = 'antmaze-umaze-v0'
        self.env = gym.make(self.env_name)
        self.reset()

        self.render = render
        self.seed = seed

        random.seed(seed)
        np.random.seed(seed)
        self.env.seed(seed)

        salient_positions = [np.array((4, 0)),
                             np.array((8, 0)),
                             np.array((8, 4)),
                             np.array((8, 8)),
                             np.array((4, 8)),
                             np.array((0, 8))]
        self.dataset = self.env.get_dataset()
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
        """ Convert np obs array from gym into a State object. """
        obs = np.copy(observation)
        position = obs[:2]
        others = obs[2:]
        state = D4RLAntMazeState(position, others, done)
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
        super(D4RLAntMazeMDP, self).reset()

    def set_xy(self, position):
        """ Used at test-time only. """
        position = tuple(position)  # `maze_model.py` expects a tuple
        self.env.env.set_xy(position)
        obs = np.concatenate((np.array(position), self.init_state.features()[2:]), axis=0)
        self.cur_state = self._get_state(obs, done=False)
        self.init_state = deepcopy(self.cur_state)

    # --------------------------------
    # Used for visualizations only
    # --------------------------------

    def _determine_x_y_lims(self):
        dataset = self.dataset
        observations = dataset["observations"]
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

    def sample_random_state(self, num_samples):
        data = self.dataset
        idx = np.random.choice(data.shape[0], size=num_samples)
        return data[idx, :]
