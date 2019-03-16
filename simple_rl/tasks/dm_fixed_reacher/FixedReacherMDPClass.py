# Python imports.
import numpy as np

# Other imports.
from dm_control import suite
from simple_rl.mdp.MDPClass import MDP
from simple_rl.tasks.dm_fixed_reacher.FixedReacherStateClass import FixedReacherState


class FixedReacherMDP(MDP):
    def __init__(self, seed, difficulty="easy", render=False):
        self.seed = seed
        self.env_name = "reacher"
        self.env = suite.load(self.env_name, difficulty, visualize_reward=True, task_kwargs={"random": seed})
        self.render = render

        MDP.__init__(self, range(self.env.action_spec().minimum.shape[0]), self._transition_func, self._reward_func,
                     init_state=FixedReacherState(self.env.reset().observation))

    def _reward_func(self, state, action):
        '''
        Args:
            state (State)
            action (np.array)

        Returns
            (float)
        '''
        time_limit = self.env.step(action)

        reward = time_limit.reward if time_limit.reward is not None else 0.
        reward = 1. if reward > 0 else -0.001
        observation = time_limit.observation
        done = reward == 1

        # TODO: Figure out how to render in DMCS
        if self.render:
            self.env.render()

        self.next_state = FixedReacherState(observation, is_terminal=done)

        return reward

    def _transition_func(self, state, action):
        '''
        Args:
            state (State)
            action (np.array)

        Returns
            (State)
        '''
        return self.next_state

    def reset(self):
        self.env.reset()

    def __str__(self):
        return "dm_control_suite_" + str(self.env_name)
