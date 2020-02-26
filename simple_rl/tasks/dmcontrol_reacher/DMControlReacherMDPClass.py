import numpy as np
from dm_control import suite
from dm_control import viewer
from dm_control.suite.wrappers import pixels
import dmc2gym
from simple_rl.tasks.gym.wrappers import *

from simple_rl.mdp.MDPClass import MDP
from simple_rl.tasks.dmcontrol_reacher.DMControlReacherStateClass import DMControlReacherState


class DMControlReacherMDP(MDP):
    def __init__(self, seed, pixel_observation=True, dense_reward=False, env_name='dmcontrol-reacher', render=False, num_stacks=4):
        self.env_name = env_name
        env = dmc2gym.make(domain_name="reacher", task_name="hard", from_pixels=pixel_observation, height=84, width=84, visualize_reward=False)
        env = FrameStack(env, num_stack=num_stacks)
        self.env = env
        self.render = render
        self.dense_reward = dense_reward
        self.seed = seed


        self.action_space = self.env.action_space,
        init_observation = self.env.reset()
        MDP.__init__(self, range(self.env.action_space.shape[0]),
            self._transition_func, self._reward_func,
            init_state=DMControlReacherState(init_observation,
            reward=0, info={'internal_state': [0,0,0,0], 'discount': 1.0}))


    def _reward_func(self, state, action):
        self.env.render()
        obs, gym_reward, _, info = self.env.step(action)

        self.next_state = DMControlReacherState(obs, gym_reward, info)

        return gym_reward


    def _transition_func(self, state, action):
        return self.next_state

    def execute_agent_action(self, action, option_idx=None):
        reward, next_state = super(DMControlReacherMDP, self).execute_agent_action(action)
        return reward, next_state

    def reset(self):
        self.env.reset()
        self.env.render()
        super(DMControlReacherMDP, self).reset()

    def __str__(self):
        return str(self.env.name)

    def state_space_size(self):
        return self.env.observation_space.shape

    def is_action_space_discrete(self):
        return False

    def action_space_size(self):
        return self.env.action_space.shape[0]

    def is_goal_state(self, state):
        if isinstance(state, DMControlReacherState):
            return state.is_terminal()


    @staticmethod
    def is_primitive_action(action):
        return -1. <= action.all() <= 1.
