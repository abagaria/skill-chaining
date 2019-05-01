import gym
import numpy as np

from simple_rl.mdp.MDPClass import MDP
from simple_rl.tasks.fixed_reacher.FixedReacherStateClass import FixedReacherState


class FixedReacherMDP(MDP):
    def __init__(self, seed, dense_reward=False, env_name='Reacher-v2', render=False):
        self.env_name = env_name
        self.env = gym.make(env_name)
        self.render = render
        self.dense_reward = dense_reward

        self.seed = seed
        self.env.seed(seed)

        init_observation = self.env.reset()
        self.goal_tolerance = 0.02

        MDP.__init__(self, range(self.env.action_space.shape[0]), self._transition_func, self._reward_func,
                     init_state=FixedReacherState(init_observation, done=False))

    def _reward_func(self, state, action):
        obs, gym_reward, gym_terminal, info = self.env.step(action)
        goal_distance = abs(info["reward_dist"])
        done = goal_distance <= self.goal_tolerance

        if self.dense_reward:
            reward = 0. if done else gym_reward
        else:
            reward = 0. if done else -1.

        if self.render:
            self.env.render()

        self.next_state = FixedReacherState(obs, done)

        return reward

    def _transition_func(self, state, action):
        return self.next_state
    
    def execute_agent_action(self, action, option_idx=None):
        reward, next_state = super(FixedReacherMDP, self).execute_agent_action(action)
        return reward, next_state

    def reset(self):
        self.env.reset()
        super(FixedReacherMDP, self).reset()

    def __str__(self):
        return "gym-" + str(self.env_name)

    def state_space_size(self):
        return self.init_state.features().shape[0]

    def action_space_size(self):
        return self.env.action_space.shape[0]

    @staticmethod
    def action_space_bound():
        return 1.

    def distance_to_goal(self, position):
        distance_vector = position - self.env.get_body_com("target")[:2]
        return np.linalg.norm(distance_vector)

    def is_goal_state(self, state):
        if isinstance(state, FixedReacherState):
            return state.is_terminal()
        position = state[:2]
        return self.distance_to_goal(position) <= self.goal_tolerance

    @staticmethod
    def is_primitive_action(action):
        return -1. <= action.all() <= 1.
