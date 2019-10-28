import random

from simple_rl.agents.AgentClass import Agent
from simple_rl.agents.func_approx.dqn.replay_buffer import ReplayBuffer
from simple_rl.agents.func_approx.dqn.DQNAgentClass import BATCH_SIZE, BUFFER_SIZE


class DeepRandomAgent(Agent):
    def __init__(self, actions, device, pixel_observation, seed=0, buffer_size=BUFFER_SIZE, batch_size=BATCH_SIZE):
        action_size = len(actions)

        self.epsilon = 1.0
        self.replay_buffer = ReplayBuffer(action_size, buffer_size, batch_size, seed, device, pixel_observation)

        random.seed(seed)
        Agent.__init__(self, "Random-Agent", actions)

    def act(self, state, position, train_mode=True):
        action = random.choice(self.actions)
        return action

    def step(self, state, position, action, reward, next_state, next_position, done, num_steps=1):
        self.replay_buffer.add(state, position, action, reward, next_state, next_position, done, num_steps)

    def update_epsilon(self):
        pass
