# Python imports.
from collections import deque
import random
import numpy as np

# Other imports.
from simple_rl.agents.func_approx.ddpg.hyperparameters import BUFFER_SIZE, BATCH_SIZE

class ReplayBuffer(object):
    def __init__(self, buffer_size=BUFFER_SIZE, name_buffer=''):
        self.buffer_size = buffer_size
        self.num_exp = 0
        self.buffer = deque(maxlen=buffer_size)
        self.name = name_buffer

    def add(self, state, action, reward, next_state, terminal):
        experience = state, action, reward, next_state, terminal
        self.buffer.append(experience)
        self.num_exp += 1

    def size(self):
        return self.buffer_size

    def __len__(self):
        return self.num_exp

    def sample(self, batch_size=BATCH_SIZE):
        if self.num_exp < batch_size:
            batch = random.sample(self.buffer, self.num_exp)
        else:
            batch = random.sample(self.buffer, batch_size)

        state, action, reward, next_state, terminal = map(np.stack, zip(*batch))

        return state, action, reward, next_state, terminal

    def clear(self):
        self.buffer = deque(maxlen=self.buffer_size)
        self.num_exp = 0
