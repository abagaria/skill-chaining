import ipdb
import random
import torch
import numpy as np
from torch.utils.data import Dataset
from simple_rl.agents.func_approx.dqn.DQNAgentClass import ReplayBuffer as DQNReplayBuffer
from simple_rl.agents.func_approx.ddpg.replay_buffer import ReplayBuffer as DDPGReplayBuffer


class DCODataset(Dataset):

    def __init__(self, replay_buffer, use_xy=True, sub_sample=False):
        assert isinstance(replay_buffer, (DDPGReplayBuffer, DQNReplayBuffer))

        if sub_sample:
            self.transitions = random.sample(replay_buffer.memory, k=1000)
        else:
            self.transitions = replay_buffer.memory

        self.shuffled_transitions = np.copy(self.transitions)
        self.use_xy = use_xy

        assert id(self.transitions) != id(self.shuffled_transitions)
        assert len(self.transitions) == len(self.shuffled_transitions)

        np.random.shuffle(self.shuffled_transitions)

    def __len__(self):
        return len(self.transitions)

    def __getitem__(self, i):
        def _to_tensor(array):
            return torch.from_numpy(array).float()

        def _to_positions(array):
            return array[:2]

        states1, _, _, next_states, *_ = self.transitions[i]
        states2, *_ = self.shuffled_transitions[i]

        assert states1.shape == states2.shape == next_states.shape

        if self.use_xy:
            states1 = _to_positions(states1)
            states2 = _to_positions(states2)
            next_states = _to_positions(next_states)

        states1 = _to_tensor(states1)
        states2 = _to_tensor(states2)
        next_states = _to_tensor(next_states)

        return states1, states2, next_states
