import torch
import pfrl
from pfrl import nn as pnn
from pfrl.initializers import init_chainer_default
from pfrl.q_functions import DiscreteActionValueHead


class MLPQFunction(torch.nn.Module):

    def __init__(self, obs_size, n_actions):
        super().__init__()
        self.l1 = torch.nn.Linear(obs_size, 50)
        self.l2 = torch.nn.Linear(50, 50)
        self.l3 = torch.nn.Linear(50, n_actions)

    def forward(self, x):
        h = x
        h = torch.nn.functional.relu(self.l1(h))
        h = torch.nn.functional.relu(self.l2(h))
        h = self.l3(h)
        return pfrl.action_value.DiscreteActionValue(h)


class ConvQFunction(torch.nn.Module):
    def __init__(self, obs_size, n_actions):
        super().__init__()
        self.model = torch.nn.Sequential(
            pnn.LargeAtariCNN(),
            init_chainer_default(torch.nn.Linear(512, n_actions)),
            DiscreteActionValueHead(),
        )
    
    def forward(self, x):
        return self.model.forward(x)
