import torch
import torch.nn as nn


class SpectrumNetwork(nn.Module):
    def __init__(self, state_dim, h1=200, h2=200, device=torch.device("cpu"), seed=0):
        super(SpectrumNetwork, self).__init__()
        self.device = device

        self.linear1 = nn.Linear(state_dim, h1)
        self.linear2 = nn.Linear(h1, h2)
        self.linear3 = nn.Linear(h2, 1)

        self.relu = nn.ReLU()

        self.seed = seed
        torch.manual_seed(seed)

        self.to(device)

    def forward(self, state):
        x = self.relu(self.linear1(state))
        x = self.relu(self.linear2(x))
        x = self.linear3(x)

        return x


class LinearNetwork(nn.Module):
    def __init__(self, state_dim, device=torch.device("cpu"), seed=0):
        super(LinearNetwork, self).__init__()
        self.device = device

        self.linear1 = nn.Linear(state_dim, 1)

        self.seed = seed
        torch.manual_seed(seed)

        self.to(device)

    def forward(self, state):
        x = self.linear1(state)

        return x


class PositionNetwork(nn.Module):
    def __init__(self, state_dim, h1=3, device=torch.device("cpu"), seed=0):
        super(PositionNetwork, self).__init__()
        self.device = device

        self.linear1 = nn.Linear(state_dim, h1)
        self.linear2 = nn.Linear(h1, 1)

        self.relu = nn.ReLU()

        self.seed = seed
        torch.manual_seed(seed)

        self.to(device)

    def forward(self, state):
        x = self.relu(self.linear1(state))
        x = self.linear2(x)

        return x
