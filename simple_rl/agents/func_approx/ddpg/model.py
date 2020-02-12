# Python imports.
import numpy as np
import pdb

# PyTorch imports.
import torch
import torch.nn as nn

# Other imports.
from simple_rl.agents.func_approx.ddpg.hyperparameters import *

class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, h1=HIDDEN_1, h2=HIDDEN_2, device=torch.device("cpu"), seed=0):
        super(Critic, self).__init__()
        self.device = device

        self.linear1 = nn.Linear(state_dim, h1)
        self.linear2 = nn.Linear(h1 + action_dim, h2)
        self.linear3 = nn.Linear(h2, 1)

        self.linear3.weight.data.uniform_(-0.003, 0.003)

        self.relu = nn.ReLU()

        self.seed = seed
        torch.manual_seed(seed)

        self.to(device)

    def forward(self, state, action):
        x = self.relu(self.linear1(state))

        x = self.relu(self.linear2(torch.cat([x, action], 1)))
        x = self.linear3(x)

        return x

    def get_q_value(self, state, action):
        """
        Args:
            state (np.ndarray)
            action (np.ndarray)
        Returns:
            q_value (float)
        """
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        action = torch.FloatTensor(action).unsqueeze(0).to(self.device)
        self.eval()
        with torch.no_grad():
            q_value = self.forward(state, action).item()
        self.train()
        return q_value


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, h1=HIDDEN_1, h2=HIDDEN_2, device=torch.device("cpu"), seed=0):
        super(Actor, self).__init__()
        self.device = device
        self.linear1 = nn.Linear(state_dim, h1)
        self.linear2 = nn.Linear(h1, h2)
        self.linear3 = nn.Linear(h2, action_dim)

        self.linear3.weight.data.uniform_(-0.003, 0.003)

        # We will use batch norm to normalize the input to the tanh non-linearity
        self.norm1 = nn.BatchNorm1d(h1)
        self.norm2 = nn.BatchNorm1d(h2)

        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()

        self.seed = seed
        torch.manual_seed(seed)

        self.to(device)

    def forward(self, state):
        x = self.relu(self.linear1(state))
        x = self.norm1(x)
        x = self.relu(self.linear2(x))
        x = self.norm2(x)
        x = self.tanh(self.linear3(x))
        return x

    def get_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        self.eval()
        action = self.forward(state)
        self.train()
        return action.detach().cpu().numpy()[0]

class ConvCritic(nn.Module):
    def __init__(self, in_channels,  state_dim, action_dim, h1=HIDDEN_1, h2=HIDDEN_2, device=torch.device("cpu"), seed=0):
        super(ConvCritic, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=8, stride=4)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=4, stride=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=4, stride=1)
        self.bn3 = nn.BatchNorm2d(32)

        self.linear1 = nn.Linear(98*64+action_dim, 200)
        self.linear2 = nn.Linear(200, 200)
        self.head = nn.Linear(200, 1)

        self.head.weight.data.uniform_(-0.0003, 0.0003)

        self.relu = nn.ReLU()

        self.seed = seed
        torch.manual_seed(seed)

        self.to(device)

    def forward(self, state, action):
        x = state.float() / 255
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.relu(self.bn3(self.conv3(x)))

        x = x.view(x.size(0), -1)
        x = torch.cat([x, action], 1)
        x = self.relu(self.linear1(x))
        x = self.relu(self.linear2(x))

        return self.head(x)

    def get_q_value(self, state, action):
        """
        Args:
            state (np.ndarray)
            action (np.ndarray)
        Returns:
            q_value (float)
        """
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        action = torch.FloatTensor(action).unsqueeze(0).to(self.device)
        self.eval()
        with torch.no_grad():
            q_value = self.forward(state, action).item()
        self.train()
        return q_value



class ConvActor(nn.Module):
    def __init__(self, in_channels, state_dim, action_dim, h1=HIDDEN_1, h2=HIDDEN_2, device=torch.device("cpu"), seed=0):
        super(ConvActor, self).__init__()
        self.device = device


        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=8, stride=4)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=4, stride=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=4, stride=1)
        self.bn3 = nn.BatchNorm2d(32)

        self.linear1 = nn.Linear(98*64, 200)
        self.linear2 = nn.Linear(200, 200)
        self.head = nn.Linear(200, action_dim)

        self.head.weight.data.uniform_(-0.003, 0.003)

        # We will use batch norm to normalize the input to the tanh non-linearity
        self.norm1 = nn.BatchNorm1d(200)
        self.norm2 = nn.BatchNorm1d(200)

        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()

        self.seed = seed
        torch.manual_seed(seed)

        self.to(device)

    def forward(self, state):
        x = state.float() / 255
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.relu(self.bn3(self.conv3(x)))
        x = self.relu(self.linear1(x.view(x.size(0), -1)))
        x = self.norm1(x)
        x = self.relu(self.linear2(x))
        x = self.norm2(x)
        x = self.tanh(self.head(x))
        return x

    def get_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        self.eval()
        action = self.forward(state)
        self.train()
        return action.detach().cpu().numpy()[0]


# Based on http://math.stackexchange.com/questions/1287634/implementing-ornstein-uhlenbeck-in-matlab
class OrnsteinUhlenbeckActionNoise:
    def __init__(self, mu=0, sigma=0.2, theta=.15, dt=1e-2, x0=None, seed=0):
        self.theta = theta
        self.mu = mu
        self.sigma = sigma
        self.dt = dt
        self.x0 = x0
        self.x_prev = self.x0 if self.x0 is not None else np.zeros_like(self.mu)

        # Set random seed
        self.seed = seed
        np.random.seed(seed)

    def __call__(self):
        x = self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt + \
            self.sigma * np.sqrt(self.dt) * np.random.normal(size=self.mu.shape)
        self.x_prev = x
        return x

    def reset(self):
        self.x_prev = self.x0 if self.x0 is not None else np.zeros_like(self.mu)

    def __repr__(self):
        return 'Ornstein Uhlenbeck Action Noise(mu={}, sigma={})'.format(self.mu, self.sigma)
