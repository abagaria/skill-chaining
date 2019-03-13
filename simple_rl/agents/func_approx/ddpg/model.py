# Python imports.
import numpy as np

# PyTorch imports.
import torch
import torch.nn as nn
import torch.nn.functional as F

# Other imports.
from simple_rl.agents.func_approx.ddpg.hyperparameters import *

class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, h1=HIDDEN_1, h2=HIDDEN_2, device=torch.device("cpu")):
        super(Critic, self).__init__()
        self.device = device

        self.linear1 = nn.Linear(state_dim, h1)
        self.linear2 = nn.Linear(h1 + action_dim, h2)
        self.linear3 = nn.Linear(h2, 1)

        self.to(device)

    def forward(self, state, action):
        x = self.linear1(state)
        x = F.relu(x)
        x = self.linear2(torch.cat([x, action], 1))

        x = F.relu(x)
        x = self.linear3(x)

        return x


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, h1=HIDDEN_1, h2=HIDDEN_2, device=torch.device("cpu")):
        super(Actor, self).__init__()
        self.device = device

        self.linear1 = nn.Linear(state_dim, h1)
        self.linear2 = nn.Linear(h1, h2)
        self.linear3 = nn.Linear(h2, action_dim)

        self.to(device)

    def forward(self, state):
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        x = F.tanh(self.linear3(x))
        return x

    def get_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        action = self.forward(state)
        return action.detach().cpu().numpy()[0]


# Based on http://math.stackexchange.com/questions/1287634/implementing-ornstein-uhlenbeck-in-matlab
class OrnsteinUhlenbeckActionNoise:
    def __init__(self, mu=0, sigma=0.2, theta=.15, dt=1e-2, x0=None):
        self.theta = theta
        self.mu = mu
        self.sigma = sigma
        self.dt = dt
        self.x0 = x0
        self.x_prev = self.x0 if self.x0 is not None else np.zeros_like(self.mu)

    def __call__(self):
        x = self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt + \
            self.sigma * np.sqrt(self.dt) * np.random.normal(size=self.mu.shape)
        self.x_prev = x
        return x

    def reset(self):
        self.x_prev = self.x0 if self.x0 is not None else np.zeros_like(self.mu)

    def __repr__(self):
        return 'Ornstein Uhlenbeck Action Noise(mu={}, sigma={})'.format(self.mu, self.sigma)