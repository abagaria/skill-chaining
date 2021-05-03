import torch
import numpy as np
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super(Actor, self).__init__()

        self.l1 = nn.Linear(state_dim, 256)
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, action_dim)

        self.max_action = max_action

    def forward(self, state):
        a = F.relu(self.l1(state))
        a = F.relu(self.l2(a))
        return self.max_action * torch.tanh(self.l3(a))


class NormActor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(NormActor, self).__init__()

        self.l1 = nn.Linear(state_dim, 256)
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, action_dim)

    def forward(self, x):
        a = F.relu(self.l1(x))
        a = F.relu(self.l2(a))
        a = self.l3(a)
        return a


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()

        # Q1 architecture
        self.l1 = nn.Linear(state_dim + action_dim, 256)
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, 1)

        # Q2 architecture
        self.l4 = nn.Linear(state_dim + action_dim, 256)
        self.l5 = nn.Linear(256, 256)
        self.l6 = nn.Linear(256, 1)

    def forward(self, state, action):
        sa = torch.cat([state, action], 1)

        q1 = F.relu(self.l1(sa))
        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)

        q2 = F.relu(self.l4(sa))
        q2 = F.relu(self.l5(q2))
        q2 = self.l6(q2)
        return q1, q2

    def Q1(self, state, action):
        sa = torch.cat([state, action], 1)

        q1 = F.relu(self.l1(sa))
        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)
        return q1


class DualHeadCritic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(DualHeadCritic, self).__init__()

        self.feature_extractor = nn.Sequential(
            nn.Linear(state_dim + action_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
        )

        self.q1_external = nn.Sequential(
            self.feature_extractor,
            nn.Linear(256, 1)
        )

        self.q1_internal = nn.Sequential(
            self.feature_extractor,
            nn.Linear(256, 1)
        )

        self.q2_external = nn.Sequential(
            self.feature_extractor,
            nn.Linear(256, 1)
        )

        self.q2_internal = nn.Sequential(
            self.feature_extractor,
            nn.Linear(256, 1)
        )

    def forward(self, state, action):
        sa = torch.cat([state, action], 1)

        q1_E = self.q1_external(sa)
        q1_I = self.q1_internal(sa)

        q2_E = self.q2_external(sa)
        q2_I = self.q2_internal(sa)
        
        return q1_E, q1_I, q2_E, q2_I

    def Q1(self, state, action):
        sa = torch.cat([state, action], 1)

        q1_E = self.q1_external(sa)
        q1_I = self.q1_internal(sa)

        return q1_E + q1_I


class RNDModel(nn.Module):
    """ RND model architecture when used in conjunction with TD3. """

    def __init__(self, input_size):
        super(RNDModel, self).__init__()
        self.input_size = input_size

        self.predictor = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 128)
        )

        self.target = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 128)
        )

        # Random orthogonal initialization
        for p in self.modules():
            if isinstance(p, (nn.Linear, nn.Conv2d)):
                init.orthogonal_(p.weight, np.sqrt(2))
                p.bias.data.zero_()

        # No back-prop through the target network
        for p in self.target.parameters():
            p.requires_grad = False

    def forward(self, next_obs):
        target_feature = self.target(next_obs)
        predict_feature = self.predictor(next_obs)

        return predict_feature, target_feature
