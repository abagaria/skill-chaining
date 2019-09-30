import torch
import torch.nn
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import pdb


class NN(torch.nn.Module):
    def __init__(self, in_dim, out_dim, n_hid):
        super(NN, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.n_hid = n_hid

        self.fc1 = torch.nn.Linear(in_dim, n_hid)
        self.fc2 = torch.nn.Linear(n_hid, n_hid)
        self.fc3 = torch.nn.Linear(n_hid, out_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        y = self.fc3(x)
        return y


class RND:
    def __init__(self, in_dim, out_dim, n_hid, device):
        self.device = device
        self.target = NN(in_dim, out_dim, n_hid).to(self.device)
        self.model = NN(in_dim, out_dim, n_hid).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.0001)

    def get_reward(self, x):
        y_true = self.target(x).detach()
        y_pred = self.model(x)
        reward = F.mse_loss(y_pred, y_true)  #torch.pow(y_pred - y_true, 2).sum()
        return reward

    def update(self, Ri):
        Ri.sum().backward()
        self.optimizer.step()

    def get_single_reward(self, state):
        """ state is a numpy array of features. """
        x = torch.from_numpy(state).float().to(self.device)
        with torch.no_grad():
            y_true = self.target(x).detach()
            y_pred = self.model(x)
        reward = F.mse_loss(y_pred, y_true)
        clipped_reward = reward.detach().clamp(-1., 1.).item()
        return clipped_reward

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

class RNDModel(nn.Module):
    def __init__(self, device):
        super(RNDModel, self).__init__()

        feature_output = 7 * 7 * 64
        self.model = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=32,
                kernel_size=8,
                stride=4),
            nn.LeakyReLU(),
            nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=4,
                stride=2),
            nn.LeakyReLU(),
            nn.Conv2d(
                in_channels=64,
                out_channels=64,
                kernel_size=3,
                stride=1),
            nn.LeakyReLU(),
            Flatten(),
            nn.Linear(feature_output, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 512)
        )

        self.target = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=32,
                kernel_size=8,
                stride=4),
            nn.LeakyReLU(),
            nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=4,
                stride=2),
            nn.LeakyReLU(),
            nn.Conv2d(
                in_channels=64,
                out_channels=64,
                kernel_size=3,
                stride=1),
            nn.LeakyReLU(),
            Flatten(),
            nn.Linear(feature_output, 512)
        )

        for p in self.modules():
            if isinstance(p, nn.Conv2d):
                nn.init.orthogonal_(p.weight, np.sqrt(2))
                p.bias.data.zero_()

            if isinstance(p, nn.Linear):
                nn.init.orthogonal_(p.weight, np.sqrt(2))
                p.bias.data.zero_()

        for param in self.target.parameters():
            param.requires_grad = False

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.0001)

        self.device = device
        self.to(device)

    def forward(self, next_obs):
        target_feature = self.target(next_obs)
        predict_feature = self.model(next_obs)
        reward = F.mse_loss(target_feature, predict_feature)

        return reward

    def get_reward(self, x):
        return self(x)

    def update(self, Ri):
        Ri.sum().backward()
        self.optimizer.step()

    def get_single_reward(self, state):
        """ state is a numpy array of features. """
        if not isinstance(state, np.ndarray):
            state = np.array(state)  # Lazy frame -> numpy array

        x = torch.from_numpy(state).float().unsqueeze(0).to(self.device)

        with torch.no_grad():
            reward = self(x)
        clipped_reward = reward.detach().item()
        return clipped_reward


class RunningMeanStd(object):
    # https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm
    def __init__(self, epsilon=1e-4, shape=()):
        self.mean = np.zeros(shape, 'float64')
        self.var = np.ones(shape, 'float64')
        self.count = epsilon

    def update(self, x):
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0]
        self.update_from_moments(batch_mean, batch_var, batch_count)

    def update_from_moments(self, batch_mean, batch_var, batch_count):
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * (self.count)
        m_b = batch_var * (batch_count)
        M2 = m_a + m_b + np.square(delta) * self.count * batch_count / (self.count + batch_count)
        new_var = M2 / (self.count + batch_count)

        new_count = batch_count + self.count

        self.mean = new_mean
        self.var = new_var
        self.count = new_count
