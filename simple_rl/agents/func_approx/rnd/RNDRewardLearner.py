import ipdb
import torch
import numpy as np
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from simple_rl.agents.func_approx.td3.model import RNDModel
from simple_rl.agents.func_approx.rnd.utils import RunningMeanStd


class RND:
    def __init__(self, state_dim, lr, n_epochs, batch_size, update_interval, use_reward_norm, device):
        self.lr = lr
        self.device = device
        self.n_epochs = n_epochs
        self.state_dim = state_dim
        self.batch_size = batch_size
        self.update_interval = update_interval
        self.use_reward_norm = use_reward_norm

        self.memory = []

        self.model = RNDModel(state_dim).to(device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

        if use_reward_norm:
            self.reward_rms = RunningMeanStd()

        self.name = "rnd-reward-module"

    def get_reward(self, states):
        """ Inference: given a batch of states, return the associated batch of intrinsic rewards. """

        assert isinstance(states, (np.ndarray, torch.Tensor)), states
        assert len(states.shape) == 2, f"Expected (batch, state_dim), got {states.shape}"

        if isinstance(states, np.ndarray):
            states = torch.FloatTensor(states).to(self.device)

        with torch.no_grad():
            target_features = self.model.target(states)
            predicted_features = self.model.predictor(states)
            r_int = (target_features - predicted_features).pow(2).sum(1)

        if self.use_reward_norm:
            r_int /= np.sqrt(self.reward_rms.var)

        return r_int

    def update_reward_rms(self, episodic_rewards):
        """ Compute the mean, std and len of the rewards and update the reward_rms with it. """
        # TODO: Discount the rewards with their time indices (in RND torch implementation)
        assert self.use_reward_norm, f"use_reward_norm={self.use_reward_norm}"

        if len(episodic_rewards) > 0:
            mean = np.mean(episodic_rewards)
            std = np.std(episodic_rewards)
            size = len(episodic_rewards)

            self.reward_rms.update_from_moments(mean, std ** 2, size)

    def update(self, state):
        """ Add the new agent state to the memory. When the memory is full, train RND and reset the memory."""

        assert isinstance(state, np.ndarray), state

        self.memory.append(state)

        if len(self.memory) >= self.update_interval:
            states = np.array(self.memory)
            self.train(states)
            self.memory = []

    def train(self, states):
        """ Training: train RND by trying to predict a random transformation of the input states. """

        assert isinstance(states, (np.ndarray, torch.Tensor)), states
        assert len(states.shape) == 2, f"Expected (batch, state_dim), got {states.shape}"

        dataset = DataLoader(StateDataset(states), batch_size=self.batch_size, shuffle=True)

        for epoch in range(self.n_epochs):
            for state_batch in dataset:
                state_batch = state_batch.float().to(self.device)
                predicted_features, target_features = self.model(state_batch)
                loss = F.mse_loss(predicted_features, target_features.detach())
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()


class StateDataset(Dataset):
    def __init__(self, states):
        self.states = states

    def __len__(self):
        return len(self.states)

    def __getitem__(self, i):
        return self.states[i, :]
