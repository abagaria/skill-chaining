import ipdb
import torch
import numpy as np
import torch.nn.functional as F

from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from simple_rl.agents.func_approx.td3.model import RNDModel


class RND:
    def __init__(self, state_dim, lr, n_epochs, batch_size, device):
        self.lr = lr
        self.device = device
        self.n_epochs = n_epochs
        self.state_dim = state_dim
        self.batch_size = batch_size

        self.memory = []

        self.model = RNDModel(state_dim).to(device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

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

        return r_int

    def update(self, state):
        """ Add the new agent state to the memory. When the memory is full, train RND and reset the memory."""

        assert isinstance(state, np.ndarray), state

        self.memory.append(state)

        if len(self.memory) >= 5 * self.batch_size:
            states = np.array(self.memory)
            self.train(states)
            self.memory = []

    def train(self, states):
        """ Training: train RND by trying to predict a random transformation of the input states. """

        assert isinstance(states, (np.ndarray, torch.Tensor)), states
        assert len(states.shape) == 2, f"Expected (batch, state_dim), got {states.shape}"

        dataset = DataLoader(StateDataset(states), batch_size=self.batch_size, shuffle=True)

        for epoch in range(self.n_epochs):
            for state_batch in tqdm(dataset, desc=f"RND Training Epoch {epoch+1}/{self.n_epochs}"):
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
