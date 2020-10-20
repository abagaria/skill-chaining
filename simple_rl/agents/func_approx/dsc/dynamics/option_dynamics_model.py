import ipdb
import torch
import numpy as np
from tqdm import tqdm
import torch.optim as optim
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from simple_rl.agents.func_approx.dsc.dynamics.mdn.model import MixtureDensityNetwork


class OptionDynamicsModel(object):
    def __init__(self, in_shape, out_shape, num_mixtures, device):
        self.device = device
        self.input_dim = in_shape
        self.output_dim = out_shape

        assert self.input_dim in (2, 29), self.input_dim
        assert self.output_dim in (2, 29), self.output_dim

        self.model = MixtureDensityNetwork(in_shape, out_shape, num_mixtures).to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-3)

    def train(self, x, y):
        """
         Single epoch train p(y | x; theta) - a generative model conditioned on input.

        Args:
            x (np.ndarray): input
            y (np.ndarray): output labels

        Returns:
            loss_value (float)
        """
        losses = []
        dataset = DynamicsDataset(x, y, self.input_dim, self.output_dim)
        loader = DataLoader(dataset, shuffle=True, batch_size=32)

        for states, next_states in loader:
            states = states.to(self.device)
            next_states = next_states.to(self.device)

            self.optimizer.zero_grad()
            loss = self.model.loss(states, next_states).mean()
            loss.backward()
            self.optimizer.step()

            losses.append(loss.item())

        return np.mean(losses)

    def predict(self, x):
        """ Sample from the generative model. """
        x = x[:, :self.input_dim]
        x = torch.as_tensor(x).float().to(self.device)
        return self.model.sample(x).cpu().numpy()

    def evaluate(self, x, y, dim=None):
        """ Score the proximity of f(x) and labels y. """
        assert isinstance(x, np.ndarray), f"{type(x)}"
        assert isinstance(y, np.ndarray), f"{type(y)}"

        if dim is not None:
            y = y[:, :dim]

        predicted_next_states = self.predict(x)
        predicted_next_states = predicted_next_states[:, :dim]
        return np.linalg.norm(y - predicted_next_states)


class DynamicsDataset(Dataset):
    def __init__(self, states, states_p, input_dim, output_dim):
        """
        Dataset for training the Skill Dynamics Model.

        Args:
            states (np.ndarray): Start states
            states_p (np.ndarray): States after option rollout
            input_dim (int): Whether to use all or some of the input dimensions
            output_dim (int): Whether to use all or some of the output dimensions
        """
        assert output_dim <= input_dim, f"{input_dim, output_dim}"

        self.states = states[:, :input_dim]
        self.states_p = states_p[:, :output_dim]

    def __len__(self):
        return len(self.states)

    def __getitem__(self, idx):
        states = self.states[idx]
        next_states = self.states_p[idx]

        return torch.as_tensor(states).float(), \
               torch.as_tensor(next_states).float()
