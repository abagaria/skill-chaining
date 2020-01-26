# Python imports.
import pdb
import torch
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
from tensorboardX import SummaryWriter

# Other imports.
from . import naive_spread_counter
from .model import DensePhiNetwork, ConvPhiNetwork
from .dataset import CountingDataset


class CountingLatentSpace(object):
    def __init__(self, state_dim, epsilon=1.0, phi_type="raw",
                 device=torch.device("cuda"), experiment_name="",
                 pixel_observations=False):
        """
        Latent space useful for generating pseudo-counts for states.
        Args:
            state_dim (int): feature size
            epsilon (float): The "standard deviation" of our bell curved counting function
            phi_type (str): Use feature extractor or not
            device (str): torch.device("cpu") OR torch.device("cuda:i")
            experiment_name (str): Name of the current experiment
        """
        self.epsilon = epsilon
        self.phi_type = phi_type
        self.device = device
        self.experiment_name = experiment_name
        self.writer = SummaryWriter() if experiment_name is "" else SummaryWriter("runs/{}".format(experiment_name))

        self.buffer = np.asarray([])

        assert phi_type in ("raw", "function"), phi_type

        if phi_type == "function":
            self.model = DensePhiNetwork(state_dim)

        # if phi_type == "function":
        #     if pixel_observations:
        #         self.model = ConvPhiNetwork(state_dim)
        #     else:
        #         self.model = DensePhiNetwork(state_dim)


    def train(self, action_buffer, full_buffer=None, epochs=100):
        if self.phi_type == "raw":
            return self._train_raw_counts(action_buffer)

        return self._train_function_counts(full_buffer=full_buffer, action_buffer=action_buffer, epochs=epochs)

    def _train_raw_counts(self, buffer):
        assert isinstance(buffer, np.ndarray)
        self.buffer = buffer

    def _train_function_counts(self, *, full_buffer, action_buffer, epochs):
        """
        Args:
            full_buffer (np.ndarray): The "support" of this function. Represents the states we may query about
                in the future. Corresponds to all the states seen so far
            action_buffer (np.ndarray): States from which we have taken the current action. If you overlap
                with these, then you get a plus 1

        """
        self.model.train()
        n_iterations = 0
        optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)

        data_set = CountingDataset(full_buffer=full_buffer, action_buffer=action_buffer)
        loader = DataLoader(data_set, batch_size=32, shuffle=True)

        for epoch in tqdm(range(epochs)):
            for batch_idx, (support_batch, action_batch) in enumerate(loader):

                # Transfer data to GPU
                action_batch = action_batch.to(self.device)
                support_batch = support_batch.to(self.device)

                support_batch_transformed = self.model(support_batch)  # phi(s)
                action_batch_transformed = self.model(action_batch)    # phi(s')

                batch_loss = self._counting_loss(support_batch_transformed, action_batch_transformed)
                optimizer.zero_grad()
                batch_loss.backward()
                optimizer.step()

                self.writer.add_scalar("Loss", batch_loss.item(), n_iterations)
                n_iterations = n_iterations + 1

        self.buffer = action_buffer

    def _counting_loss(self, phi_s, phi_s_prime):
        # distance = F.mse(phi_s, phi_s_prime)
        squared_distance = ((phi_s - phi_s_prime) ** 2).sum(dim=1)
        for_exp = -squared_distance / (self.epsilon ** 2)
        all_loss = torch.exp(for_exp)

        return all_loss.sum()

    def get_counts(self, X):
        if self.phi_type == "raw":
            return self._get_raw_counts(X)
        return self._get_function_counts(X)

    def _get_raw_count_from_distances(self, distances):
        std_devs = distances / self.epsilon
        counts_per = np.exp(-(std_devs**2))
        counts = counts_per.sum(axis=1)
        return counts

    def _get_function_count_from_distances(self, distances):
        pass

    def _get_raw_counts(self, X):
        distances = naive_spread_counter.get_all_distances_to_buffer(X, self.buffer)
        counts = self._get_raw_count_from_distances(distances)
        return counts

    def _get_function_counts(self, X):
        self.model.eval()
        with torch.no_grad():
            X = torch.from_numpy(X).float().to(self.device)
            buffer = torch.from_numpy(self.buffer).float().to(self.device)
            X_transformed = self.model(X)
            Y_transformed = self.model(buffer)

        X_transformed = X_transformed.detach().cpu().numpy()
        Y_transformed = Y_transformed.detach().cpu().numpy()

        distances = naive_spread_counter.get_all_distances_to_buffer(X_transformed, Y_transformed)
        counts = self._get_raw_count_from_distances(distances)
        return counts
