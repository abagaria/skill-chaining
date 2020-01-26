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
from .dataset import MultiActionCountingDataset


class CountingLatentSpace(object):
    def __init__(self, state_dim, epsilon=1.0, phi_type="raw",
                 device=torch.device("cuda"), experiment_name="",
                 pixel_observations=False):
        """
        Latent space useful for generating pseudo-counts for states.
        Args:
            state_dim (int or tuple): feature size
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

        self.buffers = []

        assert phi_type in ("raw", "function"), phi_type

        if phi_type == "function":
            if pixel_observations:
                self.model = ConvPhiNetwork(state_dim)
            else:
                self.model = DensePhiNetwork(state_dim)

    def train(self, action_buffers, epochs=100):
        """

        Args:
            action_buffers (list):
                Each element is a numpy array containing all examples of a specific action.
                That way, we can get counts against each of a variety of actions.
            epochs (int):
                Number of epochs to train against. This is only used in the "phi" version of things.
        """
        if self.phi_type == "raw":
            return self._train_raw_counts(action_buffers)

        return self._train_function_counts(buffers=action_buffers, epochs=epochs)

    def _train_raw_counts(self, buffers):
        assert isinstance(buffers, list)
        for i, b in enumerate(buffers):
            assert isinstance(b, np.ndarray)
        self.buffers = buffers

    def _train_function_counts(self, *, buffers, epochs):
        """
        Args:
            buffers (list): A list of buffers for each action. The combination of them is the full support space.
            epochs (int): Number of training epochs
        """
        self.model.train()
        n_iterations = 0
        optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)

        data_set = MultiActionCountingDataset(action_buffers=buffers)
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

        self.buffers = buffers

    def _counting_loss(self, phi_s, phi_s_prime):
        # distance = F.mse(phi_s, phi_s_prime)
        squared_distance = ((phi_s - phi_s_prime) ** 2).sum(dim=1)
        for_exp = -squared_distance / (self.epsilon ** 2)
        all_loss = torch.exp(for_exp)

        return all_loss.sum()

    def get_counts(self, X, buffer_idx):
        """

        Args:
            X (np.ndarray): The data points we're going to be asking about
            buffer_idx (int): Which "action" we're counting against.

        Returns:
            Counts for all elements in X, when compared against all elements in the specified buffer

        """

        if self.phi_type == "raw":
            return self._get_raw_counts(X, buffer_idx)
        return self._get_function_counts(X, buffer_idx)

    def _get_raw_count_from_distances(self, distances):
        std_devs = distances / self.epsilon
        counts_per = np.exp(-(std_devs**2))
        counts = counts_per.sum(axis=1)
        return counts

    def _get_function_count_from_distances(self, distances):
        pass

    def _get_raw_counts(self, X, buffer_idx):
        buffer = self.buffers[buffer_idx]
        distances = naive_spread_counter.get_all_distances_to_buffer(X, buffer)
        counts = self._get_raw_count_from_distances(distances)
        return counts

    def _get_function_counts(self, X, buffer_idx):
        buffer = self.buffers[buffer_idx]
        self.model.eval()
        with torch.no_grad():
            X = torch.from_numpy(X).float().to(self.device)
            buffer = torch.from_numpy(buffer).float().to(self.device)
            X_transformed = self.model(X)
            Y_transformed = self.model(buffer)

        X_transformed = X_transformed.detach().cpu().numpy()
        Y_transformed = Y_transformed.detach().cpu().numpy()

        distances = naive_spread_counter.get_all_distances_to_buffer(X_transformed, Y_transformed)
        counts = self._get_raw_count_from_distances(distances)
        return counts
