# Python imports.
import torch
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
from tensorboardX import SummaryWriter
import pdb
import shutil, os

# Other imports.
from . import naive_spread_counter
from .model import DensePhiNetwork, MNISTConvPhiNetwork
from simple_rl.agents.func_approx.exploration.optimism.latent.datasets.generated_dataset import MultiActionCountingDataset, StateNextStateDataset
from simple_rl.agents.func_approx.exploration.optimism.latent.datasets.generated_bonus_dataset import BonusDataset


class CountingLatentSpace(object):
    def __init__(self, state_dim, action_dim, latent_dim=2, epsilon=1.0, phi_type="raw", device=torch.device("cuda"), experiment_name="",
                 pixel_observations=False, lam=0.1, attractive_loss_type="quadratic", repulsive_loss_type="exponential",
                 optimization_quantity="count"):
        """
        Latent space useful for generating pseudo-counts for states.
        Args:
            state_dim (int or tuple): feature size
            epsilon (float): The "standard deviation" of our bell curved counting function
            phi_type (str): Use feature extractor or not
            device (str): torch.device("cpu") OR torch.device("cuda:i")
            experiment_name (str): Name of the current experiment
            pixel_observations (bool): Whether input is images or dense feature vectors
            lam (float): scalar that balances attractive and repulsive loss terms
            attractive_loss_type (str): form of the loss term that brings together representations
                                        for pairs of states
            repulsive_loss_type (str): form of the loss term that pushes apart representations
                                       for any two states
            optimization_quantity (str): What we're trying to optimize. We either minimize "count" or maximize exploration "bonus"
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.latent_dim = latent_dim

        self.epsilon = epsilon
        self.phi_type = phi_type
        self.device = device
        self.experiment_name = experiment_name
        self.lam = lam
        self.attractive_loss_type = attractive_loss_type
        self.repulsive_loss_type = repulsive_loss_type
        self.pixel_observations = pixel_observations

        log_dir = "runs/{}".format(experiment_name)
        if os.path.exists(log_dir):
            shutil.rmtree(log_dir)
        self.writer = SummaryWriter() if experiment_name is "" else SummaryWriter("runs/{}".format(experiment_name))

        self.buffers = [None for _ in range(self.action_dim)]

        assert phi_type in ("raw", "function"), phi_type
        assert optimization_quantity in ("count", "bonus"), optimization_quantity

        self.optimization_quantity = optimization_quantity

        if phi_type == "function":
            self.reset_model()

    def reset_model(self):
        if self.pixel_observations:
            self.model = MNISTConvPhiNetwork(self.state_dim, latent_size=self.latent_dim)
        else:
            self.model = DensePhiNetwork(self.state_dim, latent_size=self.latent_dim)

    def add_transition(self, state, action):
        """"""
        assert isinstance(state, np.ndarray), type(state)
        assert state.shape in (self.state_dim, (self.state_dim, )), f"Mismatching dims: {(state.shape, self.state_dim)}"

        expanded_state = np.expand_dims(state, 0)
        if self.buffers[action] is None:
            self.buffers[action] = expanded_state
        else:
            self.buffers[action] = np.concatenate((self.buffers[action], expanded_state), axis=0)


    def train(self, action_buffers=None, state_next_state_buffer=None, epochs=100):
        """
`
        Args:
            action_buffers (list):
                Each element is a numpy array containing all examples of a specific action.
                That way, we can get counts against each of a variety of actions.
            state_next_state_buffer (list)
            epochs (int):
                Number of epochs to train against. This is only used in the "phi" version of things.
        """
        if action_buffers is None:
            assert len(self.buffers) != 0, self.buffers
            action_buffers = self.buffers

        if self.phi_type == "raw":
            return self._train_raw_counts(action_buffers)

        if self.optimization_quantity == "bonus":
            return self._train_function_bonuses(buffers=action_buffers, state_next_state_buffer=state_next_state_buffer, epochs=epochs)
        elif self.optimization_quantity == "count":
            return self._train_function_counts(buffers=action_buffers, state_next_state_buffer=state_next_state_buffer, epochs=epochs)
        raise NotImplementedError(f"Optimization quantity {self.optimization_quantity} not implemented yet.")

    def extract_features(self, states):
        """
        Extract the learned feature representations corresponding to `states`.
        Args:
            states (np.ndarray): A numpy array of states that you would like to see the latent representation of

        Returns:
            latent_reprs (np.ndarray)

        """
        self.model.eval()
        with torch.no_grad():
            features = self.model(torch.from_numpy(states).float().to(self.device))
        return features.detach().cpu().numpy()

    def _train_raw_counts(self, buffers):
        assert isinstance(buffers, list)
        for i, b in enumerate(buffers):
            assert isinstance(b, np.ndarray)
        self.buffers = buffers

    def _get_state_next_state_batch(self, sns_loader, sns_iter):
        try:
            state_next_state_batch = next(sns_iter)
        except StopIteration:
            sns_iter = iter(sns_loader)
            state_next_state_batch = next(sns_iter)

        return state_next_state_batch, sns_iter

    def _train_function_counts(self, *, buffers, state_next_state_buffer, epochs):
        """
        Args:
            buffers (list): A list of buffers for each action. The combination of them is the full support space.
            state_next_state_buffer (list): This is a list of tuples, each which is "state, next_state"
            epochs (int): Number of training epochs
        """
        self.model.train()
        n_iterations = 0
        optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-4, weight_decay=1e-2)

        data_set = MultiActionCountingDataset(action_buffers=buffers)
        loader = DataLoader(data_set, batch_size=64, shuffle=True)

        if state_next_state_buffer is not None:
            sns_data_set = StateNextStateDataset(state_next_state_buffer)
            sns_loader = DataLoader(sns_data_set, batch_size=64, shuffle=True)
            sns_iter = iter(sns_loader)

        for epoch in range(epochs):
            for batch_idx, (support_batch, action_batch) in tqdm(enumerate(loader)):

                if state_next_state_buffer is not None:
                    (state_batch, next_state_batch), sns_iter = self._get_state_next_state_batch(sns_loader, sns_iter)
                    state_batch = state_batch.to(self.device)
                    next_state_batch = next_state_batch.to(self.device)

                # Transfer data to GPU
                action_batch = action_batch.to(self.device)
                support_batch = support_batch.to(self.device)

                support_batch_transformed = self.model(support_batch)  # phi(s)
                action_batch_transformed = self.model(action_batch)    # phi(s')

                total_loss = self._counting_loss(support_batch_transformed, action_batch_transformed, loss_type=self.repulsive_loss_type)
                repulsive_loss = total_loss.item()

                if state_next_state_buffer is not None:
                    state_batch_transformed = self.model(state_batch)
                    next_state_batch_transformed = self.model(next_state_batch)
                    attractive_loss = self._counting_loss(state_batch_transformed, next_state_batch_transformed, loss_type=self.attractive_loss_type)
                    # That minus sign ain't an accident
                    total_loss -= (self.lam * attractive_loss)

                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()

                self.writer.add_scalar("TotalLoss", total_loss.item(), n_iterations)
                self.writer.add_scalar("RepulsiveLoss", repulsive_loss, n_iterations)
                if state_next_state_buffer is not None:
                    self.writer.add_scalar("AttractiveLoss", attractive_loss.item(), n_iterations)

                n_iterations = n_iterations + 1

        self.buffers = buffers

    def _train_function_bonuses(self, *, buffers, state_next_state_buffer, epochs):
        if state_next_state_buffer is None:
            self._train_repulsive_function_bonuses(buffers=buffers, epochs=epochs)
        else:
            self._train_attractive_and_repulsive_function_bonuses(buffers=buffers,
                                                                  state_next_state_buffer=state_next_state_buffer,
                                                                  epochs=epochs)

    def _train_repulsive_function_bonuses(self, *, buffers, epochs):
        self.model.train()
        n_iterations = 0
        optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-4, weight_decay=1e-2)

        data_set = BonusDataset(buffers)
        loader = DataLoader(data_set, batch_size=64, shuffle=True)

        for epoch in tqdm(range(epochs)):
            for batch_idx, support_batch in enumerate(loader):
                support_batch = support_batch.to(self.device)

                support_batch_transformed = self.model(support_batch)  # phi(s)
                bonus_loss = self._bonus_loss(phi_s=support_batch_transformed, buffers=buffers)

                optimizer.zero_grad()
                bonus_loss.backward()
                optimizer.step()

                self.writer.add_scalar("BonusLoss", bonus_loss.item(), n_iterations)
                n_iterations += 1

        self.buffers = buffers

    def _train_attractive_and_repulsive_function_bonuses(self, *, buffers, state_next_state_buffer, epochs):
        self.model.train()
        n_iterations = 0
        optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-4, weight_decay=1e-2)

        data_set = BonusDataset(buffers, state_next_state_buffer)
        loader = DataLoader(data_set, batch_size=64, shuffle=True)

        for epoch in tqdm(range(epochs)):
            for batch_idx, (support_batch, state_batch, next_state_batch) in enumerate(loader):
                support_batch = support_batch.to(self.device)
                state_batch = state_batch.to(self.device)
                next_state_batch = next_state_batch.to(self.device)

                support_batch_transformed = self.model(support_batch)  # phi(s)
                state_batch_transformed = self.model(state_batch)
                next_state_batch_transformed = self.model(next_state_batch)
                attractive_loss = self._counting_loss(state_batch_transformed, next_state_batch_transformed,
                                                      loss_type=self.attractive_loss_type)
                bonus_loss = self._bonus_loss(phi_s=support_batch_transformed, buffers=buffers)
                loss = bonus_loss - (self.lam * attractive_loss)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                self.writer.add_scalar("TotalLoss", loss.item(), n_iterations)
                self.writer.add_scalar("AttractiveLoss", attractive_loss.item(), n_iterations)
                self.writer.add_scalar("BonusLoss", bonus_loss.item(), n_iterations)

                n_iterations += 1

        self.buffers = buffers

    def _counting_loss(self, phi_s, phi_s_prime, loss_type="normal"):
        assert loss_type in ("normal", "quadratic", "exponential"), loss_type

        if loss_type == "normal":
            squared_distance = ((phi_s - phi_s_prime) ** 2).sum(dim=1)
            for_exp = -squared_distance / (self.epsilon ** 2)
            all_loss = torch.exp(for_exp)
            loss = all_loss.sum()
        if loss_type == "quadratic":
            squared_distance = ((phi_s - phi_s_prime) ** 2).sum(dim=1)
            loss = -squared_distance.sum()
        if loss_type == "exponential":
            # BUG ALERT: sqrt at zero gives a reasonable value, but a NAN gradient... makes sense.
            squared_distance = ((phi_s - phi_s_prime) ** 2).sum(dim=1)
            distance = (squared_distance + 1e-6).sqrt()
            for_exp = -distance / self.epsilon
            all_loss = torch.exp(for_exp)
            loss = all_loss.sum()

        return loss

    def _bonus_loss(self, phi_s, buffers, loss_type="exponential"):
        """
        Loss term defined to estimate the exploration bonus 1/sqrt(N(s, a)) directly.
        This automatically biases the optimization towards states that have low counts,
        since it is more important to get their bonus correct.
        Args:
            phi_s (torch.tensor): feature representation of `M` states
            buffers (list): list of all the action buffers
            loss_type (str): exponential / normal / quadratic

        Returns:
            loss (torch.tensor): scalar representing the bonus loss term to minimize.
        """
        batch_size = phi_s.shape[0]
        bonuses = torch.zeros(batch_size, device=self.device)

        for buffer_idx in range(len(buffers)):
            action_buffer = buffers[buffer_idx]  # type: np.ndarray # of shape (N, latent_size)

            if action_buffer is not None:
                action_buffer = torch.from_numpy(action_buffer).float().to(self.device)
                phi_s_prime = self.model(action_buffer)

                # (M, N) matrix of pairwise distances between phi(s) and phi(s')
                squared_distance = naive_spread_counter.torch_get_square_distances_to_buffer(phi_s, phi_s_prime)

                # Apply exponential filter around the pairwise distances to zero out the effect of far-away states
                distance = (squared_distance + 1e-6).sqrt()
                for_exp = -distance / self.epsilon
                filtered_pairwise_distances = torch.exp(for_exp)

                # Compute the counts and then the exploration bonuses for all the M states
                state_counts = filtered_pairwise_distances.sum(dim=1)
                state_bonuses = 1. / torch.sqrt(state_counts + 1e-2)

                bonuses += state_bonuses

        # Sum the bonus for each state to represent the bonus of the current mini-batch
        batch_bonus = bonuses.sum()

        # Since we want to maximize exploration bonus, the quantity to minimize is the negative bonus
        loss = -1 * batch_bonus

        return loss


    def get_counts(self, X, buffer_idx):
        """

        Args:
            X (np.ndarray): The data points we're going to be asking about
            buffer_idx (int): Which "action" we're counting against.

        Returns:
            Counts for all elements in X, when compared against all elements in the specified buffer
        TODO: This should return shape (batch_size, 1). Right now it returns (batch_size,)
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

        # If we have never taken this action before, return max counts
        if buffer is None:
            max_counts = np.zeros((X.shape[0],))
            return max_counts

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
