# Python imports.
import torch
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
from tensorboardX import SummaryWriter
import ipdb
import shutil, os

# Other imports.
from . import naive_spread_counter
from .model import DensePhiNetwork, MNISTConvPhiNetwork
from simple_rl.agents.func_approx.exploration.optimism.latent.datasets.generated_dataset import MultiActionCountingDataset, StateNextStateDataset
from simple_rl.agents.func_approx.exploration.optimism.latent.datasets.generated_bonus_dataset import BonusDataset
from simple_rl.agents.func_approx.exploration.optimism.latent.datasets.chunked_state_dataset import ChunkedStateDataset
from simple_rl.agents.func_approx.exploration.optimism.latent.datasets.transition_consistency_dataset import TransitionConsistencyDataset, collate_fn
from simple_rl.agents.func_approx.exploration.optimism.latent.utils import get_lam_for_buffer_size

# TODO: Refactor dataset filenames so they are UpperCamelCase

class CountingLatentSpace(object):
    def __init__(self, state_dim, action_dim, latent_dim=2, epsilon=1.0, phi_type="raw", device=torch.device("cuda"), experiment_name="",
                 pixel_observations=False, lam=0.1, attractive_loss_type="quadratic", repulsive_loss_type="exponential",
                 optimization_quantity="count", bonus_scaling_term="sqrt", lam_scaling_term="none", writer=None, approx_chunk_size=1000):
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
            bonus_scaling_term (str): Form of the term used to scale the bonus loss term when combined with MDP distance term
            lam_scaling_term (str):  Form of the term used to scale the MDP distance term
            writer (SummaryWriter): tensorboard logging
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
        self.approx_chunk_size = approx_chunk_size

        self.buffers = [None for _ in range(self.action_dim)]

        assert phi_type in ("raw", "function"), phi_type
        assert optimization_quantity in ("count", "bonus", "count-tc", "chunked-count", "chunked-bonus", "chunked-log"), optimization_quantity

        self.optimization_quantity = optimization_quantity
        self.bonus_scaling_term = bonus_scaling_term
        self.lam_scaling_term = lam_scaling_term

        assert bonus_scaling_term in ("none", "sqrt", "linear", "chunked-sqrt"), bonus_scaling_term
        assert lam_scaling_term in ("none", "fit")

        print(f"Created CountingLatentSpace object with bonus_scaling_term {bonus_scaling_term} and lam_scaling_term {lam_scaling_term}")

        if phi_type == "function":
            self.reset_model()

        self._create_tensor_board_logger(writer)

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

    def _create_tensor_board_logger(self, writer):
        log_dir = "runs/{}".format(self.experiment_name)

        if writer:
            self.writer = writer
        else:
            if os.path.exists(log_dir):
                shutil.rmtree(log_dir)
            self.writer = SummaryWriter("{}".format(log_dir))

        self._n_iterations = 0

    def train(self, action_buffers=None, state_next_state_buffer=None, tc_action_buffers=None, epochs=100, verbose=True):
        """
`
        Args:
            action_buffers (list):
                Each element is a numpy array containing all examples of a specific action.
                That way, we can get counts against each of a variety of actions.
            state_next_state_buffer (list)
            tc_action_buffers (list): The type of action_buffers that the transition-consistency loss expects.
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
        elif self.optimization_quantity == "count-tc":
            assert tc_action_buffers is not None
            return self._train_counts_with_transition_consistency(buffers=tc_action_buffers, epochs=epochs)
        elif self.optimization_quantity == "chunked-count":
            return self._train_chunked_attractive_and_repulsive_function_counts(buffers=action_buffers, state_next_state_buffer=state_next_state_buffer, epochs=epochs)
        elif self.optimization_quantity == "chunked-bonus" or self.optimization_quantity == "chunked-log":
            return self._train_chunked_attractive_and_repulsive_function_representations(buffers=action_buffers, state_next_state_buffer=state_next_state_buffer, epochs=epochs, verbose=verbose)
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

                self.writer.add_scalar("TotalLoss", total_loss.item(), self._n_iterations)
                self.writer.add_scalar("RepulsiveLoss", repulsive_loss, self._n_iterations)
                if state_next_state_buffer is not None:
                    self.writer.add_scalar("AttractiveLoss", attractive_loss.item(), self._n_iterations)

                self._n_iterations = self._n_iterations + 1

        self.buffers = buffers

    def _get_bonus_scaling_term(self, N):
        if self.bonus_scaling_term == "none":
            return 1.
        if self.bonus_scaling_term == "sqrt":
            return 1. / np.sqrt(N)
        if self.bonus_scaling_term == "linear":
            return 1. / N
        if self.bonus_scaling_term == "chunked-sqrt":
            return np.sqrt(N) / self.approx_chunk_size
        raise ValueError(f"Bad value for bonus_scaling_term: {self.bonus_scaling_term}")

    def _get_scaled_lam(self, N):
        if self.lam_scaling_term == "none":
            return self.lam
        if self.lam_scaling_term == "fit":
            return get_lam_for_buffer_size(N)
        raise ValueError(f"Bad value for lam_scaling_term: {self.lam_scaling_term}")

    def _get_scaled_num_epochs(self, N, num_grad_steps=600):
        return int((1. * num_grad_steps * self.approx_chunk_size) / N)

    def _train_function_bonuses(self, *, buffers, state_next_state_buffer, epochs):
        if state_next_state_buffer is None:
            self._train_repulsive_function_bonuses(buffers=buffers, epochs=epochs)
        else:
            self._train_attractive_and_repulsive_function_bonuses(buffers=buffers,
                                                                  state_next_state_buffer=state_next_state_buffer,
                                                                  epochs=epochs)

    def _train_repulsive_function_bonuses(self, *, buffers, epochs):
        self.model.train()
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

                self.writer.add_scalar("BonusLoss", bonus_loss.item(), self._n_iterations)
                self._n_iterations += 1

        self.buffers = buffers

    def _train_attractive_and_repulsive_function_bonuses(self, *, buffers, state_next_state_buffer, epochs):

        self.model.train()
        
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

                attractive_loss = self.lam * self._counting_loss(state_batch_transformed, next_state_batch_transformed,
                                                      loss_type=self.attractive_loss_type)

                # Scale the bonus loss to account for the length of the buffer: roughly speaking, the impact of the counting loss scales
                # quadratically with the size of the buffer, however that of the attractive term grows linearly. Instead of increasing
                # lam over time, we scale the bonus loss.
                bonus_loss = self._get_bonus_scaling_term(len(state_next_state_buffer)) * self._bonus_loss(phi_s=support_batch_transformed, buffers=buffers)

                loss = bonus_loss - attractive_loss

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                self.writer.add_scalar("TotalLoss", loss.item(), self._n_iterations)
                self.writer.add_scalar("AttractiveLoss", attractive_loss.item(), self._n_iterations)
                self.writer.add_scalar("BonusLoss", bonus_loss.item(), self._n_iterations)

                self._n_iterations += 1

        self.buffers = buffers

    def _train_chunked_attractive_and_repulsive_function_counts(self, *, buffers, state_next_state_buffer, epochs):

        self.model.train()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-4, weight_decay=1e-2)
        data_set = ChunkedStateDataset(buffers, state_next_state_buffer, chunk_size=self.approx_chunk_size)

        for epoch in tqdm(range(epochs)):
            data_set.set_indices()
            for batch_id, (fc_state_tensor, fc_action_tensor, sc_state_tensor, sc_action_tensor,
                           sns_state_tensor, sns_next_state_tensor) in enumerate(data_set):

                fc_state_tensor = fc_state_tensor.to(self.device)
                # fc_action_tensor = fc_action_tensor.to(self.device)
                sc_state_tensor = sc_state_tensor.to(self.device)
                # sc_action_tensor = sc_action_tensor.to(self.device)
                sns_state_tensor = sns_state_tensor.to(self.device)
                sns_next_state_tensor = sns_next_state_tensor.to(self.device)

                fc_state_transformed = self.model(fc_state_tensor)
                sc_state_transformed = self.model(sc_state_tensor)
                sns_state_transformed = self.model(sns_state_tensor)
                sns_next_state_transformed = self.model(sns_next_state_tensor)

                attractive_loss = self.lam * self._counting_loss(sns_state_transformed, sns_next_state_transformed,
                                                                 loss_type=self.attractive_loss_type)

                outer_product_counts = self._get_all_counts_for_outer_product(fc_state_transformed, sc_state_transformed, count_type="exponential")
                repulsive_loss = outer_product_counts.sum()

                loss = repulsive_loss - attractive_loss

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()


                self.writer.add_scalar("TotalLoss", loss.item(), self._n_iterations)
                self.writer.add_scalar("AttractiveLoss", attractive_loss.item(), self._n_iterations)
                self.writer.add_scalar("RepulsiveLoss", repulsive_loss.item(), self._n_iterations)

                self._n_iterations += 1

        self.buffers = buffers

    def _train_chunked_attractive_and_repulsive_function_representations(self, *, buffers, state_next_state_buffer, epochs=-1, verbose=True):
        self.model.train()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-4, weight_decay=1e-2)
        data_set = ChunkedStateDataset(buffers, state_next_state_buffer, chunk_size=self.approx_chunk_size)

        if verbose == False:
            tqdm = lambda x : x # it's identity within this loop...

        buffer_size = len(state_next_state_buffer)
        if epochs <= 0:
            epochs = self._get_scaled_num_epochs(buffer_size)

        for epoch in tqdm(range(epochs)):
            data_set.set_indices()
            for batch_id, (fc_state_tensor, fc_action_tensor, sc_state_tensor, sc_action_tensor,
                           sns_state_tensor, sns_next_state_tensor) in enumerate(data_set):
                fc_state_tensor = fc_state_tensor.to(self.device)
                fc_action_tensor = fc_action_tensor.to(self.device)
                sc_state_tensor = sc_state_tensor.to(self.device)
                sc_action_tensor = sc_action_tensor.to(self.device)
                sns_state_tensor = sns_state_tensor.to(self.device)
                sns_next_state_tensor = sns_next_state_tensor.to(self.device)

                fc_state_transformed = self.model(fc_state_tensor)
                sc_state_transformed = self.model(sc_state_tensor)
                sns_state_transformed = self.model(sns_state_tensor)
                sns_next_state_transformed = self.model(sns_next_state_tensor)

                attractive_loss = self._counting_loss(sns_state_transformed, sns_next_state_transformed,
                                                      loss_type=self.attractive_loss_type)

                self.lam = self._get_scaled_lam(buffer_size)

                attractive_loss = self.lam * attractive_loss

                repulsive_loss = self._chunked_loss(phi_s=fc_state_transformed, phi_s_prime=sc_state_transformed,
                                                    phi_s_prime_actions=sc_action_tensor, buffers=buffers,
                                                    count_type=self.repulsive_loss_type)

                repulsive_loss = self._get_bonus_scaling_term(buffer_size) * repulsive_loss

                loss = repulsive_loss - attractive_loss

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                self.writer.add_scalar("TotalLoss", loss.item(), self._n_iterations)
                self.writer.add_scalar("AttractiveLoss", attractive_loss.item(), self._n_iterations)
                self.writer.add_scalar("RepulsiveLoss", repulsive_loss.item(), self._n_iterations)

                self._n_iterations += 1

        self.buffers = buffers

    def _typecheck_transition_consistency_buffers(self, buffers):
        """
        This function checks the buffers to make sure a bunch of things are true about their shapes and sizes.
        Args:
            buffers: We hope it is a list of lists of tuples of numpy arrays, but that's a lot to assume.

        Returns:

        """
        assert len(buffers) == self.action_dim, len(buffers) # Debatable

        for b in buffers:
            for ssp in b:
                assert isinstance(ssp, tuple), type(ssp)
                assert len(ssp) == 2, len(ssp)
                assert ssp[0].shape in (self.state_dim, (self.state_dim, )), f"Bad! {ssp[0].shape}, {self.state_dim}"
                assert ssp[1].shape in (self.state_dim, (self.state_dim,)), f"Bad! {ssp[0].shape}, {self.state_dim}"

    def _get_normal_buffers_from_tc_buffers(self, tc_buffers):
        """

        Args:
            tc_buffers (list): This is a list of lists of tuples of numpy arrays, which we want to just be a list of numpy arrays...

        Returns:
            buffers (list): Just the parent states from this list (we don't care where they ended up...)

        """
        tc_buffers = [np.array(b) for b in tc_buffers]
        buffers = [b[:,0,...] for b in tc_buffers]
        return buffers

    def _get_all_counts_for_outer_product(self, state_vector_1, state_vector_2, count_type="exponential"):
        squared_distance_matrix = naive_spread_counter.torch_get_square_distances_to_buffer(state_vector_1, state_vector_2)
        all_counts = self._get_all_counts_from_squared_distance_matrix(squared_distance_matrix, count_type=count_type)
        return all_counts

    def _get_all_counts_from_squared_distance_matrix(self, squared_distance_matrix, count_type="exponential"):
        """

        Args:
            squared_distance_matrix (torch.Tensor): (M, N) tensor that represents the squared distances from one set of vectors to another.

        Returns:
            count_matrix (torch.Tensor): (M, N) tensor that represents the count-overlap betwen one set of vectors and another.
        """
        if count_type == "normal":
            for_exp = -squared_distance_matrix / (self.epsilon ** 2)
            all_counts = torch.exp(for_exp)
        if count_type == "exponential":
            # BUG ALERT: sqrt at zero gives a reasonable value, but a NAN gradient... makes sense.
            distance_matrix = (squared_distance_matrix + 1e-6).sqrt()
            for_exp = -distance_matrix / self.epsilon
            all_counts = torch.exp(for_exp)
        return all_counts

    def _train_counts_with_transition_consistency(self, count_type="exponential", scaling=1., *, buffers, epochs):
        self._typecheck_transition_consistency_buffers(buffers)
        self.model.train()
        
        optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-4, weight_decay=1e-2)
        # optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-4)

        data_set = TransitionConsistencyDataset(buffers, device=self.device)
        loader = DataLoader(data_set, batch_size=64, shuffle=True, collate_fn=collate_fn)

        for epoch in tqdm(range(epochs)):
            for batch_idx, action_to_sns_dict in enumerate(loader):
                repulsive_losses = 0
                attractive_losses = 0
                mean_counts_for_parent = 0.

                for action in action_to_sns_dict:
                    action_batch = action_to_sns_dict[action].to(self.device)
                    full_action_buffer = data_set.get_action_buffer_tensor(action).to(self.device)
                    action_batch_parent = action_batch[:, 0, ...]
                    action_batch_child = action_batch[:, 1, ...]
                    full_action_buffer_parent = full_action_buffer[:, 0, ...]
                    full_action_buffer_child = full_action_buffer[:, 1, ...]

                    action_batch_parent_transformed = self.model(action_batch_parent)
                    action_batch_child_transformed = self.model(action_batch_child)
                    full_action_buffer_parent_transformed = self.model(full_action_buffer_parent)
                    full_action_buffer_child_transformed = self.model(full_action_buffer_child)

                    squared_distance_from_parent = naive_spread_counter.torch_get_square_distances_to_buffer(action_batch_parent_transformed, full_action_buffer_parent_transformed)
                    squared_distance_from_child = naive_spread_counter.torch_get_square_distances_to_buffer(action_batch_child_transformed, full_action_buffer_child_transformed)
                    assert squared_distance_from_parent.shape == squared_distance_from_child.shape == torch.Size((len(action_batch), len(full_action_buffer)))
                    counts_for_child = self._get_all_counts_from_squared_distance_matrix(squared_distance_from_child, count_type=count_type)

                    with torch.no_grad():
                        counts_for_parent = self._get_all_counts_from_squared_distance_matrix(
                            squared_distance_from_parent, count_type=count_type).detach()
                        mean_counts_for_parent += counts_for_parent.sum()

                    repulsion_amount = (1 - scaling*counts_for_parent) * counts_for_child
                    repulsion_loss = repulsion_amount  #-1. / (repulsion_amount.sum(dim=1) + 1e-2).sqrt()
                    repulsive_losses += repulsion_loss.sum()

                    other_repulsive_losses = 0
                    for other_action in action_to_sns_dict:
                        if other_action == action:
                            continue
                        full_other_action_buffer = data_set.get_action_buffer_tensor(other_action).to(self.device)
                        full_other_action_buffer_child = full_other_action_buffer[:, 1, ...]
                        full_other_action_buffer_child_transformed = self.model(full_other_action_buffer_child)

                        other_squared_distance_from_child = naive_spread_counter.torch_get_square_distances_to_buffer(action_batch_child_transformed, full_other_action_buffer_child_transformed)
                        other_action_counts = self._get_all_counts_from_squared_distance_matrix(other_squared_distance_from_child, count_type="normal")
                        other_action_loss = other_action_counts.sum()
                        other_repulsive_losses += other_action_loss

                    attraction_loss = self._counting_loss(action_batch_parent_transformed, action_batch_child_transformed, self.attractive_loss_type)

                    attractive_losses += (self.lam * attraction_loss)

                # We scale the repulsive losses by N because we are using the counting losses for now
                # If we were to use bonus based repulsive forces, we would scale by sqrt(N)
                # as in the function _train_attractive_and_repulsive_function_bonuses(...)
                repulsive_losses = repulsive_losses / len(data_set)
                other_repulsive_losses = other_repulsive_losses / len(data_set)

                loss = repulsive_losses + other_repulsive_losses - attractive_losses

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                self.writer.add_scalar("TotalLoss", loss.item(), self._n_iterations)
                self.writer.add_scalar("TC-Loss", repulsive_losses.item(), self._n_iterations)
                self.writer.add_scalar("TC-Other-Loss", other_repulsive_losses.item(), self._n_iterations)
                self.writer.add_scalar("AttractiveLoss", attractive_losses.item(), self._n_iterations)
                self.writer.add_scalar("MeanCountsForParents", mean_counts_for_parent / (len(data_set) ** 2), self._n_iterations)

                self._n_iterations += 1

        # Unfortunately, we need the old definition of buffers to make the other functions (like get_counts) to work
        normal_buffers = self._get_normal_buffers_from_tc_buffers(buffers)
        self.buffers = normal_buffers

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

    def _actions_to_one_hot(self, action_tensor):
        assert len(action_tensor.shape) == 1, action_tensor.shape
        one_hot = torch.zeros(len(action_tensor), self.action_dim).float().to(self.device)
        one_hot[torch.arange(len(action_tensor)), action_tensor.long()] = 1

        return one_hot

    def _chunked_loss(self, scaling_term=None, count_type="exponential", *, phi_s, phi_s_prime, phi_s_prime_actions, buffers):
        """
        It's gather time!
        Args:
            scaling_term:
            count_type:
            phi_s:
            phi_s_prime:
            phi_s_prime_actions:
            buffers:

        Returns:

        """
        # f has shape (num_phi_s, num_actions)
        f = self._compute_chunked_scaling_term(phi_s, buffers, chunk_size=self.approx_chunk_size,
                                               count_type=count_type) if scaling_term is None else scaling_term

        # all_counts has shape (num_phi_s, num_phi_s_prime)
        all_counts = self._get_all_counts_for_outer_product(phi_s, phi_s_prime, count_type=count_type)
        one_hot_actions = self._actions_to_one_hot(phi_s_prime_actions)

        counts_per_action_per_phi_s = all_counts @ one_hot_actions
        assert f.shape == counts_per_action_per_phi_s.shape

        # ipdb.set_trace()

        chunked_loss = (f * counts_per_action_per_phi_s).sum()
        return chunked_loss

    def _compute_chunked_scaling_term(self, phi_s, buffers, chunk_size, count_type="exponential"):
        """

        Args:
            phi_s (torch.tensor)
            buffers (list): list of np.ndarray action buffers
            chunk_size (int)

        Returns:
            bonus_scale (torch.tensor): f(s)^(3/2) of size (mini_batch_size x num_actions)
        """
        assert len(phi_s.shape) == 2, phi_s.shape
        assert len(buffers) == self.action_dim, len(buffers)
        assert type(chunk_size) == int, chunk_size

        self.model.eval()

        scaling_term = torch.zeros(phi_s.shape[0], len(buffers), requires_grad=False).to(self.device)

        for action_idx, action_buffer in enumerate(buffers):  # type: int, np.ndarray
            num_chunks = int(np.ceil(action_buffer.shape[0] / chunk_size))
            action_buffer_chunks = np.array_split(action_buffer, num_chunks, axis=0)
            total_action_count = 0.

            for buffer_chunk in action_buffer_chunks:  # type: np.ndarray
                with torch.no_grad():
                    buffer_chunk = torch.from_numpy(buffer_chunk).float().to(self.device)
                    phi_s_prime = self.model(buffer_chunk)
                    counts = self._get_all_counts_for_outer_product(phi_s, phi_s_prime, count_type=count_type)
                    total_action_count += counts.sum(dim=1)

            # Note: 1/sqrt(x) is a monotonically *decreasing* function, whereas log is a monotonically
            # *increasing* function. As a result, we want to maximize the *negative* of 1 / count when
            # using optimized_quantity of `chuked-log`
            if self.optimization_quantity == "chunked-bonus":
                scaling_term[:, action_idx] = (total_action_count + 1e-1) ** (-3. / 2.)
            elif self.optimization_quantity == "chunked-log":
                scaling_term[:, action_idx] = -1. / (total_action_count + 1e-1)
            else:
                raise ValueError(f"{self.optimization_quantity} not supported with chunked loss")

        assert scaling_term.shape[0] == phi_s.shape[0], scaling_term.shape[0]
        assert scaling_term.shape[1] == len(buffers), scaling_term.shape[1]

        self.model.train()

        return scaling_term

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

        chunked = self.optimization_quantity in ("chunked-count", "chunked-bonus")
        chunk_size = self.approx_chunk_size if chunked else None

        return self._get_function_counts(X, buffer_idx, chunk_size)

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

    def _get_function_counts(self, X, buffer_idx, chunk_size=None):
        buffer = self.buffers[buffer_idx]  # type: np.ndarray

        # If we have never taken this action before, return max counts
        if buffer is None:
            max_counts = np.zeros((X.shape[0],))
            return max_counts

        # Chunk up the action buffer. No need to chunk up X because that is
        # usually small, whereas the action buffer can be very large
        if chunk_size is not None:
            num_chunks = int(np.ceil(buffer.shape[0] / chunk_size))
            action_buffer_chunks = np.array_split(buffer, num_chunks, axis=0)
            counts = np.zeros((X.shape[0],))

            # Sum up the counts across all the chunks of the action buffer
            for buffer_chunk in action_buffer_chunks:  # type: np.ndarray
                chunk_counts = self._get_function_counts_for_chunk(X, buffer_chunk)
                counts = counts + chunk_counts
            return counts
        return self._get_function_counts_for_chunk(X, buffer)

    def _get_function_counts_for_chunk(self, X, buffer):
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
