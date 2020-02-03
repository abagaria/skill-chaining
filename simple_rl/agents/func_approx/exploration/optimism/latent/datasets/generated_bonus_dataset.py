from torch.utils.data import Dataset, ConcatDataset
import torch
import numpy as np


class BonusDataset(Dataset):
    """ Dataset class for approximating the exploration bonus directly. """

    def __init__(self, buffers, state_next_state_buffer=None):
        """
        Combine action buffers `buffers` into a single buffer and then sample from it.
        If `state_next_state_buffer` is given, then also return samples from it.
        Args:
            buffers (list): each element is a np.ndarray of states seen under that action_idx
            state_next_state_buffer (list): each element is a tuple of np.ndarray representing states
        """
        super(BonusDataset, self).__init__()

        self.full_buffer = np.vstack(buffers)
        self.state_next_state_buffer = state_next_state_buffer

    def __len__(self):
        if self.state_next_state_buffer is not None:
            assert len(self.full_buffer) == len(self.state_next_state_buffer)
        return len(self.full_buffer)

    def __getitem__(self, i):
        full_state_vector = self.full_buffer[i]

        if self.state_next_state_buffer is not None:
            state_vector = self.state_next_state_buffer[i][0]
            next_state_vector = self.state_next_state_buffer[i][1]

            return torch.from_numpy(full_state_vector).float(),\
                   torch.from_numpy(state_vector).float(),\
                   torch.from_numpy(next_state_vector).float()

        return torch.from_numpy(full_state_vector).float()