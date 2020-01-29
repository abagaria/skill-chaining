from torch.utils.data import Dataset, ConcatDataset
import numpy as np
import itertools
import torch
import pdb


class CountingDataset(Dataset):
    """ PyTorch data set wrapper around state buffers. """

    def __init__(self, *, full_buffer, action_buffer):
        super(CountingDataset, self).__init__()

        self.full_buffer = full_buffer
        self.action_buffer = action_buffer
        self.indices = self.create_idx_buffer()

    def create_idx_buffer(self):
        full_buffer_idx = range(len(self.full_buffer))
        action_buffer_idx = range(len(self.action_buffer))

        return list(itertools.product(full_buffer_idx, action_buffer_idx))

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        idx = self.indices[i]

        full_state_vector = self.full_buffer[idx[0]]
        action_state_vector = self.action_buffer[idx[1]]

        return torch.from_numpy(full_state_vector).float(),\
               torch.from_numpy(action_state_vector).float()


class MultiActionCountingDataset(ConcatDataset):
    def __init__(self, *, action_buffers):
        """ See https://pytorch.org/docs/stable/_modules/torch/utils/data/dataset.html#ConcatDataset"""
        all_data = np.concatenate(action_buffers)
        datasets = [CountingDataset(full_buffer=all_data, action_buffer=ab) for ab in action_buffers]
        super(MultiActionCountingDataset, self).__init__(datasets)


class StateNextStateDataset(Dataset):
    """A dataset that takes in parallel arrays of states and next-states, and returns them in pairs."""

    def __init__(self, state_next_state_buffer):
        super(StateNextStateDataset, self).__init__()
        self.state_next_state_buffer = state_next_state_buffer

    def __len__(self):
        return len(self.state_next_state_buffer)

    def __getitem__(self, i):
        return torch.from_numpy(self.state_next_state_buffer[i][0]).float(),\
               torch.from_numpy(self.state_next_state_buffer[i][1]).float()



if __name__ == "__main__":
    f = np.zeros((10, 2))
    a = np.ones((20, 2))
    dset = CountingDataset(full_buffer=f, action_buffer=a)
