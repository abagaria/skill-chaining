import torch
import numpy as np
import ipdb
from torch.utils.data import Dataset
import random
import itertools

"""

What's going on?
We need to pass in a dataset. The method involves getting two shuffled index-lists.
You chunk both up into chunks of size 32. Then, you have an index for each chunk, i and j for the two lists.
You itertools.product i and j together. Then, when someone gets something from this list, it indexes an
entire block, which represents an outer product. You return the two lists of elements that make that
outer product happen.

Ideally, you also need the action-buffer which things in the second list come from. That way, you know
which bonus term that pair goes towards later.

"""

class ChunkedStateDataset(Dataset):
    def __init__(self, buffers, state_next_state_buffer, chunk_size):
        """
        Combine action buffers `buffers` into a single buffer and then sample from it.
        If `state_next_state_buffer` is given, then also return samples from it.

        We also need to chunk the state_next_state buffer...

        Args:
            buffers (list): each element is a list/np.ndarray of states seen under that action_idx
            state_next_state_buffer (list): each element is a tuple of np.ndarray representing states
        """
        super(ChunkedStateDataset, self).__init__()

        print(f"Within approx bonus dataset: chunk size is {chunk_size}")

        self.chunk_size = chunk_size

        self.action_buffers = buffers
        self.s_a_buffer = self._create_flattened_action_buffer()

        self.state_next_state_buffer = state_next_state_buffer
        self.set_indices()

    def set_indices(self):

        def _shuffle_data(data):
            """ In place data shuffling. """
            return random.sample(data, len(data))

        self.first_set_of_chunks = self._chunk_flattened_action_buffer(_shuffle_data(self.s_a_buffer))
        self.second_set_of_chunks = self._chunk_flattened_action_buffer(_shuffle_data(self.s_a_buffer))

        self.sns_chunks = self._chunk_flattened_action_buffer(_shuffle_data(self.state_next_state_buffer))

        first_indices = list(range(len(self.first_set_of_chunks)))
        second_indices = list(range(len(self.second_set_of_chunks)))

        self.indices = list(itertools.product(first_indices, second_indices))
        random.shuffle(self.indices)

        assert len(self.indices) == len(self.sns_chunks)**2

    def _chunk_flattened_action_buffer(self, flattened_action_buffer):
        """

        Args:
            flattened_action_buffer: A list of tuples of (state, action_idx)
            chunk_size: We're making chunks of this size.

        Returns:
            A list of lists of (state, action_idx) tuples. Each sub-list is a chunk.
        """
        chunk_size = self.chunk_size

        num_chunks = int(np.ceil(len(flattened_action_buffer) / chunk_size))
        chunked = [flattened_action_buffer[chunk_size*c:chunk_size*(c+1)] for c in range(num_chunks)]
        total_length = sum(len(c) for c in chunked)
        assert total_length == len(flattened_action_buffer)
        return chunked

    def _create_flattened_action_buffer(self):
        flattened_action_buffers = []
        for action_idx, action_buffer in enumerate(self.action_buffers):
            for state in action_buffer:
                flattened_action_buffers.append((state, action_idx))
        return flattened_action_buffers

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        indices = self.indices[i]
        first_chunk = self.first_set_of_chunks[indices[0]]
        second_chunk = self.second_set_of_chunks[indices[1]]
        sns_chunk = self.sns_chunks[i % len(self.sns_chunks)]

        first_chunk_states = np.array([sa[0] for sa in first_chunk])
        first_chunk_actions = np.array([sa[1] for sa in first_chunk])
        second_chunk_states = np.array([sa[0] for sa in second_chunk])
        second_chunk_actions = np.array([sa[1] for sa in second_chunk])

        sns_state_chunk = np.vstack([sns[0] for sns in sns_chunk])
        sns_next_state_chunk = np.vstack([sns[1] for sns in sns_chunk])

        fc_state_tensor = torch.from_numpy(first_chunk_states).float()
        fc_action_tensor = torch.from_numpy(first_chunk_actions).float()
        sc_state_tensor = torch.from_numpy(second_chunk_states).float()
        sc_action_tensor = torch.from_numpy(second_chunk_actions).float()

        sns_state_tensor = torch.from_numpy(sns_state_chunk).float()
        sns_next_state_tensor = torch.from_numpy(sns_next_state_chunk).float()

        return (fc_state_tensor, fc_action_tensor, sc_state_tensor,
                sc_action_tensor, sns_state_tensor, sns_next_state_tensor)
