from collections import defaultdict
import numpy as np
import torch
from torch.utils.data import Dataset


class TransitionConsistencyDataset(Dataset):
    """ Dataset encoding (s, s') buffers under different actions. """

    def __init__(self, action_buffers, device):
        """

        Args:
            action_buffers (list): each element of this list is a single action's buffer.
                                   Each action buffer is a list of s,s' tuples under that action.
            device (str)
        """
        super(TransitionConsistencyDataset, self).__init__()
        self.device = device

        self.action_buffers = action_buffers
        self.s_a_sprime_buffer = self._create_flattened_action_buffer()
        self.action_buffer_tensors = self._create_action_buffer_tensors()

    def _create_flattened_action_buffer(self):
        flattened_action_buffers = []
        for action_idx, action_buffer in enumerate(self.action_buffers):
            for s, s_prime in action_buffer:
                flattened_action_buffers.append((s, action_idx, s_prime))
        return flattened_action_buffers

    def _create_action_buffer_tensors(self):
        list_of_action_buffer_tensors = []
        for action_idx, action_buffer in enumerate(self.action_buffers):
            action_buffer_numpy = np.array(action_buffer)
            action_buffer_tensor = torch.from_numpy(action_buffer_numpy).float().to(self.device)
            list_of_action_buffer_tensors.append(action_buffer_tensor)
        return list_of_action_buffer_tensors

    def get_action_buffer_tensor(self, buffer_idx):
        return self.action_buffer_tensors[buffer_idx]

    def get_action_buffer_tensor_batched(self, buffer_idx_tensor):
        return [self.action_buffer_tensors[buffer_idx.item()] for buffer_idx in buffer_idx_tensor]

    def __len__(self):
        return len(self.s_a_sprime_buffer)

    def __getitem__(self, i):
        state, action, next_state = self.s_a_sprime_buffer[i]

        return state, next_state, action


def collate_fn(data):
    """

    Args:
        data (list): list of tuples of the form <state, next_state, action>

    Returns:
        action_to_ssp (dict): dictionary that given an action, returns a tensor
                              of shape (batch_size, 2, num_features).
    """
    action_to_ssp = defaultdict(list)
    for state, next_state, action in data:
        action_to_ssp[action].append((state, next_state))

    action_to_ssp_tensor = dict()
    for action in action_to_ssp:
        sns_pairs = action_to_ssp[action]
        sns_np = np.array(sns_pairs)
        action_to_ssp_tensor[action] = torch.from_numpy(sns_np).float()


    return action_to_ssp_tensor
