import numpy as np
from torch.utils.data import Dataset


class ReplayDataset(Dataset):
    def __init__(self, states, targets):
        self.input_data = states.astype(np.float32)
        self.output_labels = targets.astype(np.float32)

    def __len__(self):
        assert len(self.input_data) == len(self.output_labels)
        return len(self.input_data)

    def __getitem__(self, i):
        return self.input_data[i], self.output_labels[i]
