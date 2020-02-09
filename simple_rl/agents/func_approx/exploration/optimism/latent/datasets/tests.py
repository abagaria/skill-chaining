import numpy as np
from .transition_consistency_dataset import TransitionConsistencyDataset


def test_transition_dataset():
    s0 = np.array([1., 2., 3., 4.])
    s1 = np.array([2., 3., 5., 10.])
    s2 = np.array([3., 4., 8., 7.])

    action_buffers = [[(s0, s1), (s0, s1), (s1, s2)], [(s1, s2), (s1, s0), (s0, s2)]]
    import torch

    device = torch.device("cuda")
    dset = TransitionConsistencyDataset(action_buffers, device)

    for dpoint in dset:
        assert len(dpoint) == 3
        assert dpoint[0].shape == torch.Size([4])
        assert dpoint[1].shape == torch.Size([4])
        assert dpoint[2].shape == torch.Size([3, 2, 4])
