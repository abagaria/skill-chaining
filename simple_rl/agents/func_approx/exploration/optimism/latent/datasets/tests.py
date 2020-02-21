import numpy as np
import torch

from simple_rl.agents.func_approx.exploration.optimism.latent.datasets.transition_consistency_dataset import TransitionConsistencyDataset
from simple_rl.agents.func_approx.exploration.optimism.latent.datasets.chunked_state_dataset import ChunkedStateDataset


def generate_action_buffers():
    s0 = np.array([1., 2., 3., 4.])
    s1 = np.array([2., 3., 5., 10.])
    s2 = np.array([3., 4., 8., 7.])

    s1, s2, s3, s4, s5, s6 = np.arange(24).reshape(6,4)
    action_buffers = [[s1,s2,s3],[s4,s5,s6]]

    # action_buffers = [[s0, s1, s2], [s2, s0, s2]]

    return action_buffers


def test_transition_dataset():

    action_buffers = generate_action_buffers()

    device = torch.device("cuda")
    dset = TransitionConsistencyDataset(action_buffers, device)

    for dpoint in dset:
        assert len(dpoint) == 3
        assert dpoint[0].shape == torch.Size([4])
        assert dpoint[1].shape == torch.Size([4])
        assert dpoint[2].shape == torch.Size([3, 2, 4])


if __name__ == "__main__":
    action_buffers = generate_action_buffers()

    dset = ChunkedStateDataset(action_buffers, [], chunk_size=2)

    for _ in range(2):
        print("\n\n\n")

        for dpoint in dset:
            print(dpoint)
            # break
        dset.set_indices()


