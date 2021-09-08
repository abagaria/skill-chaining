import ipdb
import torch
import random
import itertools
import numpy as np
from pfrl.replay_buffer import batch_experiences
from pfrl.replay_buffers.replay_buffer import ReplayBuffer

# TODO: Mixed Monte Carlo
# TODO: Maybe store the n-step intrinsic rewards

def phi(x):
    return np.asarray(x, dtype=np.float32) / 255.


def torch_phi(x, device):
    return torch.as_tensor(x).to(device).float() / 255.


class NStepReplayBuffer(ReplayBuffer):
    def __init__(self, gamma, max_size, n_steps, device, phi=phi):
        self.phi = phi
        self.gamma = gamma
        self.device = device
        self.n_steps = n_steps
        self.max_size = max_size

        super(NStepReplayBuffer, self).__init__(max_size, n_steps)

    def append(self, state, action, reward, next_state, done, **kwargs):
        """ Add transition to the replay buffer. """

        super(NStepReplayBuffer, self).append(state, action, reward, next_state, is_state_terminal=done, **kwargs)

    def sample(self, batch_size):
        def get_all_elem_from_traj(traj, key):
            if key == "state":  # Convert observations from LazyFrames to arrays
                return [np.array(transition[key]) for transition in traj]
            return [transition[key] for transition in traj]

        # List of size batch_size where each element is a list of dictionaries of size n
        n_step_trajectories = super(NStepReplayBuffer, self).sample(num_experiences=batch_size)

        # -- Dictionary with the following keys (each is a tensor):
        # -- state and action as you would regularly have
        # -- reward is the n-step return
        # -- next_state is the final state *after* the n-steps
        # -- discount is the tensor of n-step discount factors (gamma ** n)
        batch_exp = batch_experiences(n_step_trajectories, self.device, self.phi, self.gamma)

        n_steps = np.zeros((batch_size,), dtype=np.uint8)
        n_states = np.zeros((batch_size, self.n_steps, 4, 84, 84), dtype=np.uint8)
        n_actions = np.zeros((batch_size, self.n_steps), dtype=np.long)

        for i, n_step_traj in enumerate(n_step_trajectories):
            n = len(n_step_traj)
            n_steps[i] = n
            n_states[i, :n, ...] = get_all_elem_from_traj(n_step_traj, "state")
            n_actions[i, :n] = get_all_elem_from_traj(n_step_traj, "action")

        batch_exp["n_states"] = torch_phi(n_states, device=self.device)
        batch_exp["n_steps"] = torch.as_tensor(n_steps, device=self.device, dtype=torch.uint8)
        batch_exp["n_actions"] = torch.as_tensor(n_actions).to(self.device).long()

        return batch_exp
