from collections import namedtuple, deque
import random
import torch
import numpy as np
import pdb


Transition = namedtuple('Transition', ("state", "action", "reward", "next_state", "done", "num_steps"))


class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, action_size, buffer_size, batch_size, seed, device):
        """
        Initialize a ReplayBuffer object.
        Args:
            action_size (int): dimension of each action
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
            seed (int): random seed
            device (torch.device): cpu / cuda:0 / cuda:1
        """
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done", "num_steps"])
        self.seed = random.seed(seed)
        self.device = device

        self.positive_transitions = []

    def add(self, state, action, reward, next_state, done, num_steps):
        """
        Add new experience to memory.
        Args:
            state (np.array): We add numpy arrays from gym env to the buffer, but sampling from buffer returns tensor
            action (int)
            reward (float_
            next_state (np.array)
            done (bool)
            num_steps (int): number of steps taken by the action/option to terminate
        """
        e = self.experience(state, action, reward, next_state, done, num_steps)
        self.memory.append(e)

    def sample(self, batch_size=None):
        """Randomly sample a batch of experiences from memory."""
        size = self.batch_size if batch_size is None else batch_size
        experiences = random.sample(self.memory, k=size)

        # Log the number of times we see a non-negative reward (should be sparse)
        num_positive_transitions = sum([exp.reward >= 0 for exp in experiences])
        self.positive_transitions.append(num_positive_transitions)

        states = torch.from_numpy(np.vstack([e.state[None, ...] for e in experiences if e is not None])).float().to(self.device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(self.device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(self.device)
        next_states = torch.from_numpy(np.vstack([e.next_state[None, ...] for e in experiences if e is not None])).float().to(self.device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(self.device)
        steps = torch.from_numpy(np.vstack([e.num_steps for e in experiences if e is not None])).float().to(self.device)

        return states, actions, rewards, next_states, dones, steps

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)


class ImageReplayBuffer(object):
    def __init__(self, capacity, seed):
        self.capacity = capacity
        self.memory = []
        self.position = 0

        # Set seed for sampling
        random.seed(seed)
        np.random.seed(seed)

    def add(self, *args):
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)
