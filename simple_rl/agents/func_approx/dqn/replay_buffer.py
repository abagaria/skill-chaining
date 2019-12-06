from collections import namedtuple, deque
import random
import torch
import numpy as np
import pdb


Transition = namedtuple('Transition', ("state", "action", "reward", "next_state", "done", "num_steps"))
Experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done", "num_steps"])

class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, action_size, buffer_size, batch_size, seed, device, pixel_observation):
        """
        Initialize a ReplayBuffer object.
        Args:
            action_size (int): dimension of each action
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
            seed (int): random seed
            device (torch.device): cpu / cuda:0 / cuda:1
            pixel_observation (bool): Whether observations are dense or images
        """
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.experience = Experience
        self.seed = random.seed(seed)
        self.device = device
        self.pixel_observation = pixel_observation

        self.positive_transitions = []
        self.num_sampled_positive_transitions = [0]

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

    def append_positive_transition(self, states, actions, rewards, next_states, dones, steps, pos_transitions):
        if len(pos_transitions) > 0:
            def _to_floating_tensor(np_array):
                return torch.from_numpy(np_array).float().unsqueeze(0).to(self.device)
            def _to_long_tensor(np_array):
                return torch.from_numpy(np_array).unsqueeze(0).to(self.device)

            # Create tensors corresponding to the sampled positive transition
            pos_transition = random.sample(pos_transitions, k=1)[0]  # type: Experience
            pos_state = _to_floating_tensor(pos_transition.state)
            pos_action = _to_long_tensor(np.array([pos_transition.action]))
            pos_reward = _to_floating_tensor(np.array([pos_transition.reward]))
            pos_next_state = _to_floating_tensor(pos_transition.next_state)
            pos_done = _to_floating_tensor(np.array([float(pos_transition.done)]))
            assert pos_done == 1, pos_done
            pos_steps = _to_floating_tensor(np.array([1]))

            # Add the positive transition tensor to the mini-batch
            states = torch.cat((states, pos_state), dim=0)
            actions = torch.cat((actions, pos_action), dim=0)
            rewards = torch.cat((rewards, pos_reward), dim=0)
            next_states = torch.cat((next_states, pos_next_state), dim=0)
            dones = torch.cat((dones, pos_done), dim=0)
            steps = torch.cat((steps, pos_steps), dim=0)

            # Shuffle the mini-batch to maintain the IID property
            idx = torch.randperm(states.shape[0])
            states = states[idx, :]
            actions = actions[idx, :]
            rewards = rewards[idx, :]
            next_states = next_states[idx, :]
            dones = dones[idx, :]
            steps = steps[idx, :]

        return states, actions, rewards, next_states, dones, steps

    def sample(self, batch_size=None):
        """Randomly sample a batch of experiences from memory."""
        size = self.batch_size if batch_size is None else batch_size
        experiences = random.sample(self.memory, k=size)

        # Log the number of times we see a non-negative reward (should be sparse)
        positive_transitions = [transition for transition in experiences if transition.reward > 0]

        if len(positive_transitions) > 0:
            self.positive_transitions += positive_transitions

        # With image observations, we need to add another dimension to the tensor before stacking
        if self.pixel_observation:
            states = torch.from_numpy(np.vstack([e.state[None, ...] for e in experiences if e is not None])).float().to(self.device)
        else:
            states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(self.device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(self.device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(self.device)
        if self.pixel_observation:
            next_states = torch.from_numpy(np.vstack([e.next_state[None, ...] for e in experiences if e is not None])).float().to(self.device)
        else:
            next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(self.device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(self.device)
        steps = torch.from_numpy(np.vstack([e.num_steps for e in experiences if e is not None])).float().to(self.device)

        states, actions, rewards, next_states, dones, steps = self.append_positive_transition(states, actions, rewards,
                                                                                              next_states, dones, steps,
                                                                                              self.positive_transitions)

        self.num_sampled_positive_transitions.append(sum(rewards > 0))


        return states, actions, rewards, next_states, dones, steps

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)
