from collections import namedtuple, deque
import random
import torch
import numpy as np
from simple_rl.tasks.gym.wrappers import LazyFrames
import ipdb


Transition = namedtuple('Transition', ("state", "action", "reward", "next_state", "done", "num_steps"))


class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, action_size, buffer_size, batch_size, seed, device, pixel_observation,
                 prioritize_positive_terminal_transitions=False):
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
        self.experience = namedtuple("Experience", field_names=["state", "position", "action", "reward", "next_state",
                                                                "next_position", "done", "num_steps"])
        self.seed = random.seed(seed)
        np.random.seed(seed)
        self.device = device
        self.pixel_observation = pixel_observation
        self.prioritize_positive_terminal_transitions = prioritize_positive_terminal_transitions

        self.positive_transitions = []

    def add(self, state, position, action, reward, next_state, next_position, done, num_steps):
        """
        Add new experience to memory.
        Args:
            state (np.array): We add numpy arrays from gym env to the buffer, but sampling from buffer returns tensor
            position (np.array)
            action (int)
            reward (float)
            next_state (np.array)
            next_position (np.array)
            done (bool)
            num_steps (int): number of steps taken by the action/option to terminate
        """
        e = self.experience(state, position, action, reward, next_state, next_position, done, num_steps)
        self.memory.append(e)

        if self.prioritize_positive_terminal_transitions:
            if done == 1 and reward > 0:
                self.positive_transitions.append(e)

    def _append_positive_transition(self, *, states, positions, actions, rewards, next_states,
                                    next_positions, dones, steps, pos_transitions):
        if len(pos_transitions) > 0:
            def _to_floating_tensor(sample):
                np_array = np.array(sample) if isinstance(sample, LazyFrames) else sample
                return torch.from_numpy(np_array).float().unsqueeze(0).to(self.device)

            def _to_long_tensor(np_array):
                return torch.from_numpy(np_array).unsqueeze(0).to(self.device)

            # Create tensors corresponding to the sampled positive transition
            pos_transition = random.sample(pos_transitions, k=1)[0]  # type: Experience
            pos_state = _to_floating_tensor(pos_transition.state)
            pos_position = _to_floating_tensor(pos_transition.position)
            pos_action = _to_long_tensor(np.array([pos_transition.action]))
            pos_reward = _to_floating_tensor(np.array([pos_transition.reward]))
            pos_next_state = _to_floating_tensor(pos_transition.next_state)
            pos_next_position = _to_floating_tensor(pos_transition.next_position)
            pos_done = _to_floating_tensor(np.array([float(pos_transition.done)]))
            assert pos_done == 1, pos_done
            pos_steps = _to_floating_tensor(np.array([1]))

            # Add the positive transition tensor to the mini-batch
            states = torch.cat((states, pos_state), dim=0)
            positions = torch.cat((positions, pos_position), dim=0)
            actions = torch.cat((actions, pos_action), dim=0)
            rewards = torch.cat((rewards, pos_reward), dim=0)
            next_states = torch.cat((next_states, pos_next_state), dim=0)
            next_positions = torch.cat((next_positions, pos_next_position), dim=0)
            dones = torch.cat((dones, pos_done), dim=0)
            steps = torch.cat((steps, pos_steps), dim=0)

            # Shuffle the mini-batch to maintain the IID property
            idx = torch.randperm(states.shape[0])
            states = states[idx, :]
            positions = positions[idx, :]
            actions = actions[idx, :]
            rewards = rewards[idx, :]
            next_states = next_states[idx, :]
            next_positions = next_positions[idx, :]
            dones = dones[idx, :]
            steps = steps[idx, :]

        return states, positions, actions, rewards, next_states, next_positions, dones, steps

    def sample(self, batch_size=None):
        """Randomly sample a batch of experiences from memory."""
        size = self.batch_size if batch_size is None else batch_size
        experiences = random.sample(self.memory, k=size)

        # With image observations, we need to add another dimension to the tensor before stacking
        if self.pixel_observation:
            states = torch.from_numpy(np.vstack([e.state[None, ...] for e in experiences if e is not None])).float().to(self.device)
        else:
            states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(self.device)
        positions = torch.from_numpy(np.vstack([e.position for e in experiences if e is not None])).float().to(self.device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(self.device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(self.device)
        if self.pixel_observation:
            next_states = torch.from_numpy(np.vstack([e.next_state[None, ...] for e in experiences if e is not None])).float().to(self.device)
        else:
            next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(self.device)
        next_positions = torch.from_numpy(np.vstack([e.next_position for e in experiences if e is not None])).float().to(self.device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(self.device)
        steps = torch.from_numpy(np.vstack([e.num_steps for e in experiences if e is not None])).float().to(self.device)

        if self.prioritize_positive_terminal_transitions:
            states, positions, actions, rewards, next_states, next_positions, dones, steps = self._append_positive_transition(
                                                                                                    states=states, positions=positions,
                                                                                                    actions=actions, rewards=rewards,
                                                                                                    next_states=next_states, next_positions=next_positions,
                                                                                                    dones=dones, steps=steps,
                                                                                                    pos_transitions=self.positive_transitions
                                                                                                )

        return states, positions, actions, rewards, next_states, next_positions, dones, steps

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)

    def __getitem__(self, i):
        return self.memory[i]
