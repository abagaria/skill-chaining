# Python imports.
from collections import deque
import random
import numpy as np

# Other imports.
from simple_rl.agents.func_approx.ddpg.hyperparameters import BUFFER_SIZE, BATCH_SIZE

class ReplayBuffer(object):
    def __init__(self, buffer_size=BUFFER_SIZE, name_buffer='', seed=0,
                 prioritize_positive_terminal_transitions=True):
        self.buffer_size = buffer_size
        self.num_exp = 0
        self.memory = deque(maxlen=buffer_size)
        self.name = name_buffer
        self.prioritize_positive_terminal_transitions = prioritize_positive_terminal_transitions
        self.positive_transitions = []

        self.seed = seed
        random.seed(seed)
        np.random.seed(seed)

    def add(self, state, action, reward, next_state, terminal):
        assert isinstance(state, np.ndarray) and isinstance(action, np.ndarray) and \
               isinstance(reward, (int, float)) and isinstance(next_state, np.ndarray)
        experience = state, action, reward, next_state, terminal
        self.memory.append(experience)
        self.num_exp += 1

        if self.prioritize_positive_terminal_transitions:
            if terminal == 1 and reward > 0:
                self.positive_transitions.append(experience)

    def _append_positive_transition(self, *, states, actions, rewards, next_states, dones, pos_transitions):
        def _unsqueeze(array):
            return array[None, ...]

        if len(pos_transitions) > 0:
            # Create tensors corresponding to the sampled positive transition
            pos_transition = random.sample(pos_transitions, k=1)[0]
            pos_state = _unsqueeze(pos_transition[0])
            pos_action = np.array([pos_transition[1]])
            pos_reward = np.array([pos_transition[2]])
            pos_next_state = _unsqueeze(pos_transition[3])
            pos_done = np.array([float(pos_transition[4])])
            assert pos_done == 1, pos_done

            # Add the positive transition tensor to the mini-batch
            states = np.concatenate((states, pos_state), axis=0)
            actions = np.concatenate((actions, pos_action), axis=0)
            rewards = np.concatenate((rewards, pos_reward), axis=0)
            next_states = np.concatenate((next_states, pos_next_state), axis=0)
            dones = np.concatenate((dones, pos_done), axis=0)

            # Shuffle the mini-batch to maintain the IID property
            idx = np.random.permutation(states.shape[0])
            states = states[idx, :]
            actions = actions[idx, :]
            rewards = rewards[idx]
            next_states = next_states[idx, :]
            dones = dones[idx]

        return states, actions, rewards, next_states, dones

    def size(self):
        return self.buffer_size

    def __len__(self):
        return self.num_exp

    def __getitem__(self, i):
        return self.memory[i]

    # get_tensor is here for compatability with DQNAgentClass.ReplayBuffer::sample
    def sample(self, batch_size=BATCH_SIZE, get_tensor=True):
        if self.num_exp < batch_size:
            batch = random.sample(self.memory, self.num_exp)
        else:
            batch = random.sample(self.memory, batch_size)

        state, action, reward, next_state, terminal = map(np.stack, zip(*batch))

        if self.prioritize_positive_terminal_transitions:
            state, action, reward, next_state, terminal = self._append_positive_transition(states=state,
                                                                                           actions=action,
                                                                                           rewards=reward,
                                                                                           next_states=next_state,
                                                                                           dones=terminal,
                                                                                           pos_transitions=self.positive_transitions)

        return state, action, reward, next_state, terminal

    def clear(self):
        self.memory = deque(maxlen=self.buffer_size)
        self.num_exp = 0
