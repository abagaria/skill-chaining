# Python imports.
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from collections import defaultdict
import gym

# Other imports.
from simple_rl.agents.func_approx.exploration.optimism.latent.CountingLatentSpaceClass import CountingLatentSpace


class Experiment4:
    def __init__(self, epsilon, num_steps=200, seed=0, lam=0.1, use_bonus=False):
        self.num_steps = num_steps
        self.env = gym.make("MountainCar-v0")
        self.env.seed(seed)

        loss_type = "counts" if not use_bonus else "bonus"

        self.counting_space = CountingLatentSpace(2, epsilon, phi_type="function",
                                                  experiment_name="exp4", pixel_observations=False, lam=lam,
                                                  optimization_quantity=loss_type)

    @staticmethod
    def _get_action_buffers(state_action_dict):
        action_buffers = []
        sorted_keys = sorted(state_action_dict.keys())
        for action in sorted_keys:
            states = state_action_dict[action]
            states_array = np.array(states)
            action_buffers.append(states_array)
        return action_buffers

    def generate_data(self):
        state = self.env.reset()
        state_next_buffer = []
        state_action_dict = defaultdict(list)
        for _ in tqdm(range(self.num_steps), desc="Collecting Data"):
            a = self.env.action_space.sample()
            next_state, _, done, _ = self.env.step(a)
            state_next_buffer.append((state, next_state))
            state_action_dict[a].append(state)
            state = next_state
            if done:
                state = self.env.reset()
        action_buffers = self._get_action_buffers(state_action_dict)

        normalized_state_next_buffer, normalized_action_buffers = self._pre_process_data(state_next_buffer, action_buffers)

        return normalized_state_next_buffer, normalized_action_buffers

    @staticmethod
    def _pre_process_data(state_next_state_buffer, action_buffers):
        def get_mean_std(s):
            mean = np.mean(s, axis=0)
            std = np.std(s, axis=0)
            return mean, std

        def normalize(s):
            mean, std = get_mean_std(s)
            assert mean.shape == (2,), mean.shape
            assert std.shape == (2,), std.shape
            return (s - mean) / (std + 1e-6)

        states = np.array([ss[0] for ss in state_next_state_buffer])
        next_states = np.array([ss[1] for ss in state_next_state_buffer])

        normalized_states = normalize(states)
        normalized_next_states = normalize(next_states)
        normalized_action_buffers = [normalize(action_buffer) for action_buffer in action_buffers]
        normalized_state_next_state_buffer = list(zip(normalized_states, normalized_next_states))

        return normalized_state_next_state_buffer, normalized_action_buffers

    def run_experiment(self):
        state_next_state_buffer, action_buffers = self.generate_data()

        states = np.array([ss[0] for ss in state_next_state_buffer])

        self.counting_space.train(action_buffers=action_buffers, state_next_state_buffer=state_next_state_buffer, epochs=0)

        states_repr = self.counting_space.extract_features(states)

        for buffer_idx in range(len(action_buffers)):
            counts = self.counting_space.get_counts(states, buffer_idx)
            plt.subplot(2, 3, buffer_idx + 1)
            plt.scatter(states[:, 0], states[:, 1], c=counts)
            plt.colorbar()
            plt.subplot(2, 3, buffer_idx + 4)
            plt.scatter(states_repr[:, 0], states_repr[:, 1], c=range(len(states)), alpha=0.3)

            plt.colorbar()
        plt.show()

        self.counting_space.train(action_buffers=action_buffers, state_next_state_buffer=state_next_state_buffer, epochs=10)

        states = np.array([ss[0] for ss in state_next_state_buffer])

        states_repr = self.counting_space.extract_features(states)

        for buffer_idx in range(len(action_buffers)):
            counts = self.counting_space.get_counts(states, buffer_idx)
            plt.subplot(2, 3, buffer_idx + 1)
            plt.scatter(states[:, 0], states[:, 1], c=counts, alpha=0.3)
            plt.colorbar()
            plt.subplot(2, 3, buffer_idx + 4)
            plt.scatter(states_repr[:, 0], states_repr[:, 1], c=range(len(states)), alpha=0.3)

            plt.colorbar()
        plt.show()


if __name__  == '__main__':
    exp = Experiment4(0.3, 3000, lam=10., use_bonus=True)
    exp.run_experiment()
