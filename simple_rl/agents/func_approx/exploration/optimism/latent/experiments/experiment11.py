import gym
import numpy as np
from collections import defaultdict
from tqdm import tqdm
import matplotlib.pyplot as plt

from simple_rl.agents.func_approx.exploration.optimism.latent.CountingLatentSpaceClass import CountingLatentSpace

class Experiment11:
    """ Testing transition consistency loss (TC-loss) on mountain car. """

    def __init__(self, epsilon, num_steps=200, seed=0, lam=0.1, num_epochs=1):
        self.num_steps = num_steps
        self.num_epochs = num_epochs
        self.env = gym.make("MountainCar-v0")
        self.env.seed(seed)

        self.counting_space = CountingLatentSpace(2, 3, 2, epsilon, phi_type="function", experiment_name="exp11",
                                                  pixel_observations=False, lam=lam, optimization_quantity="count-tc")

    @staticmethod
    def _get_action_buffers(state_action_dict):
        """ For the TC-loss computation, `action_buffers` is a list of lists. Each component list
            corresponds to the action_idx and is a list of tuples representing (s, s') pairs such
            that s, s' transition happened under action action_idx. """
        action_buffers = []

        # Sorting based on the action is the key step here to ensure that
        # each buffer_idx correctly corresponds to an action
        sorted_keys = sorted(state_action_dict.keys())
        for action in sorted_keys:
            sns_tuples = state_action_dict[action]
            action_buffers.append(sns_tuples)
        return action_buffers

    def generate_data(self):
        state = self.env.reset()

        state_next_buffer = []
        state_action_dict = defaultdict(list)

        for _ in tqdm(range(self.num_steps), desc="Collecting Data"):
            a = self.env.action_space.sample()
            next_state, _, done, _ = self.env.step(a)

            state_next_buffer.append((state, next_state))
            state_action_dict[a].append((state, next_state))

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

        def normalize(s, mean, std):
            assert mean.shape == (2,), mean.shape
            assert std.shape == (2,), std.shape
            return (s - mean) / (std + 1e-6)

        mean_state, std_state = get_mean_std([sns[0] for sns in state_next_state_buffer])

        normalized_tuples = []

        for sns_tuples in action_buffers:
            normalized_sns_list = []
            for s, s_prime in sns_tuples:
                normalized_sns_list.append((normalize(s, mean_state, std_state), normalize(s_prime, mean_state, std_state)))
            normalized_tuples.append(normalized_sns_list)

        # Normalize the s, s' buffer
        states = np.array([ss[0] for ss in state_next_state_buffer])
        next_states = np.array([ss[1] for ss in state_next_state_buffer])

        normalized_states = normalize(states, mean_state, std_state)
        normalized_next_states = normalize(next_states, mean_state, std_state)
        normalized_state_next_state_buffer = list(zip(normalized_states, normalized_next_states))

        return normalized_state_next_state_buffer, normalized_tuples

    def run_experiment(self):
        obs_next_obs, action_buffers = self.generate_data()

        states = np.array([ono[0] for ono in obs_next_obs])

        self.counting_space.train(tc_action_buffers=action_buffers, epochs=self.num_epochs)

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

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--lam", type=float, help="lambda", default=1.0)
    parser.add_argument("--epsilon", type=float, help="Epsilon", default=0.1)
    parser.add_argument("--num_steps", type=int, default=200)
    parser.add_argument("--num_epochs", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)

    args = parser.parse_args()

    exp = Experiment11(args.epsilon, args.num_steps, args.seed, args.lam, args.num_epochs)
    exp.run_experiment()
