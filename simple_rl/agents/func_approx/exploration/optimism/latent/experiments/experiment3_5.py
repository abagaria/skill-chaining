# Python imports.
import torch
import pdb
import numpy as np
import matplotlib.pyplot as plt
import itertools

# Other imports.
from simple_rl.agents.func_approx.exploration.optimism.latent.CountingLatentSpaceClass import CountingLatentSpace
from simple_rl.agents.func_approx.exploration.optimism.latent.datasets.mnist_dataset import MNISTDataset


class Experiment3_5:
    def __init__(self, epsilon, classes=tuple(range(10)), num_examples=np.inf, seed=0, lam=0.0):
        self.classes = classes
        self.num_examples = num_examples
        self.lam = lam
        np.random.seed(seed)

        self.counting_space = CountingLatentSpace(2, epsilon, phi_type="function",
                                                  experiment_name="exp3_5", pixel_observations=True, lam=lam)

    def generate_data(self):
        data_set = MNISTDataset("train", self.classes, self.num_examples)
        return data_set()

    @staticmethod
    def get_most_similar_state_idx(buffer):
        """
        Args:
            buffer (np.ndarray): size num_states x 1 x 28 x 28 for MNIST

        Returns:
            similar_idx (int): state most similar to buffer[0]
        """
        assert isinstance(buffer, np.ndarray)
        query_image = buffer[0]
        sq_distances = ((query_image - buffer) ** 2).sum(axis=1)
        similar_idx = np.argmin(sq_distances[1:]) + 1
        return similar_idx

    @staticmethod
    def get_most_dissimilar_state_idx(buffer):
        assert isinstance(buffer, np.ndarray)
        query_image = buffer[0]
        sq_distances = ((query_image - buffer) ** 2).sum(axis=1)
        dissimilar_idx = np.argmax(sq_distances)
        return dissimilar_idx

    @staticmethod
    def _make_rings_from_buffer(buffer):
        one_after = np.vstack((buffer[1:], buffer[0][None, ...]))
        zipped = list(zip(buffer, one_after))
        return zipped

    def run_experiment(self):
        action_buffers = self.generate_data()

        ring_buffers = [self._make_rings_from_buffer(b) for b in action_buffers]
        ring_buffers = list(itertools.chain.from_iterable(ring_buffers))

        states = np.array(list(itertools.chain.from_iterable(action_buffers)))

        self.counting_space.train(action_buffers=action_buffers, state_next_state_buffer=ring_buffers, epochs=20)

        states_repr = self.counting_space.extract_features(states)

        c_arr = [int(i / self.num_examples) for i in range(len(states))]

        plt.scatter(states_repr[:, 0], states_repr[:, 1], c=c_arr, alpha=0.3)
        plt.colorbar()
        plt.title("Ring Buffer MNIST")

        plt.show()


if __name__ == "__main__":
    exp = Experiment3_5(0.2, (0, 1, 2), 20, seed=1, lam=10)
    # exp = Experiment3_5(0.2, tuple(range(10)), 20, seed=1, lam=10)
    exp.run_experiment()
