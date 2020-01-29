# Python imports.
import torch
import pdb
import numpy as np
import matplotlib.pyplot as plt

# Other imports.
from simple_rl.agents.func_approx.exploration.optimism.latent.CountingLatentSpaceClass import CountingLatentSpace
from simple_rl.agents.func_approx.exploration.optimism.latent.datasets.mnist_dataset import MNISTDataset


class Experiment3:
    def __init__(self, epsilon, classes=tuple(range(10)), num_examples=np.inf, seed=0):
        self.classes = classes
        self.num_examples = num_examples
        np.random.seed(seed)

        self.counting_space = CountingLatentSpace(2, epsilon, phi_type="function",
                                                  experiment_name="exp3", pixel_observations=True)

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

    def run_experiment(self):
        action_buffers = self.generate_data()

        self.counting_space.train(action_buffers=action_buffers, epochs=100)

        score_array = []

        for action_idx, action_buffer in enumerate(action_buffers):
            action_counts = [self.counting_space.get_counts(action_buffer, c).mean() for c in self.classes]
            score_array.append(action_counts)

        score_array = np.array(score_array)

        print("Generated counts:\n", np.round(score_array, 2))
        plt.imshow(score_array)
        plt.xticks([])
        plt.yticks([])
        plt.colorbar()
        plt.show()

        states = np.vstack(action_buffers)
        states_repr = self.counting_space.extract_features(states)

        for buffer_idx in range(len(action_buffers)):
            plt.subplot(1, 3, buffer_idx + 1)
            plt.scatter(states_repr[:, 0], states_repr[:, 1], c=range(len(states)), alpha=0.6)
            plt.colorbar()

        plt.show()

        most_similar_zero_idx = self.get_most_similar_state_idx(states_repr[:self.num_examples])
        most_dissimilar_zero_idx = self.get_most_dissimilar_state_idx(states_repr[:self.num_examples])
        most_dissimilar_image_idx = self.get_most_dissimilar_state_idx(states_repr)

        most_similar_zero = states[most_similar_zero_idx]
        most_dissimilar_zero = states[most_dissimilar_zero_idx]
        most_dissimilar_image = states[most_dissimilar_image_idx]

        plt.subplot(2, 2, 1)
        plt.imshow(states[0].squeeze(0), cmap="binary")
        plt.title("Query image")
        plt.xticks([])
        plt.yticks([])
        plt.subplot(2, 2, 2)
        plt.imshow(most_similar_zero.squeeze(0), cmap="binary")
        plt.title("Most similar zero")
        plt.xticks([])
        plt.yticks([])
        plt.subplot(2, 2, 3)
        plt.imshow(most_dissimilar_zero.squeeze(0), cmap="binary")
        plt.title("Most dissimilar zero")
        plt.xticks([])
        plt.yticks([])
        plt.subplot(2, 2, 4)
        plt.imshow(most_dissimilar_image.squeeze(0), cmap="binary")
        plt.title("Most dissimilar image")
        plt.xticks([])
        plt.yticks([])
        plt.show()


if __name__ == "__main__":
    exp = Experiment3(0.2, (0, 1, 2), 20, seed=1)
    exp.run_experiment()
