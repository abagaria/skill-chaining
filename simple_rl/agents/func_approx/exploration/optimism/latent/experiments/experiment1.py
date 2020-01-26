# Python imports.
import numpy as np
import matplotlib.pyplot as plt

# Other imports.
from simple_rl.agents.func_approx.exploration.optimism.latent.CountingLatentSpaceClass import CountingLatentSpace


class Experiment1:
    def __init__(self, epsilon, phi_type):
        self.counting_space = CountingLatentSpace(epsilon, phi_type)

    @staticmethod
    def generate_data():
        v1 = np.array([0., 0.])
        v2 = np.array([1., 1.])
        v3 = np.array([1.5, 2.])

        return np.vstack((v1, v1, v1, v1, v1, v2, v2, v2, v3, v3))

    def generate_counts_plot(self):
        x = np.arange(0., 3., 0.1)
        y = np.arange(0., 3., 0.1)
        xx, yy = np.meshgrid(x, y)
        states = np.c_[xx.ravel(), yy.ravel()]
        counts = self.counting_space.get_counts(states)
        assert counts.shape == (900,)

        plt.contour(xx, yy, counts.reshape(xx.shape))
        plt.colorbar()
        plt.show()

        plt.imshow(counts.reshape(xx.shape), interpolation="nearest")
        plt.xticks([])
        plt.yticks([])
        plt.colorbar()
        plt.show()

    def run_experiment(self):
        data = self.generate_data()
        self.counting_space.train(data)
        counts = self.counting_space.get_counts(data)
        print("Training counts: ", counts)
        self.generate_counts_plot()


if __name__ == "__main__":
    experiment = Experiment1(0.5, "raw")
    experiment.run_experiment()
