# Python imports.
import numpy as np
import matplotlib.pyplot as plt

# Other imports.
from simple_rl.agents.func_approx.exploration.optimism.latent.CountingLatentSpaceClass import CountingLatentSpace


class Experiment2:
    def __init__(self, epsilon):
        self.counting_space = CountingLatentSpace(2, epsilon, phi_type="function", experiment_name="exp2")


    @staticmethod
    def generate_action_data():
        v1 = np.array([0.1, 0.1])
        v2 = np.array([0.5, 0.9])
        v3 = np.array([0.9, 0.1])

        return np.vstack((v1, v1, v1, v1, v1, v2, v2, v2, v3, v3))

    @staticmethod
    def generate_support_data(method="grid"):
        if method == "random":
            return np.random.uniform(0, 1, 2000).reshape(-1, 2)

        xx, yy = np.meshgrid(np.arange(0., 1., 0.05),
                             np.arange(0., 1., 0.05))

        return np.c_[xx.ravel(), yy.ravel()]

    def generate_counts_plot(self):
        x = np.arange(0., 1., 0.05)
        y = np.arange(0., 1., 0.05)
        xx, yy = np.meshgrid(x, y)
        states = np.c_[xx.ravel(), yy.ravel()]
        counts = self.counting_space.get_counts(states)

        plt.contour(xx, yy, counts.reshape(xx.shape))
        plt.colorbar()
        plt.show()

        plt.imshow(counts.reshape(xx.shape), interpolation="nearest")
        plt.xticks([])
        plt.yticks([])
        plt.colorbar()
        plt.show()

    def run_experiment(self):
        action_data = self.generate_action_data()
        support_data = self.generate_support_data()

        self.counting_space.train(action_buffer=action_data, full_buffer=support_data, epochs=100)
        counts = self.counting_space.get_counts(support_data)

        self.generate_counts_plot()

        import pdb
        pdb.set_trace()
        print('heelo')

if __name__ == '__main__':
    exp = Experiment2(0.5)

    exp.run_experiment()
