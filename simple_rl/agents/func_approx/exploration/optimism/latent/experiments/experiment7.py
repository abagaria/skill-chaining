# Python imports.
import pdb
import torch
import numpy as np
import matplotlib.pyplot as plt

# Other imports.
from simple_rl.agents.func_approx.exploration.optimism.latent.CountingLatentSpaceClass import CountingLatentSpace
from simple_rl.agents.func_approx.exploration.optimism.latent.utils import compute_gradient_norm

class Experiment7:
    """ When fitting the exploration bonus directly, verify that the gradient is higher for rare states than popular ones. """

    def __init__(self, epsilon):
        self.counting_space = CountingLatentSpace(2, epsilon, phi_type="function", experiment_name="exp7",
                                                  lam=100., optimization_quantity="bonus")

    @staticmethod
    def get_unique_action_data():
        v1 = np.array([0.1, 0.1])
        v2 = np.array([0.5, 0.9])
        v3 = np.array([0.9, 0.1])
        return v1, v2, v3

    @staticmethod
    def generate_action_data():
        v1, v2, v3 = Experiment7.get_unique_action_data()
        return np.vstack((v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v2, v2, v2, v2, v2, v2, v3, v3))

    @staticmethod
    def generate_support_data(method="grid"):
        if method == "random":
            return np.random.uniform(0, 1, 2000).reshape(-1, 2)

        xx, yy = np.meshgrid(np.arange(0., 1., 0.1),
                             np.arange(0., 1., 0.1))

        return np.c_[xx.ravel(), yy.ravel()]

    def generate_counts_plot(self):
        x = np.arange(0., 1., 0.05)
        y = np.arange(0., 1., 0.05)
        xx, yy = np.meshgrid(x, y)
        states = np.c_[xx.ravel(), yy.ravel()]
        counts = self.counting_space.get_counts(states, 0)

        plt.contour(xx, yy, counts.reshape(xx.shape))
        plt.colorbar()
        plt.show()

        plt.imshow(counts.reshape(xx.shape), interpolation="nearest")
        plt.xticks([])
        plt.yticks([])
        plt.colorbar()
        plt.show()

    def get_gradient_for_state(self, state):
        action_data = self.generate_action_data()
        support_data = self.generate_support_data()

        self.counting_space.model.zero_grad()
        phi_s = self.counting_space.extract_features(state)
        phi_s = torch.from_numpy(phi_s).float().unsqueeze(0).to(self.counting_space.device)
        bonus_loss = self.counting_space._bonus_loss(phi_s, [action_data, support_data])

        bonus_loss.backward()
        gradient = compute_gradient_norm(self.counting_space.model)
        return gradient, bonus_loss

    def run_experiment(self, epochs=5000):
        action_data = self.generate_action_data()
        support_data = self.generate_support_data()

        v1_gradients, v2_gradients, v3_gradients = [], [], []
        v1_losses, v2_losses, v3_losses = [], [], []

        for _ in range(epochs):
            self.counting_space.train(action_buffers=[action_data, support_data], epochs=1)
            v1, v2, v3 = self.get_unique_action_data()

            grad_v1, loss_v1 = self.get_gradient_for_state(v1)
            grad_v2, loss_v2 = self.get_gradient_for_state(v2)
            grad_v3, loss_v3 = self.get_gradient_for_state(v3)

            v1_gradients.append(grad_v1)
            v2_gradients.append(grad_v2)
            v3_gradients.append(grad_v3)

            v1_losses.append(loss_v1)
            v2_losses.append(loss_v2)
            v3_losses.append(loss_v3)

        self.generate_counts_plot()

        plt.subplot(1, 2, 1)
        plt.plot(v1_gradients, label="grad_v1")
        plt.plot(v2_gradients, label="grad_v2")
        plt.plot(v3_gradients, label="grad_v3")
        plt.xlabel("Training Epoch")
        plt.ylabel("Gradient")
        plt.legend()
        plt.grid()
        plt.subplot(1, 2, 2)
        plt.plot(v1_losses, label="loss_v1")
        plt.plot(v2_losses, label="loss_v2")
        plt.plot(v3_losses, label="loss_v3")
        plt.xlabel("Training Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.grid()
        plt.show()

if __name__ == '__main__':
    exp = Experiment7(0.1)

    exp.run_experiment()
