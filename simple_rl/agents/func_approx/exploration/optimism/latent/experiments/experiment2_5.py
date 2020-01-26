# Python imports.
import torch
import numpy as np
import matplotlib.pyplot as plt

# Other imports.
from simple_rl.agents.func_approx.exploration.optimism.latent.CountingLatentSpaceClass import CountingLatentSpace


class Experiment2_5:
    def __init__(self, epsilon, seed=0, method="random"):
        self.method = method
        np.random.seed(seed)
        self.counting_space = CountingLatentSpace(2, epsilon, phi_type="function", experiment_name="exp2_5")


    @staticmethod
    def generate_action_1_data(method="random"):
        if method == "random":
            return np.random.uniform(0, 1, size=(20, 2))

        xx, yy = np.meshgrid(np.arange(0., 0.9, 0.2),
                             np.arange(0., 0.9, 0.2))

        grid_1 =  np.c_[xx.ravel(), yy.ravel()]
        grid_2 = grid_1 + 0.1

        all_points = np.vstack((grid_1, grid_2))
        return all_points

    @staticmethod
    def generate_action_2_data(method="random"):
        if method == "random":
            return np.random.uniform(0, 1, size=(20, 2))

        xx, yy = np.meshgrid(np.arange(0., 0.9, 0.2),
                             np.arange(0., 0.9, 0.2))

        grid_1 = np.c_[xx.ravel(), yy.ravel()]
        grid_2 = grid_1 + 0.1

        grid_1[:,0] += 0.1
        grid_2[:,0] -= 0.1

        all_points = np.vstack((grid_1, grid_2))
        return all_points

    @staticmethod
    def visualize_latent_space(action_1_data, action_2_data, action_1_repr, action_2_repr):
        plt.subplot(1, 2, 1)
        plt.scatter(action_1_data[:, 0], action_1_data[:, 1], label="a1 Data")
        plt.scatter(action_2_data[:, 0], action_2_data[:, 1], label="a2 Data")
        plt.legend()
        plt.title("Input Space")

        plt.subplot(1, 2, 2)
        plt.scatter(action_1_repr[:, 0], action_1_repr[:, 1], label="a1 Repr")
        plt.scatter(action_2_repr[:, 0], action_2_repr[:, 1], label="a2 Repr")
        plt.legend()
        plt.title("Latent Space")
        plt.show()

    def run_experiment(self):
        action_1_data = self.generate_action_1_data(method=self.method)
        action_2_data = self.generate_action_2_data(method=self.method)

        self.counting_space.train(action_buffers=[action_1_data, action_2_data], epochs=100)

        # Inference on action and support data
        action_1_tensor = torch.from_numpy(action_1_data).float().to(self.counting_space.device)
        action_2_tensor = torch.from_numpy(action_2_data).float().to(self.counting_space.device)

        self.counting_space.model.eval()
        with torch.no_grad():
            action_1_repr = self.counting_space.model(action_1_tensor)
            action_2_repr = self.counting_space.model(action_2_tensor)
        self.counting_space.model.train()

        action_1_repr = action_1_repr.detach().cpu().numpy()
        action_2_repr = action_2_repr.detach().cpu().numpy()

        self.visualize_latent_space(action_1_data, action_2_data, action_1_repr, action_2_repr)

if __name__ == '__main__':
    exp = Experiment2_5(0.2, method="mesh")

    exp.run_experiment()
