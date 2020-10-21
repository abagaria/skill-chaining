import torch
import random
import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("white")
from tqdm import tqdm
from simple_rl.agents.func_approx.dsc.dynamics.option_dynamics_model import OptionDynamicsModel


class TestSkillDynamicsModel(object):
    def __init__(self, experiment_name, option_name, path_to_option_data, device):
        self.experiment_name = experiment_name
        self.option_name = option_name
        self.path_to_option_data = path_to_option_data
        self.s, self.sp = self.load_option_data()
        self.dynamics = OptionDynamicsModel(in_shape=self.s.shape[1],
                                            out_shape=self.sp.shape[1],
                                            num_mixtures=3,
                                            device=device)
        self.training_data, self.testing_data = self.get_train_test_splits()

    def load_option_data(self):
        with open(self.path_to_option_data, "rb") as f:
            s, sp = pickle.load(f)
        if isinstance(s, list):
            s = np.array(s)
        if isinstance(sp, list):
            sp = np.array(sp)
        return s, sp

    def get_train_test_splits(self):
        num_training_points = int(self.s.shape[0] * 0.75)
        training_idx = random.sample(range(self.s.shape[0]), k=num_training_points)
        testing_idx = list(set(range(self.s.shape[0])) - set(training_idx))

        training_data = self.s[training_idx, :], self.sp[training_idx, :]
        testing_data = self.s[testing_idx, :], self.sp[testing_idx, :]

        return training_data, testing_data

    def plot_losses(self, training_losses, testing_losses):
        print("Plotting losses...")
        plt.figure()
        plt.subplot(1, 2, 1)
        plt.plot(training_losses, marker="o")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Training Loss")
        plt.subplot(1, 2, 2)
        plt.plot(testing_losses, marker="o")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Test Loss")
        plt.savefig(f"{self.experiment_name}/{self.option_name}-losses.png")
        plt.close()

    def plot_predictions(self, states, next_states):
        print("Plotting predictions...")
        predictions = self.dynamics.predict(states)
        input_positions = states[:, :2]
        output_positions = predictions[:, :2]
        label_positions = next_states[:, :2]
        plt.figure()
        plt.subplot(1, 2, 1)
        plt.scatter(input_positions[:, 0], input_positions[:, 1], color="black", label="s")
        plt.scatter(output_positions[:, 0], output_positions[:, 1], color="red", label="s'")
        plt.title("Predictions"); plt.legend()
        plt.subplot(1, 2, 2)
        plt.scatter(input_positions[:, 0], input_positions[:, 1], color="black", label="s")
        plt.scatter(label_positions[:, 0], label_positions[:, 1], color="red", label="s'")
        plt.title("Labels"); plt.legend()
        plt.savefig(f"{self.experiment_name}/{self.option_name}-predictions.png")
        plt.close()

    def main(self):
        s_train, sp_train = self.training_data
        s_test, sp_test = self.testing_data
        training_errors, testing_errors = [], []

        for epoch in tqdm(range(500), desc="Fitting option-model"):
            training_loss = self.dynamics.train(s_train, sp_train)
            testing_error = self.dynamics.evaluate(s_test, sp_test, dim=2)

            training_errors.append(training_loss)
            testing_errors.append(testing_error)

        self.plot_losses(training_errors, testing_errors)
        self.plot_predictions(self.s, self.sp)


if __name__ == "__main__":
    option_name = "option-2"
    exp_name = "online-dsc-toy-2"
    pkl_file = f"{exp_name}/{option_name}-dynamics-data.pkl"
    device = torch.device("cuda:0")
    exp = TestSkillDynamicsModel(experiment_name=exp_name, path_to_option_data=pkl_file,
                                 option_name=option_name, device=device)
    exp.main()
