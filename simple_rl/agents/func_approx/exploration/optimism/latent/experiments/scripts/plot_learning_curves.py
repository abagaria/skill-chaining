import numpy as np
import pickle
import glob
import os
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()


def get_scores(log_dir):
    overall_scores = []
    glob_pattern = log_dir + "/scores/" + "*.pkl"
    for score_file in glob.glob(glob_pattern):
        with open(score_file, "rb") as f:
            scores = pickle.load(f)
        overall_scores.append(scores)
    score_array = np.array(overall_scores)
    return score_array


def get_plot_params(array):
    median = np.median(array, axis=0)
    means = np.mean(array, axis=0)
    std = np.std(array, axis=0)
    N = array.shape[0]
    top = means + (std / np.sqrt(N))
    bot = means - (std / np.sqrt(N))
    return median, means, top, bot


def moving_average(a, n=25):
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n-1:] / n


def smoothen_data(scores, n=10):
    print(scores.shape)
    smoothened_cols = scores.shape[1] - n + 1
    smoothened_data = np.zeros((scores.shape[0], smoothened_cols))
    for i in range(scores.shape[0]):
        smoothened_data[i, :] = moving_average(scores[i, :], n=n)
    return smoothened_data


def generate_plot(score_array, label, smoothen=False):
    if smoothen:
        score_array = smoothen_data(score_array)
    median, mean, top, bottom = get_plot_params(score_array)
    plt.plot(mean, linewidth=2, label=label, alpha=0.9)
    plt.fill_between( range(len(top)), top, bottom, alpha=0.2 )


def main():
    experiment_name = "/home/abagaria/git-repos/skill-chaining/exp12"
    run_titles = ["DQN-Sqrt-Novelty", "DQN-InverseN-Novelty", "DQN-Eps0", "DQN-Eps05", "DQN-Action-Sel-Sqrt-Novelty"]

    log_dirs = [os.path.join(experiment_name, run_title) for run_title in run_titles]
    score_arrays = [get_scores(log_dir) for log_dir in log_dirs]

    [generate_plot(score_array, run_title, smoothen=True) for score_array, run_title in zip(score_arrays, run_titles)]
    plt.ylabel("Durations")
    plt.xlabel("Episode")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()
