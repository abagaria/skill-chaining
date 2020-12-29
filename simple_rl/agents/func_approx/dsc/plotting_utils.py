import numpy as np
import pickle
import glob
import os
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()


def get_scores(log_dir):
    overall_scores = []
    glob_pattern = log_dir + "/" + "*.pkl"
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

def gp(score_array, label, smoothen=False):
    if smoothen:
        score_array = smoothen_data(score_array)
    plt.plot(range(len(score_array)), score_array, linewidth=2, label=label, alpha=0.9)
    plt.savefig("plot")

def get_successes(path):
    scores = []
    with open(path, "rb") as f:
        log_file = pickle.load(f)
    for episode in log_file:
        print(log_file[episode].keys())
        if "success" in log_file[episode].keys():
            scores.append(log_file[episode]["success"])
    return scores

def get_log_files(dir_path):
    fnames = os.path.join(dir_path, "log_file_*.pkl")
    return glob.glob(fnames)

def get_scores2(dir_path):
    files = get_log_files(dir_path)
    successes = [get_successes(_file) for _file in files]
    return successes

def truncate(scores):
    min_length = min([len(x) for x in scores])
    truncated_scores = [score[:min_length] for score in scores]
    return truncated_scores