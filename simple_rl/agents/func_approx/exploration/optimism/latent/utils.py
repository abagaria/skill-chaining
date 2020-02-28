import os
import pickle
import numpy as np


def get_mean_std(s, flat_std=False):
    mean = np.mean(s, axis=0)

    # Subtracting the mean ensures that the final
    # dataset has standard deviation of one
    if flat_std:
        std = np.std(s - mean)
    else:
        std = np.std(s, axis=0)
    return mean, std


def normalize(s, mean, std):
    return (s - mean) / (std + 1e-6)


def get_lam_for_buffer_size(buffer_size, c1=-1.726, c2=105690.):
    """
    log (vol_pp) = a * log(n) + b * log(lam) + c is the linear regression,
    the coefficients correspond to the simplified relation:
        lam = c2 * (n ^ c1)
    Please refer to experiments/scripts/grid/experiment13 to see how to get
    these coefficients.
    Args:
        buffer_size (int)
        c1 (float): Slope of the linear regression fit
        c2 (float): Intercept of the linear regression fit

    Returns:
        lambda_scaling (float)
    """
    bug_scaling = 31.6  # TODO: Recollect regression fit data with bug fixes
    return (buffer_size ** c1) * (c2 * bug_scaling)


def compute_gradient_norm(model):
    total_norm = 0.
    for p in model.parameters():
        param_norm = p.grad.data.norm(2)
        total_norm += param_norm.item() ** 2
    total_norm = total_norm ** (1. / 2)
    return total_norm


def create_log_dir(experiment_name):
    path = os.path.join(os.getcwd(), experiment_name)
    try:
        os.mkdir(path)
    except OSError:
        print("Creation of the directory %s failed" % path)
    else:
        print("Successfully created the directory %s " % path)
    return path


def moving_average(a, n=25):
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n


def save_scores(scores, experiment_name, seed, run_title=None):
    if run_title:
        filename = f"{experiment_name}/{run_title}/scores/seed_{seed}_scores.pkl"
    else:
        filename = f"{experiment_name}/{experiment_name}_seed_{seed}_scores.pkl"
    with open(filename, "wb+") as f:
        pickle.dump(scores, f)
