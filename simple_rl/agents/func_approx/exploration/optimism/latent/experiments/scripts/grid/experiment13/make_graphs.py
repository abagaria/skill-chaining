import numpy as np
import pickle
import glob
import os
import matplotlib.pyplot as plt
import seaborn as sns
import re
from collections import defaultdict
from pathlib import Path
from sklearn.linear_model import LinearRegression
import math
import ipdb

sns.set()

possible_colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple',
					   'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan']

def get_scores(log_dir):
    overall_scores = []
    glob_pattern = log_dir + "/scores/" + "*.pkl"
    for score_file in glob.glob(glob_pattern):
        with open(score_file, "rb") as f:
            scores = pickle.load(f)
        overall_scores.append(scores)
    score_array = np.array(overall_scores)
    return score_array

def get_data_points_for_run(subdir):
    """
    This gets the 6 recorded measures for a subdir.

    # run-title: seed-$seed-steps-$num_steps-lambda-$lambda-epochs-$epochs
    folder: subdir/latent_range_logs:
    file-names: we want them all! What they are is "everything before the second to last underscore." 
    """
    def extract_quantity_and_name(filename):
        # everything after slash and before second underscore.
        after_slash = filename.split("/")[-1]
        before_underscore = after_slash.split("_")[:-2]
        quantity_name = "_".join(before_underscore)
        with open(filename, "rb") as f:
            quantity_value = pickle.load(f)
        return quantity_name, quantity_value


    folder = os.path.join(subdir, "latent_range_logs")
    glob_pattern = os.path.join(folder, "*.pkl")
    values_dict = {}
    for score_file in glob.glob(glob_pattern):
        quantity_name, quantity_value = extract_quantity_and_name(score_file)
        values_dict[quantity_name] = quantity_value
    
    return values_dict


def get_all_runs_values(exp_dir_name):
    """
    Given the parent of the run-title dicts


    """
    def extract_hyperparameters_from_folder_name(folder_name):
        """
        input is something like seed-0-steps-500-lambda-0.01-epochs-1800
        Returns (lambda, num_steps)
        We'll use the power of ReGex
        """
        m = re.match(r'seed-(?P<seed>.*)-steps-(?P<steps>.*)-lambda-(?P<lambda>.*)-epochs-(?P<epochs>.*)', folder_name)
        return (float(m.group('lambda')), int(m.group('steps')))

    
    all_values_dict = {}

    exp_dir = Path(exp_dir_name)
    for child in exp_dir.iterdir():
        if child.is_dir():
            lam, num_steps = extract_hyperparameters_from_folder_name(child.name)
            values_dict = get_data_points_for_run(str(child))
            all_values_dict[(lam, num_steps)] = values_dict
    
    return all_values_dict


def get_value_vs_lambdas(all_values_dict, key):
    """ Get dictionary that maps num_steps to a list of (lam, val) tuples. """ 
    final_dict = defaultdict(list)
    for lam, steps in all_values_dict:
        spread_metrics = all_values_dict[(lam, steps)]
        if key in spread_metrics:
            val = spread_metrics[key]
            final_dict[steps].append((lam, val))
    return final_dict

def get_value_vs_steps(all_values_dict, key):
    """ Get dictionary that maps lam to a list of (steps, val) tuples. """ 
    final_dict = defaultdict(list)
    for lam, steps in all_values_dict:
        spread_metrics = all_values_dict[(lam, steps)]
        if key in spread_metrics:
            val = spread_metrics[key]
        final_dict[lam].append((steps, val))
    return final_dict

def plot_spread_vs_lam(exp_dir_name, key):
    all_values_dict = get_all_runs_values(exp_dir_name)
    val_vs_lam = get_value_vs_lambdas(all_values_dict, key)

    for steps in sorted(val_vs_lam.keys()):
        lams_and_ranges = list(sorted(val_vs_lam[steps]))
        lams = [x[0] for x in lams_and_ranges]
        x_ranges = [x[1] for x in lams_and_ranges]

        plt.plot(lams, x_ranges, "o-", label=f"{steps} steps")
    
    plt.xscale("log")
    if key == "vol_pp": 
        plt.yscale("log")
    plt.legend()
    plt.xlabel("Lambda")
    plt.ylabel(key)

def plot_spread_vs_num_steps(exp_dir_name, key):
    all_values_dict = get_all_runs_values(exp_dir_name)
    val_vs_steps = get_value_vs_steps(all_values_dict, key)

    for lam in sorted(val_vs_steps.keys()):
        steps_and_ranges = list(sorted(val_vs_steps[lam]))
        num_steps = [x[0] for x in steps_and_ranges]
        x_ranges = [x[1] for x in steps_and_ranges]

        plt.plot(num_steps, x_ranges, "o-", label=f"lambda={lam}")
    
    plt.legend()
    plt.xlabel("Num Steps")
    plt.ylabel(key)

def plot_all_spreads(exp_dir_name):
    keys = ["x_range", "y_range", "x_std", "y_std", "vol", "vol_pp"]
    
    for i, key in enumerate(keys):
        plt.subplot(3, 2, i + 1)
        plot_spread_vs_lam(exp_dir_name, key)
    
    plt.show()

    for i, key in enumerate(keys):
        plt.subplot(3, 2, i + 1)
        plot_spread_vs_num_steps(exp_dir_name, key)
    
    plt.show()



def get_n_lambda_volpp_relationship(exp_dir_name, do_filter=False):
    """
    How does this work? 

    First we get a bunch of (volpp, n, lambda) tuples.
    Then we transform to log everything.
    Then we fit linear volpp = a*log(n) + b * log(lambda)
    That gives us power for volpp = n^a + lam^b. Success!

    This will use some scipy nonsense, or maybe just linear algebra.

    """
    dataset = make_dataset(exp_dir_name) # [((vol_pp, n, lambda))]

    # print(dataset)

    print(len(dataset))

    if do_filter:
        dataset = [d for d in dataset if d[0] >=1e-5]

    # print(dataset)
    print(len(dataset))

    dataset = np.array(dataset)
    log_dataset = np.log(dataset) # base e
    log_vol_pp_arr, log_n_arr, log_lam_arr = log_dataset[:,0], log_dataset[:,1], log_dataset[:,2]
    
    # time for regression!
    X = np.concatenate((log_n_arr[..., None], log_lam_arr[..., None]), axis=1)
    y = log_vol_pp_arr

    assert len(X.shape) == 2
    assert X.shape[1] == 2

    model = LinearRegression(fit_intercept=True)
    model.fit(X, y)

    goodness = model.score(X, y)

    print(f"R2 = {goodness}")
    print(f"Coeff: {model.coef_} \t Intercept: {model.intercept_}")

    return X, y, model.coef_, model.intercept_

def get_optimal_parameters_for_volpp(log_n_coeff, log_lam_coeff, c_coeff, desired_volpp=3e-4):
    """
    We're given log(vpp) = log_n_coeff*log(n) + log_lam_coeff*log(lam) + c_coeff
    so, lam = e^(-log_n_coeff * log(n) / log_lam_coeff) * e^(-c_coeff / log_lam_coeff) * e^(ln(vpp)/log_lam_coeff)

    lam = n^(-log_n_coeff / log_lam_coeff) * e^(-c_coeff / log_lam_coeff) * vpp^(1/log_lam_coeff)
    """

    exp_1 = -log_n_coeff / log_lam_coeff
    exp_2 = -c_coeff / log_lam_coeff
    exp_3 = 1 / log_lam_coeff

    multiplicitive_coeff = (math.e ** (exp_2)) * desired_volpp ** exp_3

    return exp_1, multiplicitive_coeff



def visualize_fit(X, y, coeffs, intercept):
    log_n = X[:, 0]
    log_lam = X[:, 1]
    log_vol = y

    # Take anti-log
    num_steps = np.exp(log_n)

    unique_num_steps = list(map(np.ceil, set(num_steps.tolist())))

    steps_to_plot_idx = {}

    # Create one curve for each num_steps
    for i, steps in enumerate(unique_num_steps):
        predicted_log_vols = (coeffs[0] * np.log(steps)) + (coeffs[1] * log_lam) + intercept
        
        steps_to_plot_idx[int(steps)] = i + 1
        plt.subplot(3, 2, i + 1)
        plt.plot(np.exp(log_lam), np.exp(predicted_log_vols), "o",
                    c=possible_colors[i], label=f"Steps={steps}")
        

    # Plot the training data
    num_steps = num_steps.tolist()
    lams = np.exp(log_lam).tolist()
    vols = np.exp(log_vol).tolist()

    for steps, lam, vol in zip(num_steps, lams, vols):
        key = int(np.ceil(steps))
        plot_idx = steps_to_plot_idx[key]
        plt.subplot(3, 2, plot_idx)
        plt.plot(lam, vol, "+", c=possible_colors[plot_idx])

        plt.xscale("log")
        plt.xlabel("lambda")
        plt.ylabel("Predicted volume/point")
        plt.legend()
        plt.yscale("log")

    for steps in unique_num_steps:
        key = int(np.ceil(steps))
        plt.subplot(3, 2, steps_to_plot_idx[key])
        plt.axhline(y=3e-4, color='k', linestyle='--')

    plt.show()

def make_dataset(exp_dir_name):
    """
    We're going to get things of the form (vol_pp, n, lambda).
    Then, later we transform into log(vol_pp) = a * 
    """
    all_values_dict = get_all_runs_values(exp_dir_name)
    val_vs_lam = get_value_vs_lambdas(all_values_dict, "vol_pp") # step : [(lam, vol_pp)]
    all_thruples = []
    for step in val_vs_lam:
        data_arr =  val_vs_lam[step]
        for lam, vol_pp in data_arr:
            all_thruples.append((vol_pp, step, lam))
    return all_thruples



def main():
    # experiment_name = "/home/sam/Code/ML/optimism/skill-chaining/writes/exp13-mcar-grid"
    experiment_name = "/home/sam/Code/ML/optimism/skill-chaining/writes/exp13-mcar-grid-scaling-bugfix"

    # plot_all_spreads(experiment_name)
    X, y, coeffs, intercept = get_n_lambda_volpp_relationship(experiment_name, do_filter=True)

    assert len(coeffs) == 2

    optimal_parameters = get_optimal_parameters_for_volpp(coeffs[0], coeffs[1], intercept, desired_volpp=3e-4)

    print(f"Optimal Parameters: {optimal_parameters}")

    result_for_3000 = 3000**optimal_parameters[0] * optimal_parameters[1]

    print(f"Example: lambda for vol_pp 3e-4, n_steps 3000: {result_for_3000}")

    visualize_fit(X, y, coeffs, intercept)


if __name__ == "__main__":
    main()
