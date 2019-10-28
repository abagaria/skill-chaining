import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import glob
import numpy as np
import pdb

sns.set()
# sns.set_context("talk")

RMAX = 10.

def moving_average(a, n=10) :
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

def smoothen_data(scores, n=25):
    print(scores.shape)
    smoothened_cols = scores.shape[1] - n + 1
    smoothened_data = np.zeros((scores.shape[0], smoothened_cols))
    for i in range(scores.shape[0]):
        smoothened_data[i, :] = moving_average(scores[i, :], n=n)
    return smoothened_data

def plot_option_executions(experiment_executions, experiment_name):
    """
    exeperiment_executions is a list of dictionaries
    We will extract the number of times each option was executed as a list of lists
    """
    plt.figure()

    options = ["overall_goal_policy", "option_1", "option_2", "option_3"]
    per_option_lol = []

    for option in options:
        option_list = []
        for executions in experiment_executions:
            if option in executions.keys():
                option_list.append(executions[option])
        if len(option_list) > 0:
            per_option_lol.append(option_list)

    for i, option_lol in enumerate(per_option_lol):
        medians, top, bottom = get_plot_params(option_lol, variable_sized=True)
        plt.plot(moving_average(medians), linewidth=3, label=options[i])
        # plt.fill_between( range(len(moving_average(medians))), top, bottom, alpha=0.1 )

    plt.ylim((-1, 5))
    plt.title("Option executions")
    plt.ylabel("# executions")
    plt.xlabel("episode")
    plt.legend()
    plt.savefig("{}_option_executions.png".format(experiment_name))
    plt.close()

def plot_training_scores(training_scores, experiment_name):
    median_scores, top, bottom = get_plot_params(training_scores)

    plt.figure()
    plt.plot(median_scores, linewidth=4)
    plt.fill_between(range(len(median_scores)), top, bottom, alpha=0.2)
    # plt.ylim((5, 10))

    # plt.yscale("log")
    plt.title("Episodic Scores (Training)")
    plt.ylabel("Reward")
    plt.xlabel("Episode")
    plt.savefig("{}_training_scores.png".format(experiment_name))
    plt.close()

def plot_comparison_training_scores(training_scores_batch, experiment_names, experiment="sc"):
    plt.figure(figsize=(8, 5))
    for training_scores, experiment_name in zip(training_scores_batch, experiment_names):
        median_scores, top, bottom = get_plot_params(training_scores)
        plt.plot(moving_average(median_scores), alpha=0.9, label=experiment_name.split("/")[-1])
        plt.fill_between(range(len(median_scores)), top, bottom, alpha=0.2)
        # plt.ylim((5, 10))
    plt.title("Episodic Scores (Training)")
    plt.ylabel("Reward")
    plt.xlabel("Episode")
    plt.legend()
    plt.savefig("{}_training_scores_comparison.png".format(experiment))
    plt.close()

def plot_comparison_training_durations(training_durations_batch, experiment_names, experiment="sc"):
    plt.figure(figsize=(8, 5))
    for training_durations, experiment_name in zip(training_durations_batch, experiment_names):
        median_scores, top, bottom = get_plot_params(training_durations, clamp_top=False)
        plt.plot(median_scores, linewidth=4, label=experiment_name.replace("_hard_pinball", ""))
        plt.fill_between( range(len(median_scores)), top, bottom, alpha=0.1 )
    plt.yscale("log")
    plt.title("Episodic Durations (Training)")
    plt.ylabel("Durations (# steps)")
    plt.xlabel("Episode")
    plt.legend()
    plt.savefig("{}_training_durations_comparison.png".format(experiment))
    plt.close()

def plot_training_durations(training_durations, experiment_name):
    median_durations, top, bottom = get_plot_params(training_durations, clamp_top=False)

    plt.figure()
    plt.plot(median_durations, linewidth=4)
    plt.fill_between(range(len(median_durations)), top, bottom, alpha=0.2)
    # plt.ylim((-100, RMAX*2000 + 100))
    plt.yscale("log")
    plt.title("Episodic Durations (Training)")
    plt.ylabel("Duration (# steps)")
    plt.xlabel("Episode")
    plt.savefig("{}_training_durations.png".format(experiment_name))
    plt.close()

def get_plot_params(scores, clamp_top=True, variable_sized=False):
    if variable_sized:
        lengths = map(len, scores)
        trunc_length = min(lengths)
        truncated_scores = [score_list[:trunc_length] for score_list in scores]
        s = np.array(truncated_scores)
    else:
        s = np.array(scores)
    medians = np.median(s, axis=0)
    means = np.mean(s, axis=0)
    stds = np.std(s, axis=0)
    if clamp_top:
        top = np.max(s, axis=0)
        bot = means - stds
    else:
        top = means + stds
        bot = means - stds
    return medians, top, bot

def get_scores(experiment_name):
    # if "sc" in experiment_name:
    #     score_names = experiment_name + "/" + "sc_pretrained_False_training_scores_*.pkl".format(experiment_name)
    #     duration_names = experiment_name + "/" + "sc_pretrained_False_training_durations_*.pkl".format(experiment_name)
    # else:
    score_names = experiment_name + "/" + "*_training_scores.pkl"
    duration_names = experiment_name + "/" "*_training_durations.pkl"
    training_scores, training_durations = [], []
    for score_file in glob.glob(score_names):
        with open(score_file, "rb") as f:
            scores = pickle.load(f)
            training_scores.append(scores)
    for duration_file in glob.glob(duration_names):
        with open(duration_file, "rb") as f:
            durations = pickle.load(f)
            training_durations.append(durations)
    return training_scores, training_durations

def get_option_executions(experiment_name):
    names = experiment_name + "/" + "sc_pretrained_False_option_executions_*.pkl"
    executions = []
    for f in glob.glob(names):
        with open(f, "rb") as _f:
            executions.append(pickle.load(_f))
    return executions

def produce_comparison_plots():
    #meta_experiment_name = "four_rooms_ddqn_exploration_2"  # "/home/abagaria/skill-chaining/discrete_four_rooms_eps_decay", "/home/abagaria/skill-chaining/discrete_four_rooms_eps05_2gradientSteps", "/home/abagaria/skill-chaining/discrete_four_rooms_rmax"
    #experiment_names = ["/home/abagaria/skill-chaining/discrete_four_rooms_eps05", "/home/abagaria/skill-chaining/discrete_four_rooms_rmax_iid_sa_toy", "/home/abagaria/skill-chaining/discrete_four_rooms_eps05_bs64"]
    # , "/home/abagaria/skill-chaining/optimistic_init_mcar_ocsvm_nu_01"
    meta_experiment_name = "mountain_car_optimistic_initialization_ocsvm"
    experiment_names = ["/home/abagaria/skill-chaining/optimistic_init_mcar_toy", "/home/abagaria/skill-chaining/mcar_dqn_eps2", "/home/abagaria/skill-chaining/optimistic_init_mcar_ocsvm_nu_01_buffer_001_noRounding"]
    training_scores_batch = []
    training_durations_batch = []
    for experiment_name in experiment_names:
        print("Generating plot for experiment {}".format(experiment_name))
        scores, durations = get_scores(experiment_name)
        training_scores_batch.append(scores)
        training_durations_batch.append(durations)
        # plot_option_executions(get_option_executions(experiment_name), experiment_name)
    plot_comparison_training_scores(training_scores_batch, experiment_names, experiment=meta_experiment_name)
    # plot_comparison_training_durations(training_durations_batch, experiment_names, experiment=meta_experiment_name)

def produce_experiment_plots():
    experiment_name = "acrobot_sc_1"
    training_scores, training_durations = get_scores(experiment_name)
    smoothened_data = smoothen_data(training_scores, 100)
    plot_training_scores(smoothened_data, experiment_name)
    # plot_training_durations(training_durations, experiment_name)

if __name__ == "__main__":
    produce_comparison_plots()
    # produce_experiment_plots()

