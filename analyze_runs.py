import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import glob
import numpy as np
import pdb
from matplotlib.ticker import ScalarFormatter

sns.set()
sns.set_context("talk")

RMAX = 10.

def moving_average(a, n=50) :
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

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
    plt.figure(figsize=(12, 6))
    for training_scores, experiment_name in zip(training_scores_batch, experiment_names):
        median_scores, top, bottom = get_plot_params(training_scores)
        plt.plot(median_scores, label=experiment_name.split("_")[-1])
        plt.fill_between(range(len(median_scores)), top, bottom, alpha=0.2)
        # plt.ylim((5, 10))
    plt.title("Episodic Scores (Training)")
    plt.ylabel("Reward")
    plt.xlabel("Episode")
    plt.legend()
    plt.savefig("{}_training_scores_comparison.png".format(experiment))
    plt.close()

def plot_comparison_training_scores_variance(training_scores_batch, experiment_names, experiment="sc"):
    cr_skills = []
    for i in range(len(training_scores_batch)):
        experiment_scores = training_scores_batch[i]  # n_seeds, n_episodes
        x = np.array(experiment_scores)
        cr_i_skills = np.sum(x, axis=1)
        cr_skills.append(cr_i_skills)
    cr_means = [cr.mean() / 1000 for cr in cr_skills]
    cr_stds  = [cr.std() / 1000 for cr in cr_skills]
    fig, ax = plt.subplots(1, 1)
    ax1 = sns.barplot(list(range(2, len(training_scores_batch) + 2)), cr_means, yerr=cr_stds)
    plt.xlabel("Number of skills")
    plt.ylabel("Mean cumulative reward (x1000)")
    plt.title("DSC on Point-Maze after 300 episodes")
    plt.show()

def plot_comparison_training_durations(training_durations_batch, experiment_names, experiment="sc"):
    # plt.figure(figsize=(8, 5))
    for training_durations, experiment_name in zip(training_durations_batch, experiment_names):
        median_scores, top, bottom = get_plot_params(training_durations, clamp_top=False)
        plt.plot(median_scores, label=experiment_name.split()[-1])
        plt.fill_between( range(len(median_scores)), top, bottom, alpha=0.2 )
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
        bot = np.min(s, axis=0)
    return medians, top, bot

def get_scores(experiment_name):    
    score_names = experiment_name + "/" + "*_training_scores_*.pkl"
    duration_names = experiment_name + "/" + "*_training_durations_*.pkl"
    # score_names = experiment_name + "/" + experiment_name + "*_training_scores.pkl"
    # duration_names = experiment_name + "/" + experiment_name + "*_training_durations.pkl"
    training_scores, training_durations = [], []
    for score_file in glob.glob(score_names):
        with open(score_file, "rb") as f:
            scores = pickle.load(f)
            training_scores.append(scores)
    for duration_file in glob.glob(duration_names):
        with open(duration_file, "rb") as f:
            durations = pickle.load(f)
            training_durations.append(durations)
    scores = [moving_average(score_list) for score_list in training_scores]
    durations = [moving_average(duration_list) for duration_list in training_durations]
    return scores, durations

def get_option_executions(experiment_name):
    names = experiment_name + "/" + "sc_pretrained_False_option_executions_*.pkl"
    executions = []
    for f in glob.glob(names):
        with open(f, "rb") as _f:
            executions.append(pickle.load(_f))
    return executions

def produce_comparison_plots():
    # meta_experiment_name = "performance_vs_num_skills"
    # experiment_names = ["pm_sparse_smdp1_bufferLen20_sg0_g6_onlyGoalOption", 
    #                     "pm_sparse_smdp1_bufferLen20_sg0_onlyGoalAndOption1",
    #                     "pm_sparse_smdp1_bufferLen20_sg0_onlyGoalOption1AndOption2",
    #                     "pm_sparse_smdp1_bufferLen20_sg0_g6_goalO1O2O3"]
    # experiment_names = ["pm_sparse_smdp1_bufferLen20_sg0_nSkills2",
    #                     "pm_sparse_smdp1_bufferLen20_sg0_g6_numSkills3",
    #                     "pm_sparse_smdp1_bufferLen20_sg0_g6_numSkills4"]
    meta_experiment_name = "FixedInitSetsNoDoorHack"
    experiment_names = ["sc_src_adjacentRooms_sg100g10i10", "sc_transfer_adjacentRooms_sg100g10i10"]
    #["fixed_option_idx_src_task", "fixed_option_idx_transfer_task"]
    training_scores_batch = []
    training_durations_batch = []
    for experiment_name in experiment_names:
        print("Generating plot for experiment {}".format(experiment_name))
        scores, durations = get_scores(experiment_name)
        training_scores_batch.append(scores)
        training_durations_batch.append(durations)
        # plot_option_executions(get_option_executions(experiment_name), experiment_name)
    plot_comparison_training_scores(training_scores_batch, experiment_names, experiment=meta_experiment_name)
    plot_comparison_training_durations(training_durations_batch, experiment_names, experiment=meta_experiment_name)
    # plot_comparison_training_scores_variance(training_scores_batch, experiment_names, experiment=meta_experiment_name)

def produce_experiment_plots():
    experiment_name = "acrobot_sc_1"
    training_scores, training_durations = get_scores(experiment_name)
    plot_training_scores(training_scores, experiment_name)
    # plot_training_durations(training_durations, experiment_name)

if __name__ == "__main__":
    produce_comparison_plots()
    # produce_experiment_plots()

