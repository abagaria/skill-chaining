# Python imports.
import math
import os
import ipdb
import numpy as np
import scipy.interpolate
import matplotlib.pyplot as plt
import time
import torch
import imageio
import seaborn as sns

from simple_rl.tasks.leap_wrapper.LeapWrapperStateClass import LeapWrapperState

sns.set()
from PIL import Image
from tqdm import tqdm

# Other imports.
from simple_rl.tasks.point_maze.PointMazeMDPClass import PointMazeMDP
from simple_rl.tasks.point_maze.PointMazeStateClass import PointMazeState
from simple_rl.tasks.point_reacher.PointReacherStateClass import PointReacherState


class Experience(object):
    def __init__(self, s, a, r, s_prime):
        self.state = s
        self.action = a
        self.reward = r
        self.next_state = s_prime

    def serialize(self):
        return self.state, self.action, self.reward, self.next_state

    def __str__(self):
        return "s: {}, a: {}, r: {}, s': {}".format(self.state, self.action, self.reward, self.next_state)

    def __repr__(self):
        return str(self)

    def __eq__(self, other):
        return isinstance(other, Experience) and self.__dict__ == other.__dict__

    def __ne__(self, other):
        return not self == other


# ---------------
# Plotting utils
# ---------------
def plot_trajectory(trajectory, color='k', marker="o"):
    for i, state in enumerate(trajectory):
        if isinstance(state, PointMazeState):
            x = state.position[0]
            y = state.position[1]
        else:
            x = state[0]
            y = state[1]

        if marker == "x":
            plt.scatter(x, y, c=color, s=100, marker=marker)
        else:
            plt.scatter(x, y, c=color, alpha=float(i) / len(trajectory), marker=marker)


def plot_all_trajectories_in_initiation_data(initiation_data, marker="o"):
    possible_colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple',
                       'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan']
    for i, trajectory in enumerate(initiation_data):
        color_idx = i % len(possible_colors)
        plot_trajectory(trajectory, color=possible_colors[color_idx], marker=marker)


def get_grid_states():
    ss = []
    for x in np.arange(-11., 11., 1.):
        for y in np.arange(-11., 11., 0.5):
            s = PointReacherState(position=np.array([x, y]), velocity=np.array([0., 0.]),
                                  theta=0., theta_dot=0., done=False)
            ss.append(s)
    return ss


def get_values(solver, init_values=False):
    values = []
    for x in np.arange(0., 11., 0.5):
        for y in np.arange(0., 11., 0.5):
            v = []
            for vx in [-0.01, -0.1, 0., 0.01, 0.1]:
                for vy in [-0.01, -0.1, 0., 0.01, 0.1]:
                    for theta in [-90., -45., 0., 45., 90.]:
                        for theta_dot in [-1., 0., 1.]:
                            s = PointMazeState(position=np.array([x, y]), velocity=np.array([vx, vy]),
                                               theta=theta, theta_dot=theta_dot, done=False)

                            if not init_values:
                                v.append(solver.get_value(s.features()))
                            else:
                                v.append(solver.is_init_true(s))

            if not init_values:
                values.append(np.mean(v))
            else:
                values.append(np.sum(v))

    return values


def render_sampled_value_function(solver, episode=None, experiment_name=""):
    states = get_grid_states()
    values = get_values(solver)

    x = np.array([state.position[0] for state in states])
    y = np.array([state.position[1] for state in states])
    xi, yi = np.linspace(x.min(), x.max(), 1000), np.linspace(y.min(), y.max(), 1000)
    xx, yy = np.meshgrid(xi, yi)
    rbf = scipy.interpolate.Rbf(x, y, values, function="linear")
    zz = rbf(xx, yy)
    plt.imshow(zz, vmin=min(values), vmax=max(values), extent=[x.min(), x.max(), y.min(), y.max()], origin="lower")
    plt.colorbar()
    name = solver.name if episode is None else solver.name + "_{}_{}".format(experiment_name, episode)
    plt.savefig("value_function_plots/{}/{}_value_function.png".format(experiment_name, name))
    plt.close()


def make_meshgrid(x, y, h=.02):
    x_min, x_max = x.min() - 1, x.max() + 1
    y_min, y_max = y.min() - 1, y.max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    return xx, yy


def sampled_initiation_states(option, trajectories):
    s = 0.02
    box_low = np.amin(trajectories, 0)
    box_high = np.amax(trajectories, 0)
    mesh = np.meshgrid(*[np.arange(axis_min, axis_max, s) for axis_min, axis_max in zip(box_low, box_high)])
    states = np.transpose([mesh_dim.ravel() for mesh_dim in mesh])
    return np.array([state for state in states if option.is_init_true(state)])


def _plot_initiation_sets(indices, which_classifier, option, episode, logdir, two_class=False):
    print(f"Plotting sampled initiation sets of {option.name}")
    print(f"Plotting initiation set trajectories of {option.name}")

    # sawyer constants
    x_low, x_high, y_low, y_high = -0.4, 0.4, 0.2, 1.
    axis_labels = ['endeff_x', 'endeff_y', 'endeff_z', 'puck_x', 'puck_y']
    titles = ['Endeff', 'Puck']

    # trajectories and sampled meshgrid for refined initiation sets
    positive_examples = option.construct_feature_matrix(option.positive_examples)
    negative_examples = option.construct_feature_matrix(option.negative_examples)
    initiation_states = sampled_initiation_states(option, positive_examples)

    fig, axs = plt.subplots(2, 2, sharex='all', sharey='all')
    fig.set_size_inches(15, 13)
    fig.suptitle(f"{option.name} {which_classifier} Initiation Set")

    # doesn't matter which axis we set these for because sharey and sharex are true
    axs[0, 0].set_xlim(x_low, x_high)
    axs[0, 0].set_ylim(y_low, y_high)
    axs[0, 0].set_xticks(np.linspace(x_low, x_high, 9))
    axs[0, 0].set_yticks(np.linspace(y_low, y_high, 9))

    for i, (x_idx, y_idx) in enumerate(indices):
        # graphing constants
        trajectory_axis = axs[1, i]
        sampled_axis = axs[0, i]
        x_label, y_label = axis_labels[x_idx], axis_labels[y_idx]

        ipdb.set_trace()

        # plot sampled initiation set
        sampled_axis.hexbin(initiation_states[:, x_idx], initiation_states[:, y_idx], cmap=plt.cm.get_cmap("Blues"))

        # plot positive and negative trajectories
        trajectory_axis.scatter(positive_examples[:, x_idx], positive_examples[:, y_idx], label="positive", c="b", alpha=0.5, s=50)
        if two_class and negative_examples.shape[0] > 0:
            trajectory_axis.scatter(negative_examples[:, x_idx], negative_examples[:, y_idx], label="negative", c="r", alpha=0.5, s=50)

        # plot option's target state
        if option.target_salient_event is not None:
            target = option.target_salient_event.target_state
            trajectory_axis.scatter(target[x_idx], target[y_idx], label="target salient event", c="black", marker="x", s=100)
            sampled_axis.scatter(target[x_idx], target[y_idx], label="target salient event", c="black", marker="x", s=100)

        # set title and legend
        trajectory_axis.set_title(f"{titles[i]} Trajectories")
        sampled_axis.set_title(f"{titles[i]} Mesh")
        if i == 1:
            trajectory_axis.legend()

        # set axes
        trajectory_axis.set_xlabel(x_label)
        trajectory_axis.set_ylabel(y_label)
        sampled_axis.set_xlabel(x_label)
        sampled_axis.set_ylabel(y_label)

    # save plot as png
    file_name = f"{option.name}_episode_{episode}_{option.seed}_{which_classifier}.png"
    plt.savefig(os.path.join(logdir, "initiation_set_plots", file_name))


def plot_one_class_initiation_classifier(option, episode, logdir):
    _plot_initiation_sets([(0, 1), (3, 4)], "One Class", option, episode, logdir)


def plot_two_class_classifier(option, episode, logdir):
    _plot_initiation_sets([(0, 1), (3, 4)], "Two Class", option, episode, logdir)


def visualize_dqn_replay_buffer(solver, experiment_name=""):
    goal_transitions = list(filter(lambda e: e[2] >= 0 and e[4] == 1, solver.replay_buffer.memory))
    cliff_transitions = list(filter(lambda e: e[2] < 0 and e[4] == 1, solver.replay_buffer.memory))
    non_terminals = list(filter(lambda e: e[4] == 0, solver.replay_buffer.memory))

    goal_x = [e[3][0] for e in goal_transitions]
    goal_y = [e[3][1] for e in goal_transitions]
    cliff_x = [e[3][0] for e in cliff_transitions]
    cliff_y = [e[3][1] for e in cliff_transitions]
    non_term_x = [e[3][0] for e in non_terminals]
    non_term_y = [e[3][1] for e in non_terminals]

    # background_image = imread("pinball_domain.png")

    plt.figure()
    plt.scatter(cliff_x, cliff_y, alpha=0.67, label="cliff")
    plt.scatter(non_term_x, non_term_y, alpha=0.2, label="non_terminal")
    plt.scatter(goal_x, goal_y, alpha=0.67, label="goal")
    # plt.imshow(background_image, zorder=0, alpha=0.5, extent=[0., 1., 1., 0.])

    plt.legend()
    plt.title("# transitions = {}".format(len(solver.replay_buffer)))
    plt.savefig("value_function_plots/{}/{}_replay_buffer_analysis.png".format(experiment_name, solver.name))
    plt.close()


def visualize_smdp_updates(global_solver, experiment_name=""):
    smdp_transitions = list(filter(lambda e: isinstance(e.action, int) and e.action > 0, global_solver.replay_buffer.memory))
    negative_transitions = list(filter(lambda e: e.done == 0, smdp_transitions))
    terminal_transitions = list(filter(lambda e: e.done == 1, smdp_transitions))
    positive_start_x = [e.state[0] for e in terminal_transitions]
    positive_start_y = [e.state[1] for e in terminal_transitions]
    positive_end_x = [e.next_state[0] for e in terminal_transitions]
    positive_end_y = [e.next_state[1] for e in terminal_transitions]
    negative_start_x = [e.state[0] for e in negative_transitions]
    negative_start_y = [e.state[1] for e in negative_transitions]
    negative_end_x = [e.next_state[0] for e in negative_transitions]
    negative_end_y = [e.next_state[1] for e in negative_transitions]

    plt.figure(figsize=(8, 5))
    plt.scatter(positive_start_x, positive_start_y, alpha=0.6, label="+s")
    plt.scatter(negative_start_x, negative_start_y, alpha=0.6, label="-s")
    plt.scatter(positive_end_x, positive_end_y, alpha=0.6, label="+s'")
    plt.scatter(negative_end_x, negative_end_y, alpha=0.6, label="-s'")
    plt.legend()
    plt.title("# updates = {}".format(len(smdp_transitions)))
    plt.savefig("value_function_plots/{}/DQN_SMDP_Updates.png".format(experiment_name))
    plt.close()


def visualize_next_state_reward_heat_map(solver, episode=None, experiment_name=""):
    next_states = [experience[3] for experience in solver.replay_buffer.memory]
    rewards = [experience[2] for experience in solver.replay_buffer.memory]
    x = np.array([state[0] for state in next_states])
    y = np.array([state[1] for state in next_states])

    plt.scatter(x, y, None, c=rewards, cmap=plt.cm.coolwarm)
    plt.colorbar()
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Replay Buffer Reward Heat Map")
    plt.xlim((-3, 11))
    plt.ylim((-3, 11))

    name = solver.name if episode is None else solver.name + "_{}_{}".format(experiment_name, episode)
    plt.savefig("value_function_plots/{}/{}_replay_buffer_reward_map.png".format(experiment_name, name))
    plt.close()


def replay_trajectory(trajectory, dir_name):
    colors = ["0 0 0 1", "0 0 1 1", "0 1 0 1", "1 0 0 1", "1 1 0 1", "1 1 1 1", "1 0 1 1", ""]

    for i, (option, state) in enumerate(trajectory):
        color = colors[option % len(colors)]
        mdp = PointMazeMDP(seed=0, render=True, color_str=color)

        qpos = np.copy(state.features()[:3])
        qvel = np.copy(state.features()[3:])

        mdp.env.wrapped_env.set_state(qpos, qvel)
        mdp.env.render()
        img_array = mdp.env.render(mode="rgb_array")
        img = Image.fromarray(img_array)
        img.save("{}/frame{}.png".format(dir_name, i))


def visualize_dqn_shaped_rewards(dqn_agent, option, episode, seed, logdir):
    next_states = np.array([exp.next_state for exp in dqn_agent.replay_buffer.memory])
    if option.should_target_with_bonus():
        shaped_rewards = 1.0 * option.batched_is_init_true(next_states)
        plt.scatter(next_states[:, 0], next_states[:, 1], c=shaped_rewards)
        plt.colorbar()
        file_name = f"{option.name}_high_level_shaped_rewards_seed_{seed}_episode_{episode}.png"
        plt.savefig(os.path.join(logdir, "value_function_plots", file_name))
        plt.close()


def visualize_ddpg_shaped_rewards(global_option, other_option, episode, seed, logdir):
    ddpg_agent = global_option.solver
    next_states = np.array([exp[-2] for exp in ddpg_agent.replay_buffer.memory])
    if other_option.should_target_with_bonus():
        shaped_rewards = 1. * other_option.batched_is_init_true(next_states)
        plt.scatter(next_states[:, 0], next_states[:, 1], c=shaped_rewards)
        plt.colorbar()
        file_name = f"{other_option.name}_low_level_shaped_rewards_seed_{seed}_episode_{episode}.png"
        plt.savefig(os.path.join(logdir, "value_function_plots", file_name))
        plt.close()


def visualize_ddpg_replay_buffer(solver, episode, seed, logdir):
    states = np.array([exp[0] for exp in solver.replay_buffer.memory])
    actions = np.array([exp[1] for exp in solver.replay_buffer.memory])
    states_tensor = torch.from_numpy(states).float().to(solver.device)
    actions_tensor = torch.from_numpy(actions).float().to(solver.device)
    qvalues = solver.get_qvalues(states_tensor, actions_tensor).cpu().numpy().squeeze(1)
    plt.scatter(states[:, 0], states[:, 1], c=qvalues)
    plt.colorbar()
    file_name = f"{solver.name}_value_function_seed_{seed}_episode_{episode}.png"
    plt.savefig(os.path.join(logdir, "value_function_plots", file_name))
    plt.close()


def visualize_best_option_to_take(policy_over_options_dqn, episode, seed, logdir):
    states = np.array([exp.state for exp in policy_over_options_dqn.replay_buffer.memory])
    states_tensor = torch.from_numpy(states).float().to(policy_over_options_dqn.device)
    options = policy_over_options_dqn.get_best_actions_batched(states_tensor).cpu().numpy()
    x = [s[0] for s in states]
    y = [s[1] for s in states]
    plt.scatter(x, y, c=options, cmap=plt.cm.Dark2)
    plt.colorbar()
    file_name = f"{policy_over_options_dqn.name}_best_options_seed_{seed}_episode_{episode}.png"
    plt.savefig(os.path.join(logdir, "value_function_plots", file_name))
    plt.close()


def visualize_buffer(option, episode, seed, logdir):
    buffer = option.solver.replay_buffer.memory
    terminal_states = [transition[-2] for transition in buffer if transition[-1]]
    terminal_start_states = [transition[0] for transition in buffer if transition[-1]]
    non_terminal_states = [transition[0] for transition in buffer if transition[-1] == 0]
    plt.figure()
    plt.scatter([t[0] for t in non_terminal_states], [t[1] for t in non_terminal_states], alpha=0.1)
    plt.scatter([t[0] for t in terminal_states], [t[1] for t in terminal_states], alpha=1.0)
    plt.scatter([t[0] for t in terminal_start_states], [t[1] for t in terminal_start_states], alpha=1.0)
    plt.title(f"{option.solver.name}s replay buffer with length {len(buffer)}")
    file_name = f"{option.solver.name}_replay_buffer_seed_{seed}_episode_{episode}.png"
    plt.savefig(os.path.join(logdir, "value_function_plots", file_name))
    plt.close()


def make_chunked_value_function_plot(solver, episode, seed, logdir, chunk_size=1000, replay_buffer=None):
    replay_buffer = replay_buffer if replay_buffer is not None else solver.replay_buffer

    # Checks if option has hasn't been executed yet. Making state and action chunks errors if the array is empty.
    if len(replay_buffer.memory) > 0:
        states = np.array([exp[0] for exp in replay_buffer.memory])
        actions = np.array([exp[1] for exp in replay_buffer.memory])
        # Chunk up the inputs so as to conserve GPU memory
        num_chunks = int(np.ceil(states.shape[0] / chunk_size))
        state_chunks = np.array_split(states, num_chunks, axis=0)
        action_chunks = np.array_split(actions, num_chunks, axis=0)
        qvalues = np.zeros((states.shape[0],))
        current_idx = 0

        for chunk_number, (state_chunk, action_chunk) in tqdm(enumerate(zip(state_chunks, action_chunks)),
                                                              desc="Making VF plot"):  # type: (int, np.ndarray)
            state_chunk = torch.from_numpy(state_chunk).float().to(solver.device)
            action_chunk = torch.from_numpy(action_chunk).float().to(solver.device)
            chunk_qvalues = solver.get_qvalues(state_chunk, action_chunk).cpu().numpy().squeeze(1)
            current_chunk_size = len(state_chunk)
            qvalues[current_idx:current_idx + current_chunk_size] = chunk_qvalues
            current_idx += current_chunk_size

        plt.scatter(states[:, 0], states[:, 1], c=qvalues, alpha=0.4)
        plt.colorbar()
        file_name = f"{solver.name}_value_function_seed_{seed}_episode_{episode}.png"
        plt.savefig(os.path.join(logdir, "value_function_plots", file_name))
        plt.close()

        return np.max(qvalues)


def create_log_dirs(directory_path):
    def _create_log_dir(path_name):
        try:
            os.makedirs(path_name)
        except OSError:
            print("Creation of the directory %s failed" % os.path.join(os.getcwd(), path_name))
        else:
            print("Successfully created the directory %s " % os.path.join(os.getcwd(), path_name))

    # initiation sets of options
    _create_log_dir(os.path.join(directory_path, "initiation_set_plots"))
    # value function for policy over options (DQN), intra-option policies (DDPG), and reward shaping
    _create_log_dir(os.path.join(directory_path, "value_function_plots"))
    # actor, critic dictionaries for all options
    _create_log_dir(os.path.join(directory_path, "saved_model"))
    # training/validation scores, selected options, and training duration
    _create_log_dir(os.path.join(directory_path, "scores"))


def rotate_file_name(file_path):
    if os.path.isdir(file_path):
        suffix = 1
        file_path += "_" + str(suffix)
        while os.path.isdir(file_path):
            suffix += 1
            file_path = file_path.rsplit("_", 1)[0] + "_" + str(suffix)
        return file_path
    else:
        return file_path
