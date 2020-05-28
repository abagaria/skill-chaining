# Python imports.
import pdb

import ipdb
import numpy as np
import scipy.interpolate
import matplotlib.pyplot as plt
import time
import torch
import imageio
import seaborn as sns
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


def _plot_initiation_sets(x_idx, y_idx, positive_examples, which_classifier, option, episode, experiment_name, negative_examples=None):
    axis_low = [-0.28, 0.3, 0.05, -0.4, 0.2]
    axis_high = [0.28, 0.9, 0.05, 0.4, 1.]
    axis_labels = ['endeff_x', 'endeff_y', 'endeff_z', 'puck_x', 'puck_z']
    plt.scatter(positive_examples[:, x_idx], positive_examples[:, y_idx], label="positive", c="blue", alpha=0.3)
    if negative_examples is not None:
        plt.scatter(negative_examples[:, x_idx], negative_examples[:, x_idx], label="negative", c="red", alpha=0.3)
        plt.legend()
    if axis_low is not None and axis_high is not None:
        plt.xlim(axis_low[x_idx], axis_high[x_idx])
        plt.xlim(axis_low[y_idx], axis_high[y_idx])
    if axis_labels is not None:
        plt.xlabel(axis_labels[x_idx])
        plt.ylabel(axis_labels[y_idx])
    if option.target_salient_event is not None:
        target = option.target_salient_event.target_state
        plt.scatter(target[x_idx], target[y_idx], c="black", marker="x", s=30)
    file_name = f"{option.name}_{axis_labels[x_idx]}_{axis_labels[y_idx]}_{episode}_{option.seed}_{which_classifier}"
    plt.title(f"{option.name} {which_classifier} Initiation Set")
    plt.savefig(f"initiation_set_plots/{experiment_name}/{file_name}.png")
    plt.close()


def plot_one_class_initiation_classifier(option, episode, experiment_name):
    positive_examples = option.construct_feature_matrix(option.positive_examples)
    _plot_initiation_sets(0, 1, positive_examples, "One Class", option, episode, experiment_name)
    _plot_initiation_sets(3, 4, positive_examples, "One Class", option, episode, experiment_name)


def plot_two_class_classifier(option, episode, experiment_name):
    positive_examples = option.construct_feature_matrix(option.positive_examples)
    negative_examples = option.construct_feature_matrix(option.negative_examples)
    _plot_initiation_sets(0, 1, positive_examples, "Two Class", option, episode, experiment_name, negative_examples)
    _plot_initiation_sets(3, 4, positive_examples, "Two Class", option, episode, experiment_name, negative_examples)


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


def visualize_dqn_shaped_rewards(dqn_agent, option, episode, seed, experiment_name):
    next_states = np.array([exp.next_state for exp in dqn_agent.replay_buffer.memory])
    if option.should_target_with_bonus():
        shaped_rewards = 1.0 * option.batched_is_init_true(next_states)
    else:
        shaped_rewards = np.zeros_like(next_states[:, 0])
    plt.scatter(next_states[:, 0], next_states[:, 1], c=shaped_rewards)
    plt.colorbar()
    file_name = f"{option.name}_high_level_shaped_rewards_seed_{seed}_episode_{episode}"
    plt.savefig(f"value_function_plots/{experiment_name}/{file_name}.png")
    plt.close()


def visualize_ddpg_shaped_rewards(global_option, other_option, episode, seed, experiment_name):
    ddpg_agent = global_option.solver
    next_states = np.array([exp[-2] for exp in ddpg_agent.replay_buffer.memory])
    if other_option.should_target_with_bonus():
        shaped_rewards = 1. * other_option.batched_is_init_true(next_states)
    else:
        shaped_rewards = np.zeros_like(next_states[:, 0])
    plt.scatter(next_states[:, 0], next_states[:, 1], c=shaped_rewards)
    plt.colorbar()
    file_name = f"{other_option.name}_low_level_shaped_rewards_seed_{seed}_episode_{episode}"
    plt.savefig(f"value_function_plots/{experiment_name}/{file_name}.png")
    plt.close()


def visualize_ddpg_replay_buffer(solver, episode, seed, experiment_name):
    states = np.array([exp[0] for exp in solver.replay_buffer.memory])
    actions = np.array([exp[1] for exp in solver.replay_buffer.memory])
    states_tensor = torch.from_numpy(states).float().to(solver.device)
    actions_tensor = torch.from_numpy(actions).float().to(solver.device)
    qvalues = solver.get_qvalues(states_tensor, actions_tensor).cpu().numpy().squeeze(1)
    plt.scatter(states[:, 0], states[:, 1], c=qvalues)
    plt.colorbar()
    file_name = f"{solver.name}_value_function_seed_{seed}_episode_{episode}"
    plt.savefig(f"value_function_plots/{experiment_name}/{file_name}.png")
    plt.close()


def visualize_best_option_to_take(policy_over_options_dqn, episode, seed, experiment_name):
    states = np.array([exp.state for exp in policy_over_options_dqn.replay_buffer.memory])
    states_tensor = torch.from_numpy(states).float().to(policy_over_options_dqn.device)
    options = policy_over_options_dqn.get_best_actions_batched(states_tensor).cpu().numpy()
    x = [s[0] for s in states]
    y = [s[1] for s in states]
    plt.scatter(x, y, c=options, cmap=plt.cm.Dark2)
    plt.colorbar()
    file_name = f"{policy_over_options_dqn.name}_best_options_seed_{seed}_episode_{episode}"
    plt.savefig(f"value_function_plots/{experiment_name}/{file_name}.png")
    plt.close()


def visualize_buffer(option, episode, seed, experiment_name):
    buffer = option.solver.replay_buffer.memory
    terminal_states = [transition[-2] for transition in buffer if transition[-1]]
    terminal_start_states = [transition[0] for transition in buffer if transition[-1]]
    non_terminal_states = [transition[0] for transition in buffer if transition[-1] == 0]
    plt.figure()
    plt.scatter([t[0] for t in non_terminal_states], [t[1] for t in non_terminal_states], alpha=0.1)
    plt.scatter([t[0] for t in terminal_states], [t[1] for t in terminal_states], alpha=1.0)
    plt.scatter([t[0] for t in terminal_start_states], [t[1] for t in terminal_start_states], alpha=1.0)
    plt.title(f"{option.solver.name}s replay buffer with length {len(buffer)}")
    file_name = f"{option.solver.name}_replay_buffer_seed_{seed}_episode_{episode}"
    plt.savefig(f"value_function_plots/{experiment_name}/{file_name}.png")
    plt.close()


def make_chunked_value_function_plot(solver, episode, seed, experiment_name, chunk_size=1000, replay_buffer=None):
    replay_buffer = replay_buffer if replay_buffer is not None else solver.replay_buffer
    states = np.array([exp[0] for exp in replay_buffer.memory])
    actions = np.array([exp[1] for exp in replay_buffer.memory])

    # Chunk up the inputs so as to conserve GPU memory
    num_chunks = int(np.ceil(states.shape[0] / chunk_size))
    state_chunks = np.array_split(states, num_chunks, axis=0)
    action_chunks = np.array_split(actions, num_chunks, axis=0)
    qvalues = np.zeros((states.shape[0],))
    current_idx = 0

    for chunk_number, (state_chunk, action_chunk) in tqdm(enumerate(zip(state_chunks, action_chunks)), desc="Making VF plot"):  # type: (int, np.ndarray)
        state_chunk = torch.from_numpy(state_chunk).float().to(solver.device)
        action_chunk = torch.from_numpy(action_chunk).float().to(solver.device)
        chunk_qvalues = solver.get_qvalues(state_chunk, action_chunk).cpu().numpy().squeeze(1)
        current_chunk_size = len(state_chunk)
        qvalues[current_idx:current_idx + current_chunk_size] = chunk_qvalues
        current_idx += current_chunk_size

    plt.scatter(states[:, 0], states[:, 1], c=qvalues)
    plt.colorbar()
    file_name = f"{solver.name}_value_function_seed_{seed}_episode_{episode}"
    plt.savefig(f"value_function_plots/{experiment_name}/{file_name}.png")
    plt.close()

    return qvalues.max()
