# Python imports.
import pdb
import random

import numpy as np
import scipy.interpolate
import matplotlib
import matplotlib.pyplot as plt
import matplotlib
import time
import torch
import imageio
import seaborn as sns
sns.set()
sns.set_style("white")
from PIL import Image
from tqdm import tqdm
import os
import ipdb
from mpl_toolkits.mplot3d import Axes3D
import math

# Other imports.
from simple_rl.mdp.StateClass import State
from simple_rl.agents.func_approx.dsc.SalientEventClass import SalientEvent
from simple_rl.agents.func_approx.dqn.DQNAgentClass import DQNAgent

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

def plot_trajectory(trajectory, experiment_name, episode, color='k'):
    pos = [s.features()[:2] if isinstance(s, State) else s[:2] for s in trajectory]
    x = [p[0] for p in pos]; y = [p[1] for p in pos]
    plt.scatter(x, y, c=color)
    plt.colorbar()
    fname = f"value_function_plots/{experiment_name}/rnd_rollout_episode_{episode}_{random.randint(0, 100)}.png"
    plt.savefig(fname)
    plt.close()

def plot_all_trajectories_in_initiation_data(initiation_data, marker="o"):
    possible_colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple',
                       'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan']
    for i, trajectory in enumerate(initiation_data):
        color_idx = i % len(possible_colors)
        plot_trajectory(trajectory, color=possible_colors[color_idx], marker=marker)


def get_grid_states(mdp):
    ss = []
    x_low_lim, y_low_lim = mdp.get_x_y_low_lims()
    x_high_lim, y_high_lim = mdp.get_x_y_high_lims()
    for x in np.arange(x_low_lim, x_high_lim, 1):
        for y in np.arange(y_low_lim, y_high_lim, 1):
            ss.append(np.array((x, y)))
    return ss


def get_initiation_set_values(option):
    values = []
    x_low_lim, y_low_lim = option.overall_mdp.get_x_y_low_lims()
    x_high_lim, y_high_lim = option.overall_mdp.get_x_y_high_lims()
    for x in np.arange(x_low_lim, x_high_lim, 1):
        for y in np.arange(y_low_lim, y_high_lim, 1):
            values.append(option.is_init_true(np.array((x, y))))

    return values

# def get_values(solver, init_values=False):
#     values = []
#     for x in np.arange(0., 11., 0.5):
#         for y in np.arange(0., 11., 0.5):
#             v = []
#             for vx in [-0.01, -0.1, 0., 0.01, 0.1]:
#                 for vy in [-0.01, -0.1, 0., 0.01, 0.1]:
#                     for theta in [-90., -45., 0., 45., 90.]:
#                         for theta_dot in [-1., 0., 1.]:
#                             s = PointMazeState(position=np.array([x, y]), velocity=np.array([vx, vy]),
#                                                theta=theta, theta_dot=theta_dot, done=False)
#
#                             if not init_values:
#                                 v.append(solver.get_value(s.features()))
#                             else:
#                                 v.append(solver.is_init_true(s))
#
#             if not init_values:
#                 values.append(np.mean(v))
#             else:
#                 values.append(np.sum(v))
#
#     return values
#
# def render_sampled_value_function(solver, episode=None, experiment_name=""):
#     states = get_grid_states()
#     values = get_values(solver)
#
#     x = np.array([state.position[0] for state in states])
#     y = np.array([state.position[1] for state in states])
#     xi, yi = np.linspace(x.min(), x.max(), 1000), np.linspace(y.min(), y.max(), 1000)
#     xx, yy = np.meshgrid(xi, yi)
#     rbf = scipy.interpolate.Rbf(x, y, values, function="linear")
#     zz = rbf(xx, yy)
#     plt.imshow(zz, vmin=min(values), vmax=max(values), extent=[x.min(), x.max(), y.min(), y.max()], origin="lower")
#     plt.colorbar()
#     name = solver.name if episode is None else solver.name + "_{}_{}".format(experiment_name, episode)
#     plt.savefig("value_function_plots/{}/{}_value_function.png".format(experiment_name, name))
#     plt.close()

def make_meshgrid(x, y, h=.02):
    x_min, x_max = x.min() - 1, x.max() + 1
    y_min, y_max = y.min() - 1, y.max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    return xx, yy

def plot_one_class_initiation_classifier(option, episode=None, experiment_name=""):

    colors = ["blue", "yellow", "green", "red", "cyan", "brown"]

    # plt.figure(figsize=(8.0, 5.0))
    X = option.construct_feature_matrix(option.positive_examples)
    X0, X1 = X[:, 0], X[:, 1]
    xx, yy = make_meshgrid(X0, X1)
    Z1 = option.pessimistic_classifier.decision_function(np.c_[xx.ravel(), yy.ravel()])
    Z1 = Z1.reshape(xx.shape)

    color = colors[option.option_idx % len(colors)]
    plt.contour(xx, yy, Z1, levels=[0], linewidths=2, colors=[color])

    # plot_all_trajectories_in_initiation_data(option.positive_examples)

    # background_image = imageio.imread("emaze_domain.png")
    # plt.imshow(background_image, zorder=0, alpha=0.5, extent=[-2.5, 10., -2.5, 18.])
    #
    # plt.xticks([])
    # plt.yticks([])

    # plt.xlim((-10, 10))
    # plt.ylim((-10, 10))

    # plt.title("Name: {}\tParent: {}".format(option.name, option.parent))
    # name = option.name if episode is None else option.name + "_{}_{}".format(experiment_name, episode)
    # plt.savefig("initiation_set_plots/{}/{}_{}_one_class_svm.png".format(experiment_name, name, option.seed))
    # plt.close()


def plot_two_class_classifier(option, episode, experiment_name, plot_examples=True):
    states = get_grid_states(option.overall_mdp)
    values = get_initiation_set_values(option)

    x = np.array([state[0] for state in states])
    y = np.array([state[1] for state in states])
    xi, yi = np.linspace(x.min(), x.max(), 1000), np.linspace(y.min(), y.max(), 1000)
    xx, yy = np.meshgrid(xi, yi)
    rbf = scipy.interpolate.Rbf(x, y, values, function="linear")
    zz = rbf(xx, yy)
    plt.imshow(zz, vmin=min(values), vmax=max(values), extent=[x.min(), x.max(), y.min(), y.max()], origin="lower", alpha=0.6, cmap=plt.cm.coolwarm)
    plt.colorbar()

    # Plot trajectories
    positive_examples = option.construct_feature_matrix(option.positive_examples)
    negative_examples = option.construct_feature_matrix(option.negative_examples)

    if positive_examples.shape[0] > 0 and plot_examples:
        plt.scatter(positive_examples[:, 0], positive_examples[:, 1], label="positive", c="black", alpha=0.3, s=10)

    if negative_examples.shape[0] > 0 and plot_examples:
        plt.scatter(negative_examples[:, 0], negative_examples[:, 1], label="negative", c="magenta", alpha=0.3, s=10)

    if option.pessimistic_classifier is not None:
        plot_one_class_initiation_classifier(option, episode, experiment_name)

    # background_image = imageio.imread("four_room_domain.png")
    # plt.imshow(background_image, zorder=0, alpha=0.5, extent=[-2.5, 10., -2.5, 10.])

    name = option.name if episode is None else option.name + "_{}_{}".format(experiment_name, episode)
    plt.title("{} Initiation Set".format(option.name))
    # plt.legend()
    plt.savefig("initiation_set_plots/{}/{}_initiation_classifier_{}.png".format(experiment_name, name, option.seed))
    plt.close()


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

def visualize_next_state_reward_heat_map(solver, overall_mdp, episode=None, experiment_name=""):
    # TODO: Generalize for model-free extrapolator as well
    next_states = solver.replay_buffer.obs2_buf
    rewards = solver.replay_buffer.rew_buf
    x = np.array([state[0] for state in next_states])
    y = np.array([state[1] for state in next_states])

    plt.scatter(x, y, None, c=rewards.squeeze())
    plt.colorbar()
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Replay Buffer Reward Heat Map")

    x_low_lim, y_low_lim = overall_mdp.get_x_y_low_lims()
    x_high_lim, y_high_lim = overall_mdp.get_x_y_high_lims()

    plt.xlim((x_low_lim, x_high_lim))
    plt.ylim((y_low_lim, y_high_lim))

    plt.savefig(f"value_function_plots/{experiment_name}/rnd_reward_map_episode_{episode}.png")
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
    next_states = np.array([exp[-2] for exp in ddpg_agent.replay_buffer])
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
    states = np.array([exp[0] for exp in replay_buffer])
    actions = np.array([exp[1] for exp in replay_buffer])

    # Chunk up the inputs so as to conserve GPU memory
    num_chunks = int(np.ceil(states.shape[0] / chunk_size))

    if num_chunks == 0:
        return 0.

    state_chunks = np.array_split(states, num_chunks, axis=0)
    action_chunks = np.array_split(actions, num_chunks, axis=0)
    qvalues = np.zeros((states.shape[0],))
    current_idx = 0

    for chunk_number, (state_chunk, action_chunk) in tqdm(enumerate(zip(state_chunks, action_chunks)), desc="Making VF plot"):  # type: (int, np.ndarray)
        state_chunk = torch.from_numpy(state_chunk).float().to(solver.device)
        action_chunk = torch.from_numpy(action_chunk).float().to(solver.device)
        if isinstance(solver, DQNAgent):
            chunk_qvalues = solver.get_batched_qvalues(state_chunk).cpu().numpy()
            chunk_qvalues = np.max(chunk_qvalues, axis=1)
        else:
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

def visualize_mpc_next_state_value_distribution(mpc, state, episode, seed, experiment_name):
    num_steps = 7

    def simulate(self, s, num_rollouts=14000):
        """ Perform N simulations of length H. """

        torch_actions = 2 * torch.rand((num_rollouts, num_steps, self.action_size), device=self.device) - 1
        features = s if isinstance(s, np.ndarray) else s.features()
        torch_states = torch.tensor(features, device=self.device).repeat(num_rollouts, 1)
        costs = torch.zeros((num_rollouts, num_steps))

        for j in range(num_steps):
            actions = torch_actions[:, j, :]
            torch_states = self.model.predict_next_state(torch_states.float(), actions.float())
            costs[:, j] = self._get_costs(torch_states)

        return torch_states.cpu().numpy(), costs.cpu().numpy()

    next_states, undiscounted_costs = simulate(mpc, state)
    gammas = np.power(mpc.gamma * np.ones(num_steps), np.arange(0, num_steps))
    cumulative_rewards = -np.sum(undiscounted_costs * gammas, axis=1)
    
    x = [sp[0] for sp in next_states]
    y = [sp[1] for sp in next_states]

    plt.scatter(x, y, c=cumulative_rewards)
    plt.colorbar()          
    plt.scatter(state[0], state[1], marker="x", s=200)
    plt.savefig(f"value_function_plots/{experiment_name}/mpc_next_state_dist_{seed}.png")
    plt.close()


def make_chunked_goal_conditioned_value_function_plot(solver, goal, episode, seed, experiment_name, chunk_size=1000, replay_buffer=None):
    replay_buffer = replay_buffer if replay_buffer is not None else solver.replay_buffer

    # Take out the original goal and append the new goal
    states = [exp[0] for exp in replay_buffer]
    states = [state[:-2] for state in states]
    states = np.array([np.concatenate((state, goal), axis=0) for state in states])

    actions = np.array([exp[1] for exp in replay_buffer])

    # Chunk up the inputs so as to conserve GPU memory
    num_chunks = int(np.ceil(states.shape[0] / chunk_size))

    if num_chunks == 0:
        return 0.

    state_chunks = np.array_split(states, num_chunks, axis=0)
    action_chunks = np.array_split(actions, num_chunks, axis=0)
    qvalues = np.zeros((states.shape[0],))
    current_idx = 0

    for chunk_number, (state_chunk, action_chunk) in tqdm(enumerate(zip(state_chunks, action_chunks)), desc="Making VF plot"):  # type: (int, np.ndarray)
        state_chunk = torch.from_numpy(state_chunk).float().to(solver.device)
        action_chunk = torch.from_numpy(action_chunk).float().to(solver.device)
        if isinstance(solver, DQNAgent):
            chunk_qvalues = solver.get_batched_qvalues(state_chunk).cpu().numpy()
            chunk_qvalues = np.max(chunk_qvalues, axis=1)
        else:
            chunk_qvalues = solver.get_qvalues(state_chunk, action_chunk).cpu().numpy().squeeze(1)
        current_chunk_size = len(state_chunk)
        qvalues[current_idx:current_idx + current_chunk_size] = chunk_qvalues
        current_idx += current_chunk_size

    plt.scatter(states[:, 0], states[:, 1], c=qvalues)
    plt.colorbar()
    file_name = f"{solver.name}_value_function_seed_{seed}_episode_{episode}"
    plt.title(f"VF Targeting {np.round(goal, 2)}")
    plt.savefig(f"value_function_plots/{experiment_name}/{file_name}.png")
    plt.close()

    return qvalues.max()


def visualize_goal_conditioned_trajectory(trajectory, goal, reward_func, experiment_name, seed):
    x = np.array([transition[-1][0] for transition in trajectory])
    y = np.array([transition[-1][1] for transition in trajectory])
    rewards = np.array([reward_func(transition[-1], goal, info={})[0] for transition in trajectory])
    plt.scatter(x, y, c=rewards)
    plt.colorbar()
    goal_label = np.round(goal, decimals=1)
    plt.savefig(f"value_function_plots/{experiment_name}/gc_trajectory_{goal_label}_seed_{seed}.png")
    plt.close()


def create_log_dir(experiment_name):
    path = os.path.join(os.getcwd(), experiment_name)
    try:
        os.mkdir(path)
    except OSError:
        print("Creation of the directory %s failed" % path)
    else:
        print("Successfully created the directory %s " % path)
    return path

def plot_values(salient_event, init_state, replay_buffer, experiment_name=""):
    option = salient_event.covering_option
    is_low = salient_event.is_low

    fig = plt.figure()
    ax3d = fig.add_subplot(1, 1, 1, projection='3d')

    states = np.array(replay_buffer.sample(min(4000, len(replay_buffer)), get_tensor=False)[0])

    init_state, = option.states_to_tensor([init_state])
    init_value, = option.initiation_classifier([init_state]).flatten()

    target_state = salient_event.target_state
    target_value, = option.initiation_classifier([target_state]).flatten()

    option_term = option.batched_is_term_true(states, is_low)
    option_init_states, option_term_states = states[~option_term], states[option_term]
    option_init_values = option.initiation_classifier(option_init_states).flatten()

    cmap = matplotlib.cm.get_cmap('Blues')
    normalize = matplotlib.colors.Normalize(vmin=min(option_init_values), vmax=max(option_init_values))
    colors = [cmap(normalize(v)) for v in option_init_values]

    ax3d.scatter(option_init_states[:, 0], option_init_states[:, 1], init_value - option_init_values, c=colors)

    # salient event initiation set is a subset of the covering option termination set
    event_init_states = np.array([s for s in option_term_states if salient_event.is_init_true(s)])
    event_init_values = option.initiation_classifier(event_init_states).flatten()
    event_not_init_states = np.array([s for s in option_term_states if not salient_event.is_init_true(s)])
    event_not_init_values = option.initiation_classifier(event_not_init_states).flatten()

    ax3d.scatter(event_not_init_states[:, 0], event_not_init_states[:, 1], init_value - event_not_init_values, c="orange")
    ax3d.scatter(event_init_states[:, 0], event_init_states[:, 1], init_value - event_init_values, c="green")

    ax3d.scatter([target_state[0]], [target_state[1]], [init_value - target_value], c="black", s=[400])

    low_bound_x, up_bound_x = -11, 11
    # low_bound_y, up_bound_y = -3, 11

    ax3d.set_xlim((low_bound_x, up_bound_x))
    # ax3d.set_ylim((low_bound_y, up_bound_y))

    ax3d.set_xlabel("x")
    ax3d.set_ylabel("y")

    name = option.name
    threshold = option.threshold
    beta = option.beta
    fig.suptitle(f"Covering Options with threshold {threshold}, buffer size {len(replay_buffer)},\n and beta {beta:.4f}")
    ax3d.set_title("f(s_0) - f(s) vs (x, y)-position")
    plt.savefig("initiation_set_plots/{}/{}-{}_threshold-is_low={}.png".format(experiment_name, name, threshold, is_low))
    plt.close()

def plot_effect_sets(options):
    plt.figure()
    for option in options:
        states = option.effect_set
        x = [state.position[0] for state in states]
        y = [state.position[1] for state in states]
        sns.kdeplot(x, y, shade=True)
    plt.show()

def visualize_graph_nodes_with_expansion_probabilities(planner, episode, experiment_name, seed, background_img_fname="ant_maze_big_domain"):
    def _get_candidate_nodes():
        return planner.get_candidate_nodes_for_expansion()

    def _get_node_probability(e, normalizing_factor):
        score = e.compute_intrinsic_reward_score(planner.exploration_agent)
        return score / normalizing_factor

    def _get_node_probability_normalization_factor(descendants):
        """ Sum up the selection scores of all the nodes in the graph. """
        scores = np.array([node.compute_intrinsic_reward_score(planner.exploration_agent) for node in descendants])
        return scores.sum()

    def _get_representative_point(node):
        if isinstance(node, SalientEvent):
            return _get_event_representative_point(node)
        return _get_option_representative_point(node)

    def _get_event_representative_point(event):
        assert isinstance(event, SalientEvent)
        if event.get_target_position() is not None:
            return event.get_target_position()
        trigger_positions = [event._get_position(s) for s in event.trigger_points]
        trigger_positions = np.array(trigger_positions)
        return trigger_positions.mean(axis=0)

    def _get_option_representative_point(option):
        effect_positions = np.array([planner.mdp.get_position(state) for state in option.effect_set])
        return np.median(effect_positions, axis=0)

    nodes = _get_candidate_nodes()
    norm = _get_node_probability_normalization_factor(nodes)

    points = [_get_representative_point(n) for n in nodes]
    x_coords = [point[0] for point in points]
    y_coords = [point[1] for point in points]
    probabilities = [_get_node_probability(n, norm) for n in nodes]

    plt.scatter(x_coords, y_coords, c=probabilities)
    plt.colorbar()
    plt.xticks([])
    plt.yticks([])

    x_low_lim, y_low_lim = planner.mdp.get_x_y_low_lims()
    x_high_lim, y_high_lim = planner.mdp.get_x_y_high_lims()

    filename = os.path.join(os.getcwd(), f"{background_img_fname}.png")
    if os.path.isfile(filename):
        background_image = imageio.imread(filename)
        plt.imshow(background_image, zorder=0, alpha=0.5, extent=[x_low_lim, x_high_lim, y_low_lim, y_high_lim])

    plt.xlim((x_low_lim, x_high_lim))
    plt.ylim((y_low_lim, y_high_lim))

    prefix = "event_node_prob_graph"
    plt.savefig(f"value_function_plots/{experiment_name}/{prefix}_episode_{episode}_seed_{seed}.png")
    plt.close()


def visualize_graph(planner, episode, experiment_name, seed, use_target_states=True, background_img_fname="ant_maze_big_domain"):
    def _get_representative_point(event):
        assert isinstance(event, SalientEvent)
        if event.get_target_position() is not None:
            return event.get_target_position()
        trigger_positions = [event._get_position(s) for s in event.trigger_points]
        trigger_positions = np.array(trigger_positions)
        return trigger_positions.mean(axis=0)

    def _plot_event_pair(e1, e2, marker="o-", edge_color="black"):
        x1, y1 = _get_representative_point(e1)
        x2, y2 = _get_representative_point(e2)
        x = [x1, x2]; y = [y1, y2]
        plt.plot(x, y, marker, c=edge_color, alpha=0.075)
        plt.scatter(x, y, c="black")

    def is_connected(n1, n2):
        if use_target_states:
            return planner.plan_graph.does_path_exist(_get_representative_point(n1), n2)
        return planner.plan_graph.does_path_exist_between_nodes(n1, n2)

    events = planner.mdp.all_salient_events_ever + [planner.mdp.get_start_state_salient_event()]

    for event1 in events:
        for event2 in events:
            if event1 != event2:
                if is_connected(event1, event2) and is_connected(event2, event1):
                    _plot_event_pair(event1, event2, "o-", edge_color="black")
                elif is_connected(event1, event2) or is_connected(event2, event1):
                    _plot_event_pair(event1, event2, "o--", edge_color="tab:gray")

    plt.xticks([])
    plt.yticks([])

    x_low_lim, y_low_lim = planner.mdp.get_x_y_low_lims()
    x_high_lim, y_high_lim = planner.mdp.get_x_y_high_lims()

    filename = os.path.join(os.getcwd(), f"{background_img_fname}.png")
    if os.path.isfile(filename):
        background_image = imageio.imread(filename)
        plt.imshow(background_image, zorder=0, alpha=0.5, extent=[x_low_lim, x_high_lim, y_low_lim, y_high_lim])

    plt.xlim((x_low_lim, x_high_lim))
    plt.ylim((y_low_lim, y_high_lim))

    prefix = "bi_state_event_graph" if use_target_states else "bi_event_graph"
    plt.savefig(f"value_function_plots/{experiment_name}/{prefix}_episode_{episode}_seed_{seed}.png")
    plt.close()


def visualize_chain_graph(planner, episode, experiment_name, seed):
    from simple_rl.agents.func_approx.dsc.OptionClass import Option

    def _get_option_representative_point(option):
        effect_positions = np.array([planner.mdp.get_position(state) for state in option.effect_set])
        return np.median(effect_positions, axis=0)

    def _get_event_representative_point(event):
        if event.get_target_position() is not None:
            return event.get_target_position()
        trigger_positions = [event._get_position(s) for s in event.trigger_points]
        trigger_positions = np.array(trigger_positions)
        return np.median(trigger_positions, axis=0)

    def _get_representative_point(event):
        if isinstance(event, Option):
            return _get_option_representative_point(event)
        elif isinstance(event, SalientEvent):
            return _get_event_representative_point(event)
        raise ValueError(event)

    def _plot_pair(xa, ya, xb, yb, marker="o-", edge_color="black"):
        x = [xa, ya]; y = [xb, yb]
        plt.plot(x, y, marker, c=edge_color, alpha=0.1)
        plt.scatter(x, y, c="black")

    for chain in planner.chainer.chains:
        if chain.is_chain_completed():
            if chain.completing_vertex is not None:
                x1, y1 = _get_representative_point(chain.completing_vertex[0])
                x2, y2 = _get_representative_point(chain.target_salient_event)
                _plot_pair(x1, y1, x2, y2)

    plt.xticks([])
    plt.yticks([])

    x_low_lim, y_low_lim = planner.mdp.get_x_y_low_lims()
    x_high_lim, y_high_lim = planner.mdp.get_x_y_high_lims()

    plt.xlim((x_low_lim, x_high_lim))
    plt.ylim((y_low_lim, y_high_lim))

    plt.savefig(f"value_function_plots/{experiment_name}/chain_graph_episode_{episode}_seed_{seed}.png")
    plt.close()

def plot_dco_salient_event(ax, salient_event, states):
    option = salient_event.covering_option
    is_low = salient_event.is_low

    target_state = salient_event.target_state

    # option_term = option.batched_is_term_true(states, is_low)
    # option_init_states, option_term_states = states[~option_term], states[option_term]
    # option_init_values = option.initiation_classifier(option_init_states).flatten()
    values = option.evaluate(states)

    print(f"For {salient_event}, Max-F: {max(values)}, Min-F: {min(values)}")

    cmap = matplotlib.cm.get_cmap('Blues')
    normalize = matplotlib.colors.Normalize(vmin=min(values), vmax=max(values))
    colors = [cmap(normalize(v)) for v in values]

    ax.scatter(states[:, 0], states[:, 1], c=colors)

    # # salient event initiation set is a subset of the covering option termination set
    # event_init_states = np.array([s for s in option_term_states if salient_event.is_init_true(s)])
    # event_not_init_states = np.array([s for s in option_term_states if not salient_event.is_init_true(s)])
    #
    # if len(event_not_init_states) > 0:
    #     ax.scatter(event_not_init_states[:, 0], event_not_init_states[:, 1], c="orange")
    # if len(event_init_states) > 0:
    #     ax.scatter(event_init_states[:, 0], event_init_states[:, 1], c="green")

    ax.scatter([target_state[0]], [target_state[1]], c="black", s=[300])

    low_bound_x, up_bound_x = -10, 10
    low_bound_y, up_bound_y = -10, 10

    ax.set_xlim((low_bound_x, up_bound_x))
    ax.set_ylim((low_bound_y, up_bound_y))

    ax.set_xlabel("x")
    ax.set_ylabel("y")

def plot_dco_salient_event_comparison(low_event, high_event, replay_buffer, episode, reject_low, reject_high, experiment_name=""):
    fig, axs = plt.subplots(2, figsize=(8, 10))

    # states = np.array(replay_buffer.sample(min(4000, len(replay_buffer)), get_tensor=False)[0])
    states = np.array([trans.state for trans in replay_buffer.memory])
    plot_dco_salient_event(axs[0], low_event, states)
    plot_dco_salient_event(axs[1], high_event, states)

    # global_states = np.array(global_buffer.sample(min(4000, len(global_buffer)), get_tensor=False)[0])
    # plot_dco_salient_event(axs[0, 0], global_low_event, global_states)
    # plot_dco_salient_event(axs[1, 0], global_high_event, global_states)

    # smdp_states = np.array(smdp_buffer.sample(min(4000, len(smdp_buffer)), get_tensor=False)[0])
    # plot_dco_salient_event(axs[0, 1], smdp_low_event, smdp_states)
    # plot_dco_salient_event(axs[1, 1], smdp_high_event, smdp_states)

    option = low_event.covering_option
    name = option.name
    threshold = 0. # option.threshold
    beta = option.beta

    axs[0].set_title(f"Min (replay buffer of size {len(replay_buffer)})")
    axs[1].set_title(f"Max (replay buffer of size {len(replay_buffer)})")
    # axs[0, 1].set_title(f"SMDP Min (replay buffer of size {len(smdp_buffer)})")
    # axs[1, 1].set_title(f"SMDP Max (replay buffer of size {len(smdp_buffer)})")

    if not os.path.isdir(f"initiation_set_plots/{experiment_name}"):
        os.makedirs(f"initiation_set_plots/{experiment_name}")

    fig.suptitle(f"Salient events with threshold={threshold} and beta={beta:.4f}")
    plt.savefig("initiation_set_plots/{}/{}_threshold_{}-episode_{}-reject_({}, {}).png".format(experiment_name, name, threshold, episode, reject_low, reject_high))
    plt.close()

def get_intersecting_events(source_chain, events):
    """ events come from the plan_graph """
    connecting_events = []
    for event in events:
        for option in source_chain.options:
            if source_chain.should_exist_edge_from_option_to_event(option, event) and \
                    event != source_chain.init_salient_event:
                connecting_events.append(event)
    return connecting_events


def visualize_graph_with_all_edges(planner, chains, experiment_name,
                                   plot_completed_events, background_img_fname="ant_maze_big_domain"):

    global kGraphIterationNumber

    def _plot_event_pair(event1, event2):
        if event1.target_state is not None and event2.target_state is not None:
            x = [event1.target_state[0], event2.target_state[0]]
            y = [event1.target_state[1], event2.target_state[1]]
            plt.plot(x, y, "o-", c="black")

    sns.set_style("white")

    plt.figure()

    completed = lambda chain: chain.is_chain_completed() if plot_completed_events else True

    forward_chains = [chain for chain in chains if not chain.is_backward_chain and completed(chain)]
    events = [event for event in planner.plan_graph.plan_graph.nodes if isinstance(event, SalientEvent)]

    for chain in forward_chains:
        intersecting_events = get_intersecting_events(chain, events)
        for event in intersecting_events:
            _plot_event_pair(chain.init_salient_event, event)
        _plot_event_pair(chain.init_salient_event, chain.target_salient_event)

    plt.xticks([]); plt.yticks([])

    x_low_lim, y_low_lim = chains[0].options[0].overall_mdp.get_x_y_low_lims()
    x_high_lim, y_high_lim = chains[0].options[0].overall_mdp.get_x_y_high_lims()

    filename = os.path.join(os.getcwd(), f"{background_img_fname}.png")
    if os.path.isfile(filename):
        background_image = imageio.imread(filename)
        plt.imshow(background_image, zorder=0, alpha=0.5, extent=[x_low_lim, x_high_lim, y_low_lim, y_high_lim])

    plt.xlim((x_low_lim, x_high_lim))
    plt.ylim((y_low_lim, y_high_lim))

    plt.savefig(f"value_function_plots/{experiment_name}/event_graphs_episode_{kGraphIterationNumber}.png")
    plt.close()

def make_arrow(a, b):
    head_length = 0.25
    dx = b[0] - a[0]
    dy = b[1] - a[1]

    vec_ab_magnitude = math.sqrt(dx**2+dy**2)

    dx = dx / vec_ab_magnitude
    dy = dy / vec_ab_magnitude
    vec_ab_magnitude = vec_ab_magnitude - head_length

    ax = plt.axes()
    ax.arrow(a[0], a[1], vec_ab_magnitude*dx, vec_ab_magnitude*dy, head_width=head_length, head_length=head_length, ec='black')
    plt.scatter(a[0],a[1],color='black')
    plt.scatter(b[0],b[1],color='black')

def visualize_bi_graph(planner, mdp):
    events = mdp.all_salient_events_ever + [mdp.get_start_state_salient_event()]
    for event1 in events:
        for event2 in events:
            if event1 != event2:
                if planner.plan_graph.does_path_exist(event1.target_state, event2):
                    make_arrow(event1.target_state, event2.target_state)

def visualize_mpc_rollout_and_graph_result(planner, goal_state, start_state, result_state, steps_taken, episode,
                                           rejected, experiment_name, use_target_states=True):
    def _get_representative_point(event):
        assert isinstance(event, SalientEvent)
        if event.get_target_position() is not None:
            return event.get_target_position()
        trigger_positions = [event._get_position(s) for s in event.trigger_points]
        trigger_positions = np.array(trigger_positions)
        return trigger_positions.mean(axis=0)

    def _plot_event_pair(e1, e2, marker="o-", edge_color="black"):
        x1, y1 = _get_representative_point(e1)
        x2, y2 = _get_representative_point(e2)
        x = [x1, x2]; y = [y1, y2]
        plt.plot(x, y, marker, c=edge_color, alpha=0.1)
        plt.scatter(x, y, c="black")

    def is_connected(n1, n2):
        if use_target_states:
            return planner.plan_graph.does_path_exist(_get_representative_point(n1), n2)
        return planner.plan_graph.does_path_exist_between_nodes(n1, n2)

    events = planner.mdp.all_salient_events_ever + [planner.mdp.get_start_state_salient_event()]

    for event1 in events:
        for event2 in events:
            if event1 != event2:
                if is_connected(event1, event2) and is_connected(event2, event1):
                    _plot_event_pair(event1, event2, "o-", edge_color="black")
                elif is_connected(event1, event2) or is_connected(event2, event1):
                    _plot_event_pair(event1, event2, "o--", edge_color="red")

    x_low_lim, y_low_lim = planner.mdp.get_x_y_low_lims()
    x_high_lim, y_high_lim = planner.mdp.get_x_y_high_lims()

    plt.xlim((x_low_lim, x_high_lim))
    plt.ylim((y_low_lim, y_high_lim))

    # plot mpc points
    plt.scatter(goal_state[0], goal_state[1], color='purple', label='goal state')
    plt.scatter(start_state[0], start_state[1], color='green', label='start state')
    plt.scatter(result_state[0], result_state[1], color='pink', label='MPC rollout')

    plt.title(f"Steps Taken {steps_taken}, Rejected: {rejected}")

    plt.savefig(f"value_function_plots/{experiment_name}/mpc_episode_{episode}_{rejected}.png")
    plt.close()

def visualize_mpc_train_data_distribution(mdp, states, episode, experiment_name):
    sns.set_style("white")
    plt.figure()

    x_low_lim, y_low_lim = mdp.get_x_y_low_lims()
    x_high_lim, y_high_lim = mdp.get_x_y_high_lims()

    plt.xlim((x_low_lim, x_high_lim))
    plt.ylim((y_low_lim, y_high_lim))

    x_list = []
    y_list = []

    for state in states:
        x_list.append(state[0])
        y_list.append(state[1])

    plt.scatter(x_list, y_list, s=2, color='blue', alpha=0.4)

    plt.title("Data Distribution for {} points".format(len(states)))

    plt.savefig(f"value_function_plots/{experiment_name}/mpc_train_data_episode_{episode}.png")
    plt.close()

def visualize_graph_nodes_with_vf_expansion_probabilities(planner, episode, experiment_name, seed):
    def _get_representative_point(node):
        if isinstance(node, SalientEvent):
            return _get_event_representative_point(node)
        return _get_option_representative_point(node)

    def _get_event_representative_point(event):
        assert isinstance(event, SalientEvent)
        if event.get_target_position() is not None:
            return event.get_target_position()
        trigger_positions = [event._get_position(s) for s in event.trigger_points]
        trigger_positions = np.array(trigger_positions)
        return trigger_positions.mean(axis=0)

    def _get_option_representative_point(option):
        effect_positions = np.array([planner.mdp.get_position(state) for state in option.effect_set])
        return np.median(effect_positions, axis=0)

    nodes = planner.get_candidate_nodes_for_expansion()
    rnd_scores = planner.get_rnd_scores(nodes)
    distance_scores = planner.get_graph_distance_scores(nodes)
    count_scores = planner.get_count_based_scores(nodes)
    combined_scores = planner.combine_scores(rnd_scores, distance_scores, count_scores)

    points = [_get_representative_point(n) for n in nodes]
    x_coords = [point[0] for point in points]
    y_coords = [point[1] for point in points]

    x_low_lim, y_low_lim = planner.mdp.get_x_y_low_lims()
    x_high_lim, y_high_lim = planner.mdp.get_x_y_high_lims()

    plt.figure(figsize=(10.0, 8.0))
    plt.subplot(2, 2, 1)
    plt.scatter(x_coords, y_coords, c=rnd_scores)
    plt.colorbar()
    plt.title("RND Scores")
    plt.xlim((x_low_lim, x_high_lim))
    plt.ylim((y_low_lim, y_high_lim))

    plt.subplot(2, 2, 2)
    plt.scatter(x_coords, y_coords, c=np.log(count_scores))
    plt.title("Count Scores")
    plt.colorbar()
    plt.xlim((x_low_lim, x_high_lim))
    plt.ylim((y_low_lim, y_high_lim))

    plt.subplot(2, 2, 3)
    plt.scatter(x_coords, y_coords, c=distance_scores)
    plt.colorbar()
    plt.title("Graph Distance Scores")
    plt.xlim((x_low_lim, x_high_lim))
    plt.ylim((y_low_lim, y_high_lim))

    plt.subplot(2, 2, 4)
    plt.scatter(x_coords, y_coords, c=np.log(combined_scores))
    plt.colorbar()
    plt.title("Combined Scores")
    plt.xlim((x_low_lim, x_high_lim))
    plt.ylim((y_low_lim, y_high_lim))

    plt.savefig(f"value_function_plots/{experiment_name}/event_node_prob_graph_episode_{episode}_seed_{seed}.png")
    plt.close()

def chunked_inference(func, data, device=torch.device("cuda"), chunk_size=1000):
    """ Apply and aggregate func on chunked versions of data. """

    num_chunks = int(np.ceil(data.shape[0] / chunk_size))

    if num_chunks == 0:
        return 0.

    current_idx = 0
    chunks = np.array_split(data, num_chunks, axis=0)
    values = np.zeros((data.shape[0],))

    for data_chunk in chunks:
        data_chunk = torch.from_numpy(data_chunk).float().to(device)
        value_chunk = func(data_chunk).cpu().numpy()
        current_chunk_size = len(data_chunk)
        values[current_idx:current_idx + current_chunk_size] = value_chunk
        current_idx += current_chunk_size

    return values

def make_chunked_intrinsic_reward_plot(solver, overall_mdp, episode=None, experiment_name=""):
    if hasattr(solver.replay_buffer, "next_state"):
        next_states = solver.replay_buffer.next_state
    else:
        next_states = solver.replay_buffer.obs2_buf
    rewards = chunked_inference(solver.rnd.get_reward, next_states, device=solver.device)
    x = np.array([state[0] for state in next_states])
    y = np.array([state[1] for state in next_states])

    plt.scatter(x, y, None, c=rewards.squeeze())
    plt.colorbar()
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Replay Buffer Reward Heat Map")

    x_low_lim, y_low_lim = overall_mdp.get_x_y_low_lims()
    x_high_lim, y_high_lim = overall_mdp.get_x_y_high_lims()

    plt.xlim((x_low_lim, x_high_lim))
    plt.ylim((y_low_lim, y_high_lim))

    plt.savefig(f"value_function_plots/{experiment_name}/rnd_reward_map_episode_{episode}.png")
    plt.close()

    return rewards.max()
