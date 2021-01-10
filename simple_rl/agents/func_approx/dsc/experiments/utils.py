import torch
import scipy
import itertools
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("white")
sns.set_context("talk")

from collections import deque
from treelib import Tree, Node
from simple_rl.agents.func_approx.dqn.DQNAgentClass import DQNAgent


class SkillTree(object):
    def __init__(self, options):
        self._tree = Tree()
        self.options = options

        if len(options) > 0:
            [self.add_node(option) for option in options]

    def add_node(self, option):
        if option.name not in self._tree:
            print(f"Adding {option} to the skill-tree")
            self.options.append(option)
            parent = option.parent.name if option.parent is not None else None
            self._tree.create_node(tag=option.name, identifier=option.name, data=option, parent=parent)

    def get_option(self, option_name):
        if option_name in self._tree.nodes:
            node = self._tree.nodes[option_name]
            return node.data

    def get_depth(self, option):
        return self._tree.depth(option.name)

    def get_children(self, option):
        return self._tree.children(option.name)

    def traverse(self):
        """ Breadth first search traversal of the skill-tree. """
        return list(self._tree.expand_tree(mode=self._tree.WIDTH))

    def show(self):
        """ Visualize the graph by printing it to the terminal. """
        self._tree.show()


def make_meshgrid(x, y, h=.02):
    x_min, x_max = x.min() - 1, x.max() + 1
    y_min, y_max = y.min() - 1, y.max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    return xx, yy

def get_grid_states(mdp):
    ss = []
    x_low_lim, y_low_lim = mdp.get_x_y_low_lims()
    x_high_lim, y_high_lim = mdp.get_x_y_high_lims()
    for x in np.arange(x_low_lim, x_high_lim+1, 1):
        for y in np.arange(y_low_lim, y_high_lim+1, 1):
            ss.append(np.array((x, y)))
    return ss


def get_initiation_set_values(option):
    values = []
    x_low_lim, y_low_lim = option.overall_mdp.get_x_y_low_lims()
    x_high_lim, y_high_lim = option.overall_mdp.get_x_y_high_lims()
    for x in np.arange(x_low_lim, x_high_lim+1, 1):
        for y in np.arange(y_low_lim, y_high_lim+1, 1):
            pos = np.array((x, y))
            init = option.is_init_true(pos)
            if hasattr(option.overall_mdp.env, 'env'):
                init = init and not option.overall_mdp.env.env._is_in_collision(pos)
            values.append(init)
    return values

def plot_one_class_initiation_classifier(option):

    colors = ["blue", "yellow", "green", "red", "cyan", "brown"]

    X = option.construct_feature_matrix(option.positive_examples)
    X0, X1 = X[:, 0], X[:, 1]
    xx, yy = make_meshgrid(X0, X1)
    Z1 = option.pessimistic_classifier.decision_function(np.c_[xx.ravel(), yy.ravel()])
    Z1 = Z1.reshape(xx.shape)

    color = colors[option.option_idx % len(colors)]
    plt.contour(xx, yy, Z1, levels=[0], linewidths=2, colors=[color])

def plot_two_class_classifier(option, episode, experiment_name, plot_examples=True, seed=0):
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
        plt.scatter(negative_examples[:, 0], negative_examples[:, 1], label="negative", c="lime", alpha=1.0, s=10)

    if option.pessimistic_classifier is not None:
        plot_one_class_initiation_classifier(option)

    # background_image = imageio.imread("four_room_domain.png")
    # plt.imshow(background_image, zorder=0, alpha=0.5, extent=[-2.5, 10., -2.5, 10.])

    name = option.name if episode is None else option.name + "_{}_{}".format(experiment_name, episode)
    plt.title("{} Initiation Set".format(option.name))
    plt.savefig("initiation_set_plots/{}/{}_initiation_classifier_{}.png".format(experiment_name, name, seed))
    plt.close()


def plot_initiation_distribution(option, mdp, episode, experiment_name, chunk_size=10000):
    assert option.initiation_distribution is not None
    data = mdp.dataset[:, :2]

    num_chunks = int(np.ceil(data.shape[0] / chunk_size))
    if num_chunks == 0:
        return 0.

    state_chunks = np.array_split(data, num_chunks, axis=0)
    pvalues = np.zeros((data.shape[0],))
    current_idx = 0

    for chunk_number, state_chunk in tqdm(enumerate(state_chunks)):
        probabilities = np.exp(option.initiation_distribution.score_samples(state_chunk))
        pvalues[current_idx:current_idx + len(state_chunk)] = probabilities
        current_idx += len(state_chunk)

    plt.scatter(data[:, 0], data[:, 1], c=pvalues)
    plt.colorbar()
    plt.title("Density Estimator Fitted on Pessimistic Classifier")
    plt.savefig(f"initiation_set_plots/{experiment_name}/{option.name}_initiation_distribution_{episode}.png")
    plt.close()


def make_chunked_goal_conditioned_value_function_plot(solver, goal, episode, seed, experiment_name, chunk_size=1000, replay_buffer=None, option_idx=None):
    sns.set_palette(sns.color_palette("viridis"))
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

    if option_idx is None:
        file_name = f"{solver.name}_value_function_seed_{seed}_episode_{episode}"
    else:
        file_name = f"{solver.name}_value_function_seed_{seed}_episode_{episode}_option_{option_idx}"
    plt.title(f"VF Targeting {np.round(goal, 2)}")
    plt.savefig(f"value_function_plots/{experiment_name}/{file_name}.png")
    plt.close()

    return qvalues.max()

def monte_plot_option_init_set(exp, option, history):
    info = {'positives': {'x': [], 'y': [], 'colors': []},
            'negatives': {'x': [], 'y': [], 'colors': []}}

    for state in history:
        pos = state.get_position()
        x, y = pos[0], pos[1]

        if option.pessimistic_is_init_true(state):
            info['positives']['x'].append(x)
            info['positives']['y'].append(y)
            info['positives']['colors'].append('green')
        else:
            info['negatives']['x'].append(x)
            info['negatives']['y'].append(y)
            info['negatives']['colors'].append('red')
        
    green = sum([1 for c in info['positives']['colors'] if c == "green"])
    print(f"There are {green} points in the pessimistic clf")

    plt.figure()
    plt.xlim((0,140))
    plt.ylim((0,300))
    plt.scatter(info['positives']['x'], info['positives']['y'], c=info['positives']['colors'], alpha=0.5)
    plt.title(f"{option.name} Initiation Set Inside")
    plt.savefig("initiation_set_plots/{}/{}_initiation_classifier_{}_inside.png".format(exp.experiment_name, option.name, exp.seed))
    
    plt.figure()
    plt.xlim((0,140))
    plt.ylim((0,300))
    plt.scatter(info['negatives']['x'], info['negatives']['y'], c=info['negatives']['colors'], alpha=0.5)
    plt.title(f"{option.name} Initiation Set Outside")
    plt.savefig("initiation_set_plots/{}/{}_initiation_classifier_{}_outside.png".format(exp.experiment_name, option.name, exp.seed))
    
    plt.close()