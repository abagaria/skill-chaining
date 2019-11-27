import numpy as np
import torch
import matplotlib.pyplot as plt
import pickle
import glob
import seaborn as sns
sns.set_style("white")
import scipy

from simple_rl.tasks.pinball.PinballStateClass import PinballState


def get_novel_states(state_space, visited_states):
    state_array = state_space
    novel_states = []
    for state in state_array:  # type: tuple
        novelty = state not in visited_states
        if novelty:
            novel_states.append(state)
    return np.array(novel_states)

def get_four_rooms_state_space(overall_mdp):
    states = []
    for x in range(1, overall_mdp.width + 1):
        for y in range(1, overall_mdp.height + 1):
            if (x, y) not in overall_mdp.walls:
                states.append([x, y])
    return np.array(states)

def get_mountain_car_state_space(x_low, x_high, x_dot_low, x_dot_high):
    states = []
    for x in np.arange(x_low, x_high, 0.01):
        for x_dot in np.arange(x_dot_low, x_dot_high, 0.01):
            states.append([np.round(x, 1), np.round(x_dot, 2)])
    return np.array(states)

def get_novel_state_action_pairs(self, batch_size=64):
    """ Novelty oracle. """
    state_array = self.state_space
    np.random.shuffle(state_array)
    novel_states, novel_actions = [], []
    for state in state_array:
        for action in self.actions:
            s_a_pair = (state[0], state[1], action)
            if self.novelty_detection_method == "oracle":
                novelty = s_a_pair not in self.visited_state_action_pairs
            elif self.novelty_detection_method == "ocsvm":
                novelty = self.one_class_classifier.predict(np.array(s_a_pair).reshape(1, -1)) != 1
                # oracle_novelty = s_a_pair not in self.visited_state_action_pairs
            else:
                raise NotImplementedError(self.novelty_detection_method)

            if novelty:
                novel_states.append(state)
                novel_actions.append(action)

            assert len(novel_states) == len(novel_actions)

            if len(novel_states) == batch_size:
                return np.array(novel_states), np.array(novel_actions)

    # In case we did not collect batch_size number of novel s, a pairs
    return np.array(novel_states), np.array(novel_actions)

def get_classification_errors(self):
    """ When training OC-SVM, compute false positive and false negative rates during training. """
    state_array = self.state_space
    np.random.shuffle(state_array)

    # False negative and false positive (s, a) pairs
    missed_novelty, false_positives = [], []

    for state in state_array:
        for action in self.actions:
            s_a_pair = (state[0], state[1], action)
            novelty = self.one_class_classifier.predict(np.array(s_a_pair).reshape(1, -1)) != 1
            oracle_novelty = s_a_pair not in self.visited_state_action_pairs
            if oracle_novelty and not novelty:
                missed_novelty.append(s_a_pair)
            if novelty and not oracle_novelty:
                false_positives.append(s_a_pair)

    return missed_novelty, false_positives

def get_grid_states():
    ss = []
    for x in np.arange(0., 1.1, 0.1):
        for y in np.arange(0., 1.1, 0.1):
            s = PinballState(x, y, 0, 0)
            ss.append(s)
    return ss

def get_init_values(clf):
    values = []
    for x in np.arange(0., 1.1, 0.1):
        for y in np.arange(0., 1.1, 0.1):
            v = []
            for vx in [-2, -1, 0., 1, 1]:
                for vy in [-2, -1, 0., 1, 1]:
                    features = np.array([x, y, vx, vy])
                    v.append(clf.predict(features.reshape(1, -1)) == 1)
            values.append(np.sum(v))

    return values

def plot_one_class_initiation_classifier(agent, action, episode=None, experiment_name="", seed=0):
    plt.figure(figsize=(8.0, 5.0))

    states = get_grid_states()
    x = np.array([state[0] for state in states])
    y = np.array([state[1] for state in states])
    xi, yi = np.linspace(x.min(), x.max(), 1000), np.linspace(y.min(), y.max(), 1000)
    xx, yy = np.meshgrid(xi, yi)

    clf = agent.one_class_classifiers[action]

    values = get_init_values(clf)

    rbf = scipy.interpolate.Rbf(x, y, values, function="linear")
    zz = rbf(xx, yy)
    plt.imshow(zz, vmin=min(values), vmax=max(values), extent=[x.min(), x.max(), y.min(), y.max()], origin="lower")

    # For pinball only
    plt.xlim((0, 1))
    plt.ylim((0, 1))
    plt.gca().invert_yaxis()
    plt.colorbar()

    plt.xlabel("x")
    plt.ylabel("y")
    name = "{}_{}".format(experiment_name, episode)
    plt.title("OC-SVM for Action {} @ Episode {}".format(action, episode))
    plt.savefig("{}/{}_{}_one_class_svm_action_{}.png".format(experiment_name, name, seed, action))
    plt.close()

def visualize_value_function(agent, experiment_name, episode, seed):
    states = agent.state_space  # numpy array of shape |S| x 2
    agent.policy_network.eval()
    with torch.no_grad():
        values = torch.max(agent.policy_network(torch.from_numpy(states).float().to(agent.device)), dim=1)[0]
    agent.policy_network.train()
    values = values.cpu().numpy()
    x = []; y = []; v = []

    for state, value in zip(states, values):
        x.append(state[0])
        y.append(state[1])
        v.append(value)

    plt.scatter(x, y, c=v)
    plt.colorbar()
    plt.savefig("{}/value_function_episode_{}_seed_{}".format(experiment_name, episode, seed))
    plt.close()

def visualize_sampled_value_function(agent, x_low, x_high, y_low, y_high, experiment_name, episode, seed):
    states = [transition.next_state for transition in agent.replay_buffer.memory]
    states = torch.from_numpy(np.array(states)).float().to(agent.device)
    values = agent._get_q_next(states, network="policy").detach().max(1)[0]

    states = states.cpu().numpy()
    values = values.cpu().numpy()

    x = [state[0] for state in states]
    y = [state[1] for state in states]
    plt.scatter(x, y, c=values)
    plt.title("Value Function @ Episode {}".format(episode))
    plt.colorbar()
    plt.xlim((x_low, x_high))
    plt.ylim((y_low, y_high))

    # For Pinball only
    plt.gca().invert_yaxis()

    plt.savefig("{}/sampled_value_function_episode_{}_seed_{}".format(experiment_name, episode, seed))
    plt.close()

def save_model(dqn_agent, episode, experiment_name):
    torch.save(dqn_agent.policy_network.state_dict(), "{}/saved_runs/policy_net_e{}".format(experiment_name, episode))

def load_model(dqn_agent, episode, experiment_name):
    policy_net_state = torch.load("{}/saved_runs/policy_net_e{}".format(experiment_name, episode))
    dqn_agent.policy_network.load_state_dict(policy_net_state)
    return dqn_agent

def save_replay_buffer(dqn_agent, episode, experiment_name):
    with open("{}/saved_runs/replay_buffer_e{}.pkl".format(experiment_name, episode), "wb") as f:
        pickle.dump(dqn_agent.replay_buffer, f)
