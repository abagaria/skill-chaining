import os
import time
import ipdb
import torch
import copy
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt


def save(td3_agent, filename):
    torch.save(td3_agent.critic.state_dict(), filename + "_critic")
    torch.save(td3_agent.critic_optimizer.state_dict(), filename + "_critic_optimizer")

    torch.save(td3_agent.actor.state_dict(), filename + "_actor")
    torch.save(td3_agent.actor_optimizer.state_dict(), filename + "_actor_optimizer")


def load(td3_agent, filename):
    td3_agent.critic.load_state_dict(torch.load(filename + "_critic"))
    td3_agent.critic_optimizer.load_state_dict(torch.load(filename + "_critic_optimizer"))
    td3_agent.critic_target = copy.deepcopy(td3_agent.critic)

    td3_agent.actor.load_state_dict(torch.load(filename + "_actor"))
    td3_agent.actor_optimizer.load_state_dict(torch.load(filename + "_actor_optimizer"))
    td3_agent.actor_target = copy.deepcopy(td3_agent.actor)

class RunningMeanStd(object):
    # https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm
    def __init__(self, epsilon=1e-4, shape=()):
        self.mean = np.zeros(shape, 'float64')
        self.var = np.ones(shape, 'float64')
        self.count = epsilon

    def update(self, x):
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0]
        self.update_from_moments(batch_mean, batch_var, batch_count)

    def update_from_moments(self, batch_mean, batch_var, batch_count):
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * (self.count)
        m_b = batch_var * (batch_count)
        M2 = m_a + m_b + np.square(delta) * self.count * batch_count / (self.count + batch_count)
        new_var = M2 / (self.count + batch_count)

        new_count = batch_count + self.count

        self.mean = new_mean
        self.var = new_var
        self.count = new_count


class RewardForwardFilter(object):
    def __init__(self, gamma):
        self.rewems = None
        self.gamma = gamma

    def update(self, rews):
        if self.rewems is None:
            self.rewems = rews
        else:
            self.rewems = self.rewems * self.gamma + rews
        return self.rewems

# -------------------------------------------------------------------------------------------------
# Visualization functions
# -------------------------------------------------------------------------------------------------

def _get_intrinsic_and_extrinsic_qvalues(solver, states, actions):

    solver.critic.eval()
    with torch.no_grad():
        q1e, q1i, q2e, q2i = solver.critic(states, actions)
        qe = torch.min(q1e, q2e)
        qi = torch.min(q1i, q2i)
    solver.critic.train()
    
    return qi, qe

def make_chunked_value_function_plot(solver, episode, experiment_name, chunk_size=1000, replay_buffer=None):
    replay_buffer = replay_buffer if replay_buffer is not None else solver.replay_buffer
    states = np.array([exp[0] for exp in replay_buffer])
    actions = np.array([exp[1] for exp in replay_buffer])

    # Chunk up the inputs so as to conserve GPU memory
    num_chunks = int(np.ceil(states.shape[0] / chunk_size))

    if num_chunks == 0:
        return 0.

    state_chunks = np.array_split(states, num_chunks, axis=0)
    action_chunks = np.array_split(actions, num_chunks, axis=0)
    
    qvalues_intrinsic = np.zeros((states.shape[0],))
    qvalues_extrinsic = np.zeros((states.shape[0],))

    current_idx = 0

    for chunk_number, (state_chunk, action_chunk) in tqdm(enumerate(zip(state_chunks, action_chunks)), desc="Making VF plot"):  # type: (int, np.ndarray)
        state_chunk = torch.from_numpy(state_chunk).float().to(solver.device)
        action_chunk = torch.from_numpy(action_chunk).float().to(solver.device)
        
        chunk_qvalues_intrinsic, chunk_qvalues_extrinsic = _get_intrinsic_and_extrinsic_qvalues(solver, state_chunk, action_chunk)
        chunk_qvalues_intrinsic = chunk_qvalues_intrinsic.cpu().numpy().squeeze(1)
        chunk_qvalues_extrinsic = chunk_qvalues_extrinsic.cpu().numpy().squeeze(1)

        current_chunk_size = len(state_chunk)
        qvalues_intrinsic[current_idx:current_idx + current_chunk_size] = chunk_qvalues_intrinsic
        qvalues_extrinsic[current_idx:current_idx + current_chunk_size] = chunk_qvalues_extrinsic

        current_idx += current_chunk_size

    plt.figure(figsize=(14, 12))
    plt.subplot(1, 2, 1)
    plt.scatter(states[:, 0], states[:, 1], c=qvalues_intrinsic)
    plt.colorbar()
    plt.title("Intrinsic Q-Function")

    plt.subplot(1, 2, 2)
    plt.scatter(states[:, 0], states[:, 1], c=qvalues_extrinsic)
    plt.colorbar()
    plt.title("Extrinsic Q-function")

    file_name = f"{solver.name}_value_function_episode_{episode}"
    plt.savefig(f"value_function_plots/{experiment_name}/{file_name}.png")
    plt.close()

    return qvalues_intrinsic.max(), qvalues_extrinsic.max()

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
        value_chunk = func(data_chunk)
        current_chunk_size = len(data_chunk)
        values[current_idx:current_idx + current_chunk_size] = value_chunk
        current_idx += current_chunk_size

    return values

def visualize_next_state_intrinsic_reward_heat_map(solver, mdp, episode, experiment_name=""):
    t0 = time.time()
    next_states = [experience[-2] for experience in solver.replay_buffer]
    intrinsic_rewards = np.array([experience[2] for experience in solver.replay_buffer]).clip(-np.inf, 10.)
    x = np.array([state[0] for state in next_states])
    y = np.array([state[1] for state in next_states])

    plt.scatter(x, y, None, c=intrinsic_rewards.squeeze(), cmap=plt.cm.coolwarm)
    plt.colorbar()
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Replay Buffer Reward Heat Map")

    x_low_lim, y_low_lim = mdp.get_x_y_low_lims()
    x_high_lim, y_high_lim = mdp.get_x_y_high_lims()

    plt.xlim((x_low_lim, x_high_lim))
    plt.ylim((y_low_lim, y_high_lim))

    name = f"{solver.name}_{experiment_name}_{episode}"
    plt.savefig(f"value_function_plots/{experiment_name}/{name}_replay_buffer_reward_map.png")
    plt.close()

    return time.time() - t0

def visualize_intrinsic_rewards_on_offline_dataset(solver, mdp, episode, experiment_name):
    t0 = time.time()
    next_states = mdp.dataset
    r_int = chunked_inference(solver.get_reward, next_states)
    x = [state[0] for state in next_states]
    y = [state[1] for state in next_states]

    plt.scatter(x, y, c=r_int, cmap=plt.cm.coolwarm)
    plt.colorbar()
    plt.title("Offline dataset Intrinsic Rewards")

    x_low_lim, y_low_lim = mdp.get_x_y_low_lims()
    x_high_lim, y_high_lim = mdp.get_x_y_high_lims()

    plt.xlim((x_low_lim, x_high_lim))
    plt.ylim((y_low_lim, y_high_lim))

    name = f"{solver.name}_{experiment_name}_{episode}"
    plt.savefig(f"value_function_plots/{experiment_name}/{name}_offline_buffer_reward_map.png")
    plt.close()

    return time.time() - t0

def plot_state_value_as_function_of_reward(solver, state, episode, experiment_name):
    """ When we augment the state-space with the agent's intrinsic rewards, we can vary the reward and see the effect on values. """

    rewards = np.linspace(0., 10., num=100)
    states = np.array([np.concatenate((state, np.array([reward]))) for reward in rewards])

    values = solver.get_values(states)

    plt.plot(rewards, values, "o--")
    plt.xlabel("Reward")
    plt.ylabel("Value")
    plt.title(f"Value of state at position {state[:2]}")
    
    plt.savefig(f"value_function_plots/{experiment_name}/value_of_state_at_{state[:2]}.png")
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