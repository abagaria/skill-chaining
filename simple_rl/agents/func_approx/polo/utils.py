import torch
import numpy as np
import matplotlib.pyplot as plt


def chunked_inference(func, data, device=torch.device("cuda"),
                      chunk_size=1000, returns_tensor=False):
    """ Apply and aggregate func on chunked versions of data. """

    num_chunks = int(np.ceil(data.shape[0] / chunk_size))

    if num_chunks == 0:
        return 0.

    current_idx = 0
    chunks = np.array_split(data, num_chunks, axis=0)
    values = np.zeros((data.shape[0],))

    for data_chunk in chunks:
        data_chunk = torch.from_numpy(data_chunk).float().to(device)
        value_chunk = func(data_chunk).cpu().numpy().squeeze() if returns_tensor else func(data_chunk)
        current_chunk_size = len(data_chunk)
        values[current_idx:current_idx + current_chunk_size] = value_chunk
        current_idx += current_chunk_size

    return values


def visualize_value_function(polo_agent, experiment_name, episode):
    buffer = polo_agent.mpc.replay_buffer
    states = buffer.obs_buf[:buffer.size]

    plt.figure(figsize=(16, 16))

    for i in range(polo_agent.ensemble_size):
        vf = polo_agent.value_function_ensemble.members[i]
        values = chunked_inference(vf.predict, states, returns_tensor=True)
        plt.subplot(3, 2, i+1)
        plt.scatter(states[:, 0], states[:, 1], c=values.squeeze())
        plt.colorbar()
        plt.title(f"VF {i}")

    values = chunked_inference(polo_agent.value, states, polo_agent.device)

    plt.subplot(3, 1, 3)
    plt.scatter(states[:, 0], states[:, 1], c=values)
    plt.colorbar()
    plt.title("Combined VF")
    plt.savefig(f"value_function_plots/{experiment_name}/vf_ensemble_episode_{episode}.png")
    plt.close()


def visualize_value_targets(polo_agent, experiment_name, episode):
    plt.figure(figsize=(18, 18))

    for i in range(polo_agent.ensemble_size):
        states = []
        reward_targets = []
        bootstrap_value_targets = []
        vf = polo_agent.value_function_ensemble.members[i]

        for _ in range(polo_agent.num_gradient_steps):
            sampled_states = polo_agent.value_function_ensemble.sample(i)
            predicted_next_states, _, costs = polo_agent.mpc.batched_simulate(sampled_states,
                                                                              num_steps=polo_agent.horizon,
                                                                              num_rollouts=polo_agent.num_rollouts)
            states.append(sampled_states.cpu().numpy())
            discounted_reward_matrix = polo_agent.get_n_step_returns(costs)
            discounted_rewards = discounted_reward_matrix.mean(dim=1).cpu().numpy().tolist()
            reward_targets += discounted_rewards

            discounted_next_values = polo_agent.get_bootstrap_next_state_values(predicted_next_states, vf)
            mean_discounted_next_values = discounted_next_values.mean(dim=1).cpu().numpy().tolist()
            bootstrap_value_targets += mean_discounted_next_values

        states = np.vstack(states)
        plt.subplot(4, 2, 2*i+1)
        plt.scatter(states[:, 0], states[:, 1], c=reward_targets)
        plt.colorbar()
        plt.title(f"Ensemble {i} Reward Target")

        plt.subplot(4, 2, 2*i+2)
        plt.scatter(states[:, 0], states[:, 1], c=bootstrap_value_targets)
        plt.title(f"Ensemble {i} Bootstrap Value Target")
        plt.colorbar()
    plt.savefig(f"value_function_plots/{experiment_name}/vf_targets_episode_{episode}.png")
    plt.close()
