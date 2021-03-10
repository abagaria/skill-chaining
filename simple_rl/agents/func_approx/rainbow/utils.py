import ipdb
from tqdm import tqdm
import torch
import random
import itertools
import numpy as np
import matplotlib.pyplot as plt
from simple_rl.agents.func_approx.rainbow.RainbowAgentClass import RainbowAgent


def get_states_from_replay_buffer(agent, num_states=1000):
    experiences = agent.agent.replay_buffer.sample(num_states)
    experiences = list(itertools.chain.from_iterable(experiences))[:num_states]
    states = np.array([experience['next_state'] for experience in experiences])
    return states


def render_state(state, filename="monte.png"):
    import cv2
    cv2.imwrite(filename, state[-1,:,:])


def render_states_with_max_val(agent, states):
    if not isinstance(states, np.ndarray):
        states = np.array(states)
    q_values = agent.get_batched_qvalues(states)
    values = q_values.max(dim=1).values
    print("Max value = ", values.max())
    for i, state in enumerate(states):
        if np.isclose(values[i].item(), values.max().item()):
            render_state(state, f"monte_good_states/{i}.png")

def render_states_with_min_val(agent, states):
    if not isinstance(states, np.ndarray):
        states = np.array(states)
    q_values = agent.get_batched_qvalues(states)
    values = q_values.max(dim=1).values
    print("Min value = ", values.min())
    for i, state in enumerate(states):
        if np.isclose(values[i].item(), values.min().item()):
            render_state(state, f"monte_bad_states/{i}.png")

def random_walk_monte(mdp, episodes=100):
    def reset(mdp, diverse_starts=True):
        mdp.reset()
        if diverse_starts:
            x, y = random.choice(mdp.spawn_states)
            mdp.set_player_position(x, y)
            for _ in range(4): mdp.execute_agent_action(0)
    states = []
    for _ in tqdm(range(episodes), desc="Collecting random walk data"):
        reset(mdp)
        while not mdp.cur_state.is_terminal():
            a = mdp.sample_random_action()
            _, sp = mdp.execute_agent_action(a)
            states.append(sp)
    return states


def make_chunked_value_function_plot(solver, episode, seed, experiment_name, state_buffer, chunk_size=1000):
    obs = np.array([state.image for state in state_buffer])
    pos = np.array([state.position for state in state_buffer])

    # Chunk up the inputs so as to conserve GPU memory
    num_chunks = int(np.ceil(obs.shape[0] / chunk_size))

    if num_chunks == 0:
        return 0.

    state_chunks = np.array_split(obs, num_chunks, axis=0)
    qvalues = np.zeros((obs.shape[0],))
    current_idx = 0

    for chunk_number, state_chunk in tqdm(enumerate(state_chunks), desc="Making VF plot"):  # type: (int, np.ndarray)
        state_chunk = torch.from_numpy(state_chunk).float().to(solver.device)
        assert isinstance(solver, PFRLAgent)
        chunk_qvalues = solver.get_batched_qvalues(state_chunk).cpu().numpy()
        chunk_qvalues = np.max(chunk_qvalues, axis=1)
        current_chunk_size = len(state_chunk)
        qvalues[current_idx:current_idx + current_chunk_size] = chunk_qvalues
        current_idx += current_chunk_size

    plt.figure()
    plt.scatter(pos[:, 0], pos[:, 1], c=qvalues)
    plt.colorbar()
    file_name = f"{solver.name}_value_function_seed_{seed}_episode_{episode}"
    plt.savefig(f"value_function_plots/{experiment_name}/{file_name}.png")
    plt.close()

    return qvalues.max()