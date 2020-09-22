# Core imports
import numpy as np
import gym
import torch
import ipdb

# I/O imports
import os
import argparse 
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import pickle
from scipy.ndimage.filters import uniform_filter1d
from collections import deque

# Other imports
from simple_rl.tasks.ant_reacher.AntReacherMDPClass import AntReacherMDP
from typing import List, Tuple, Callable

from copy import deepcopy

def plot_learning_curve(
    directory: Path, 
    title,
    pes) -> None:

    smoothed_pes = uniform_filter1d(pes, 20, mode='nearest')

    fig, ax = plt.subplots()
    ax.plot(smoothed_pes)

    ax.set(xlabel='score', ylabel='episode', title=title)
    fig.savefig(directory / 'learning_curve.png')

    plt.close()

def make_chunked_goal_conditioned_value_function_plot(directory, solver, goal, episode, seed, experiment_name, chunk_size=1000, replay_buffer=None):
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
    for chunk_number, (state_chunk, action_chunk) in enumerate(zip(state_chunks, action_chunks)): 
        state_chunk = torch.from_numpy(state_chunk).float().to(solver.device)
        action_chunk = torch.from_numpy(action_chunk).float().to(solver.device)
        chunk_qvalues = solver.get_qvalues(state_chunk, action_chunk).cpu().numpy().squeeze(1)
        current_chunk_size = len(state_chunk)
        qvalues[current_idx:current_idx + current_chunk_size] = chunk_qvalues
        current_idx += current_chunk_size
    plt.scatter(states[:, 0], states[:, 1], c=qvalues)
    plt.colorbar()
    file_name = f"{solver.name}_value_function_seed_{seed}_episode_{episode}"
    plt.title(f"VF Targeting {np.round(goal, 2)}")
    plt.savefig(f"{directory}/{file_name}.png")
    plt.close()
    return qvalues.max()

def pickle_solver(directory, solver):
    pickle.dump(solver, open(directory / 'ddpg.pkl', 'wb'))

def her_rollout(agent, goal, mdp, steps, dense_reward):
    score = 0.
    mdp.reset()
    trajectory = []

    mdp.set_current_goal(goal)

    for step in range(steps):
        state = deepcopy(mdp.cur_state)
        aug_state = np.concatenate((state.features(), goal), axis=0)

        if np.random.uniform() < 0.25:
            action = mdp.sample_random_action()
        else:
            action = agent.act(aug_state)

        _, next_state = mdp.execute_agent_action(action)
        agent.update_epsilon()

        reward_func = mdp.dense_gc_reward_function if dense_reward else mdp.sparse_gc_reward_function
        reward, done = reward_func(next_state, goal, {})

        score = score + reward
        trajectory.append((state, action, reward, next_state))

        if done:
            break

    return score, trajectory

# I totally butchered this function on this branch, def don't merge this back in to main -Kiran
def her_train(agent, mdp, episodes, steps, goal_state=None, sampling_strategy="fixed", dense_reward=False):

    assert sampling_strategy in ("fixed", "diverse", "test"), sampling_strategy
    if sampling_strategy == "test": assert goal_state is not None, goal_state

    trajectories = []
    per_episode_scores = []
    last_10_scores = deque(maxlen=10)

    for episode in range(episodes):

        # Set the goal to a feasible random state
        if sampling_strategy == "fixed":
            goal_state = np.array([0., 8.])
        if sampling_strategy == "diverse":
            goal_state = np.random.uniform([-4,-4], [4,4])

        # Roll-out current policy for one episode
        score, trajectory = her_rollout(agent, goal_state, mdp, steps, dense_reward)

        # Debug log the trajectories
        trajectories.append(trajectory)

        # Regular Experience Replay
        for state, action, _, next_state in trajectory:

            reward_func = mdp.dense_gc_reward_function if dense_reward else mdp.sparse_gc_reward_function
            reward, done = reward_func(next_state, goal_state, {})

            augmented_state = np.concatenate((state.features(), goal_state), axis=0)
            augmented_next_state = np.concatenate((next_state.features(), goal_state), axis=0)
            agent.step(augmented_state, action, reward, augmented_next_state, done)

        # Logging
        per_episode_scores.append(score)
        last_10_scores.append(score)
        print(f"[Goal={goal_state}] Episode: {episode} \t Score: {score} \t Average Score: {np.mean(last_10_scores)}")

    return per_episode_scores, trajectories

if __name__ == '__main__':
    directory = Path.cwd() / "ddpg_empty_replay_buffer_0"
    os.mkdir(directory)

    # (i) Maybe pretrain our DDPG agent using HER
    mdp = AntReacherMDP(
        seed=0, 
        render=False, 
        goal_reward=0.
        )
    
    solver = pickle.load(open('ddpg_with_her_big_buffer_long_0', 'rb'))
    solver.replay_buffer.memory = deque(maxlen=int(1e6))

    # (ii) Test on a fixed-goal domain, maybe pretrained
    goal_state = np.array([3.8,3.7])
    pes_ii, _ = her_train(
        solver,
        mdp,
        1500,
        1000,
        goal_state = goal_state,
        sampling_strategy="test",
        dense_reward=True
        )
    
    plot_learning_curve(directory, 'test time', pes_ii)
    make_chunked_goal_conditioned_value_function_plot(directory, solver, goal_state, 1500, 0, "ddpg_empty_replay_buffer")

    pickle_solver(directory, solver)

    ipdb.set_trace()