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

# Other imports
from simple_rl.agents.func_approx.ddpg.DDPGAgentClass import DDPGAgent
from typing import List, Tuple, Callable


################################################################################
# ARGUMENT PARSING
################################################################################

def parse_goal_states(goal_dimension: int, goal_indices: List[float]) -> List[np.ndarray]:
    assert(len(goal_indices) % goal_dimension == 0, 'Invalid goal indices')

    goal_states = []
    for i in range(0, len(goal_indices), goal_dimension):
        goal_states.append(np.array(goal_indices[i:i+goal_dimension]))

    return goal_states

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment_name', type=str)
    parser.add_argument('--seed', type=int)
    parser.add_argument('--environment', type=str)
    parser.add_argument('--num_episodes', type=int)
    parser.add_argument('--num_steps', type=int)
    parser.add_argument('--save_plots', type=bool, action='store_true', default=False)
    parser.add_argument('--save_pickles', type=bool, action='store_true', default=False)
    parser.add_argument('--dense_reward', type=bool, action='store_true', default=False)
    parser.add_argument('--goal_threshold', type=int)
    parser.add_argument('--goal_indices', nargs='+', type=float)
    return parser.parse_args()

################################################################################
# PLOTTING AND PICKLING FUNCTIONS
################################################################################


def plot_learning_curve(directory: Path, results: np.ndarray) -> None:
    episode_nums = np.arange(results.shape[0])
    smooth_scores = uniform_filter1d(results[:,1])

    fig, ax = plt.subplots()
    ax.plot(episode_nums, smooth_scores)

    ax.set(xlabel='score', ylabel='episode')
    fig.savefig(directory / 'learning_curve.png')

def plot_value_function(directory: Path, solver: DDPGAgent) -> None:
    buffer = solver.replay_buffer.memory
    states = np.array([x[0] for x in buffer])
    actions = np.array([x[1] for x in buffer])

    states_tensor = torch.from_numpy(states).float.to(solver.device)
    actions_tensor = torch.from_numpy(actions).float.to(solver.device)
    qvalues = solver.get_qvalues(states_tensor, actions_tensor).cpu().numpy.squeeze(1)
    
    fig, ax = plt.subplots()
    scatter = ax.scatter(states[:, 0], states[:, 1], c = qvalues)
    fig.colorbar(scatter, ax=ax)

    plt.savefig(directory / 'value_function.png')

def pickle_run(directory: Path, solver: DDPGAgent, results: np.ndarray) -> None:
    buffer = solver.replay_buffer.memory
    pickle.dump(buffer, open('buffer.pkl', 'wb'))
    pickle.dump(results, open('results.pkl', 'wb'))

################################################################################
# TRAIN OUR DDPG
################################################################################

def train_solver(
    solver: DDPGAgent,
    environment, # Gym Environment
    reward_metric: Callable[np.ndarray, Tuple[float, bool]],
    num_episodes: int, 
    num_steps: int, 
    ) -> Tuple[...,np.ndarray]:
    
    results = np.zeros(num_episodes, 2)
    for episode in range(num_episodes):
        state = environment.reset()
        for _ in range(num_steps):
            action = solver.act(state)
            next_state, _, _, _ = environment.step(action)
            reward, is_terminal = reward_metric(next_state)

            solver.step(state, action, reward, next_state, reward, is_terminal)
            results[episode] += 1, reward

            state = next_state
            if is_terminal:
                break
    
    return solver, results


################################################################################
# MAIN FUNCTION
################################################################################

def get_ball_reward_metric(
    goal_dimension: int,
    goal_threshold: float, 
    goal_state: np.ndarray, 
    dense_reward: bool
    ) -> Callable[np.ndarray, float]:

    def reward_metric(state):
        distance = np.linalg.norm(goal_state[:goal_dimension] - state[:goal_dimension])
        is_terminal = distance < goal_threshold
        if dense_reward:
            reward = -1 * distance
        else:
            reward = 10 if is_terminal else -1

        return reward, is_terminal

    return reward_metric

def main():
    args = parse_args()
    directory = Path.cwd() / f'{args.experiment_name}_{args.seed}'

    if args.environment == 'ant':
        environment = gym.make('Ant-v2')
        goal_dimension = 2

    goal_states = parse_goal_states(goal_dimension, args.goal_indices)

    for i, goal_state in enumerate(goal_states):
        # Set up our goal state
        reward_metric = get_ball_reward_metric(
            goal_dimension, 
            args.goal_threshold, 
            goal_state, 
            args.dense_reward
            )

        # Set up our DDPG
        solver = DDPGAgent(
            environment.observation_space.shape[0],
            environment.action_space.shape[0],
            args.seed,
            args.device
            )
        
        # Run our training
        solver, results = train_solver(
            solver,
            environment,
            reward_metric,
            args.num_episodes,
            args.num_steps)

        # And save our results
        subdirectory = directory / f'goal_{i}'
        if args.save_plots:
            plot_learning_curve(subdirectory, results)
            plot_value_function(subdirectory, solver)
        
        if args.save_pickles:
            pickle_run(subdirectory, solver, results)
            

if __name__ == '__main__':
    main()


    