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

parser = argparse.ArgumentParser()
parser.add_argument('--experiment_name', type=str)
parser.add_argument('--seed', type=int)
parser.add_argument('--environment', type=str)
parser.add_argument('--num_episodes', type=int)
parser.add_argument('--num_steps', type=int)
parser.add_argument('--device', type=str, default='cpu')
parser.add_argument('--save_plots', action='store_true', default=False)
parser.add_argument('--save_pickles', action='store_true', default=False)
parser.add_argument('--render', action='store_true', default=False)
parser.add_argument('--fixed_epsilon', type=float, default=0)
parser.add_argument('--goal_dimension', type=int)
parser.add_argument('--add_goal', action='append', nargs='+')
args = parser.parse_args()

################################################################################
# PLOTTING AND PICKLING FUNCTIONS
################################################################################

def plot_learning_curve(
    directory: Path, 
    results: np.ndarray, 
    num_steps: int) -> None:

    episode_nums = np.arange(results.shape[0])
    smooth_scores = uniform_filter1d(results[:,1], 10, mode='nearest')

    fig, ax = plt.subplots()
    ax.plot(episode_nums, smooth_scores)

    ax.set(xlabel='score', ylabel='episode', title=f'{num_steps} per episode')
    fig.savefig(directory / 'learning_curve.png')


def plot_value_function(
    directory: Path, 
    environment, 
    solver: DDPGAgent,
    goal_dimension: int,
    goal_threshold: float,
    goal_state: np.ndarray,
    dense_reward: bool
    ) -> None:

    if goal_dimension != 2:
        print('We don\'t have code to plot an n-d value function for n != 2')
        return

    buffer = solver.replay_buffer.memory
    states = np.array([x[0] for x in buffer])
    actions = np.array([x[1] for x in buffer])

    states_tensor = torch.from_numpy(states).float().to(solver.device)
    actions_tensor = torch.from_numpy(actions).float().to(solver.device)
    qvalues = solver.get_qvalues(states_tensor, actions_tensor).cpu().numpy().squeeze(1)
    
    fig, ax = plt.subplots()
    scatter = ax.scatter(states[:, 0], states[:, 1], c = qvalues)
    fig.colorbar(scatter, ax=ax)

    box = environment.observation_space
    ax.set_xlim((box.low[0], box.high[0]))
    ax.set_ylim((box.low[1], box.high[1]))

    ax.add_patch(
        plt.Circle(goal_state, goal_threshold, alpha=0.7, color='gold'))
    
    reward_name = "dense" if dense_reward else "sparse"
    ax.set(xlabel='x axis', ylabel='y axis', title=f'{reward_name} targeting {goal_state}')

    plt.savefig(directory / 'value_function.png')

def pickle_run(directory: Path, solver: DDPGAgent, results: np.ndarray) -> None:
    buffer = solver.replay_buffer.memory
    pickle.dump(buffer, open('buffer.pkl', 'wb'))
    pickle.dump(results, open('results.pkl', 'wb'))

################################################################################
# TRAIN OUR DDPG
################################################################################

def reward_metric(
    state: np.ndarray,
    goal_dimension,
    goal_threshold: float,
    goal_state: np.ndarray,
    dense_reward: bool
    ) -> Tuple[float, bool]:

    distance = np.linalg.norm(state[:goal_dimension] - goal_state)
    is_terminal = distance <= goal_threshold
    if dense_reward:
        reward = distance * -1
    else:
        reward = 10 if is_terminal else -1
    
    return reward, is_terminal

def train_solver(
    solver: DDPGAgent,
    environment, # Gym Environment
    goal_dimension: int,
    goal_threshold: float,
    goal_state: np.ndarray,
    dense_reward: bool,
    fixed_epsilon: float,
    num_episodes: int, 
    num_steps: int, 
    render: bool,
    ) -> Tuple[DDPGAgent, np.ndarray]:
    
    results = np.zeros((num_episodes, 2))
    for episode in range(num_episodes):
        print(f'Executing episode {episode}/{num_episodes}')
        state = environment.reset()
        for _ in range(num_steps):
            # (i) Render the current state
            if render:
                environment.render()

            # (ii) Get our next action
            if fixed_epsilon < np.random.random():
                action = environment.action_space.sample()
            else:
                action = solver.act(state)
            
            # (iii) Take a step and get our rewards
            next_state, _, _, _ = environment.step(action)
            reward, is_terminal = reward_metric(
                next_state, 
                goal_dimension, 
                goal_threshold, 
                goal_state, 
                dense_reward
                )

            # (iv) Train on and log our transition
            solver.step(state, action, reward, next_state, is_terminal)
            results[episode] += 1, reward

            # Iterate to the next state
            state = next_state
            if is_terminal:
                break
            
        print(f'Episode terminated in {results[episode,0]} steps with reward {results[episode,1]}')
    
    return solver, results


################################################################################
# MAIN FUNCTION
################################################################################

def main():
    directory = Path.cwd() / f'{args.experiment_name}_{args.seed}'
    os.mkdir(directory)

    environment = gym.make(args.environment)
    np.random.seed(args.seed)
    environment.seed(args.seed)

    for i, goal in enumerate(args.add_goal):
        # (i) Set up our goal state
        goal_threshold = float(goal[0])
        goal_state = np.array(goal[1:1 + args.goal_dimension], dtype=np.float)
        dense_reward = goal[3] == 'True'

        # (ii) Set up our DDPG
        solver = DDPGAgent(
            environment.observation_space.shape[0],
            environment.action_space.shape[0],
            args.seed,
            args.device
            )
        
        # (iii) Run our training
        solver, results = train_solver(
            solver,
            environment,
            args.goal_dimension,
            goal_threshold,
            goal_state,
            dense_reward,
            args.fixed_epsilon,
            args.num_episodes,
            args.num_steps,
            args.render
            )

        # (iv) And save our results
        subdirectory = directory / f'goal_{i}'
        os.mkdir(subdirectory)
        if args.save_plots:
            plot_learning_curve(
                subdirectory, 
                results, 
                args.num_steps
                )

            plot_value_function(
                subdirectory, 
                environment, 
                solver, 
                args.goal_dimension,
                goal_threshold, 
                goal_state, 
                dense_reward
                )
        
        if args.save_pickles:
            pickle_run(subdirectory, solver, results)


if __name__ == '__main__':
    main()


    