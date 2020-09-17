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
from simple_rl.agents.func_approx.ddpg.DDPGAgentClass import DDPGAgent, her_train, train
from simple_rl.tasks.ant_reacher.AntReacherMDPClass import AntReacherMDP
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
# parser.add_argument('--save_pickles', action='store_true', default=False)
parser.add_argument('--render', action='store_true', default=False)
parser.add_argument('--fixed_epsilon', type=float, default=0)
parser.add_argument('--pretrain_with_her', action='store_true', default=False)
parser.add_argument('--goal_threshold', type=float, default=0.6)
parser.add_argument('--goal_dimension', type=int, default=2)
parser.add_argument('--goal_state', type=float, nargs='+')
parser.add_argument('--dense_reward', action='store_true', default=False)
args = parser.parse_args()

################################################################################
# PLOTTING AND PICKLING FUNCTIONS
################################################################################

def plot_learning_curve(
    directory: Path, 
    pes) -> None:

    fig, ax = plt.subplots()
    ax.plot(pes)

    ax.set(xlabel='score', ylabel='episode', title=f'per episode scores')
    fig.savefig(directory / 'learning_curve.png')

def plot_ant_value_function(
    directory: Path, 
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

    ax.set_xlim((-10, 10))
    ax.set_ylim((-10, 10))

    ax.add_patch(
        plt.Circle(goal_state, goal_threshold, alpha=0.7, color='gold'))
    
    reward_name = 'dense' if dense_reward else 'sparse'
    ax.set(xlabel='x axis', ylabel='y axis', title=f'{reward_name} targeting {goal_state}')

    plt.savefig(directory / 'value_function.png')

# def plot_learning_curve(
#     directory: Path, 
#     per_episode_scores: np.ndarray, 
#     num_steps: int) -> None:

#     fig, ax = plt.subplots()
#     ax.plot(episode_nums, smooth_scores)

#     ax.set(xlabel='score', ylabel='episode', title=f'{num_steps} per episode')
#     fig.savefig(directory / 'learning_curve.png')

# def pickle_run(directory: Path, solver: DDPGAgent, results: np.ndarray) -> None:
#     buffer = solver.replay_buffer.memory
#     pickle.dump(buffer, open(directory / 'buffer.pkl', 'wb'))
#     pickle.dump(results, open(directory / 'results.pkl', 'wb'))

def main():
    directory = Path.cwd() / f'{args.experiment_name}_{args.seed}'
    os.mkdir(directory)

    # (i) Maybe pretrain our DDPG agent using HER
    mdp = AntReacherMDP(seed=args.seed, render=args.render)

    solver = DDPGAgent(
            mdp.state_space_size() + args.goal_dimension,
            mdp.action_space_size(),
            args.seed,
            args.device
            )

    if args.pretrain_with_her:
        pes_i, _ = her_train(
            solver,
            mdp, 
            args.num_episodes, 
            args.num_steps, 
            goal_state=None, 
            sampling_strategy="diverse"
            )
    
    # (ii) Test on a fixed-goal domain, maybe pretrained
    goal_state = np.array(args.goal_state)
    pes_ii, _ = her_train(
        solver,
        mdp,
        args.num_episodes,
        args.num_steps,
        goal_state = goal_state,
        sampling_strategy="test"
        )
    
    plot_learning_curve(directory, pes_ii)
    plot_ant_value_function(directory, solver, args.goal_dimension, args.goal_threshold, goal_state, args.dense_reward)

if __name__ == '__main__':
    main()


    