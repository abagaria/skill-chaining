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
from simple_rl.tasks.gym.GymMDPClass import GymMDP
from typing import List, Tuple, Callable


################################################################################
# ARGUMENT PARSING
################################################################################

def parse_goal(goal):
    d = args.goal_dimension

    if len(goal) <= d + 2:
        try:
            assert goal[0] in ('sparse', 'dense')
            dense_reward = goal[0] == 'dense'

            assert goal[1] == 'targeting'
            goal_state = np.array(goal[2:], dtype=np.float)

            pretrain = False

            num_samples = None

            source_goals = None
        except:
            raise argparse.ArgumentTypeError('Goal not in format: \'(sparse/dense) \
                                              targeting <goal indices>\'')
    else:
        try:
            assert goal[0] in ('sparse', 'dense')
            dense_reward = goal[0] == 'dense'

            assert goal[1] == 'targeting'
            goal_state = np.array(goal[2:d + 2], dtype=np.float)

            pretrain = True

            assert goal[d + 2] == 'sample'
            num_samples = goal[d + 3]

            assert goal[d + 4] == 'from'
            source_goals = []
            for source_goal in goal[d + 5:]:
                try:
                    i = int(source_goal)
                    source_goals.append(i)
                except:
                    i = str(source_goal)
                    source_goals.append(i)
        except:
            raise argparse.ArgumentTypeError('Goal not in format: \'(sparse/dense) \
                                              targeting <goal indices> sampling <num samples> \
                                              from <source goals>\'')

    return dense_reward, goal_state, pretrain, num_samples, source_goals


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
parser.add_argument('--goal_threshold', type=float, default=0.6)
parser.add_argument('--goal_dimension', type=int, default=2)
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
    
    reward_name = 'dense' if dense_reward else 'sparse'
    ax.set(xlabel='x axis', ylabel='y axis', title=f'{reward_name} targeting {goal_state}')

    plt.savefig(directory / 'value_function.png')

def pickle_run(directory: Path, solver: DDPGAgent, results: np.ndarray) -> None:
    buffer = solver.replay_buffer.memory
    pickle.dump(buffer, open(directory / 'buffer.pkl', 'wb'))
    pickle.dump(results, open(directory / 'results.pkl', 'wb'))


def main():
    directory = Path.cwd() / f'{args.experiment_name}_{args.seed}'
    os.mkdir(directory)

    mdp = GymMDP(env_name='Ant-v2', render=args.render)

    solver = DDPGAgent(
        environment.observation_space.shape[0] + args.goal_dimension,
        environment.action_space.shape[0],
        args.seed,
        args.device
        )

        # # (iv) And save our results
        # subdirectory = directory / f'goal_{i}'
        # os.mkdir(subdirectory)
        # if args.save_plots:
        #     plot_learning_curve(
        #         subdirectory, 
        #         results, 
        #         args.num_steps
        #         )

        #     plot_value_function(
        #         subdirectory, 
        #         environment, 
        #         solver, 
        #         args.goal_dimension,
        #         args.goal_threshold, 
        #         goal_state,
        #         dense_reward
        #         )
        
        # if args.save_pickles:
        #     pickle_run(subdirectory, solver, results)


if __name__ == '__main__':
    main()


    