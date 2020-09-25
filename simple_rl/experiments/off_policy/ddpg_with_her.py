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
from simple_rl.agents.func_approx.ddpg.replay_buffer import ReplayBuffer
from simple_rl.tasks.ant_reacher.AntReacherMDPClass import AntReacherMDP
from typing import List, Tuple, Callable



################################################################################
# ARGUMENT PARSING
################################################################################

parser = argparse.ArgumentParser()
parser.add_argument('--experiment_name', type=str)
parser.add_argument('--seed', type=int)
parser.add_argument('--num_episodes', type=int)
parser.add_argument('--num_pretrain_episodes', type=int, default=0)
parser.add_argument('--num_steps', type=int)
parser.add_argument('--device', type=str, default='cpu')
parser.add_argument('--render', action='store_true', default=False)
parser.add_argument('--goal_threshold', type=float, default=0.6)
parser.add_argument('--goal_dimension', type=int, default=2)
parser.add_argument('--goal_state', type=float, nargs='+')
parser.add_argument('--dense_reward', action='store_true', default=False)
parser.add_argument('--goal_reward', type=float, default=0.)
parser.add_argument('--preload_solver_path', type=str, default=None)
parser.add_argument('--her_at_test_time', action='store_true', default=False)
args = parser.parse_args()

################################################################################
# PLOTTING AND PICKLING FUNCTIONS
################################################################################

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
    pickle.dump(solver, open(directory / f'ddpg_{solver.name}.pkl', 'wb'))

if __name__ == '__main__':
    directory = Path.cwd() / f'{args.experiment_name}_{args.seed}'
    os.mkdir(directory)

    # (i) Maybe pretrain our DDPG agent using HER
    mdp = AntReacherMDP(
        seed=args.seed, 
        render=args.render, 
        goal_reward=args.goal_reward
        )

    goal_state = np.array(args.goal_state)

    if args.preload_solver_path is None:
        solver = DDPGAgent(
                mdp.state_space_size() + args.goal_dimension,
                mdp.action_space_size(),
                args.seed,
                args.device,
                name="pretrain"
                )
        
        if args.num_pretrain_episodes > 0:
            pes_i, _ = her_train(
                solver,
                mdp, 
                args.num_pretrain_episodes, 
                args.num_steps, 
                goal_state=None, 
                sampling_strategy="diverse",
                dense_reward=args.dense_reward
                )
        
            plot_learning_curve(directory, 'pretrain with her', pes_i)
            pickle.dump(pes_i, open(directory / 'pretraining_scores.pkl', 'wb'))
            make_chunked_goal_conditioned_value_function_plot(directory, solver,goal_state, args.num_pretrain_episodes, args.seed, args.experiment_name)

            pickle_solver(directory, solver)
        else:
            raise ValueError("Need to specify number of pretrain episodes")
    else:
        solver = pickle.load(open(args.preload_solver_path, 'rb'))
    
    # At this point, we have a pre-trained goal conditioned solver
    test_solver = DDPGAgent(
                mdp.state_space_size() + args.goal_dimension,
                mdp.action_space_size(),
                args.seed,
                args.device,
                name="test"
            )
    ipdb.set_trace()
    
    make_chunked_goal_conditioned_value_function_plot(directory, test_solver, goal_state, -1, args.seed, args.experiment_name)
    test_solver.actor.load_state_dict(solver.actor.state_dict())
    test_solver.critic.load_state_dict(solver.critic.state_dict())
    test_solver.target_actor.load_state_dict(solver.target_actor.state_dict())
    test_solver.target_critic.load_state_dict(solver.target_critic.state_dict())  
    make_chunked_goal_conditioned_value_function_plot(directory, test_solver, goal_state, 0, args.seed, args.experiment_name)  
    
    # (ii) Test on a fixed-goal domain, maybe pretrained
    pes_ii, _ = her_train(
        test_solver,
        mdp,
        args.num_episodes,
        args.num_steps,
        goal_state = goal_state,
        sampling_strategy="test",
        dense_reward=args.dense_reward,
        use_her=args.her_at_test_time
        )
    
    pickle.dump(pes_ii, open(directory / 'test_time_scores.pkl', 'wb'))
    plot_learning_curve(directory, 'test time', pes_ii)
    make_chunked_goal_conditioned_value_function_plot(directory, test_solver, goal_state, args.num_episodes, args.seed, args.experiment_name)

    pickle_solver(directory, test_solver)

    ipdb.set_trace()


    