# Core imports
import numpy as np
import gym
import torch
import ipdb
import random

# I/O imports
import os
import argparse 
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import pickle
from scipy.ndimage.filters import uniform_filter1d
from copy import deepcopy
import torch

# Other imports
from simple_rl.agents.func_approx.ddpg.DDPGAgentClass import DDPGAgent, her_train, train
from simple_rl.agents.func_approx.ddpg.replay_buffer import ReplayBuffer
from typing import List, Tuple, Callable



################################################################################
# ARGUMENT PARSING
################################################################################

parser = argparse.ArgumentParser()
parser.add_argument('--experiment_name', type=str)
parser.add_argument("--env", type=str, help="MDP environment", required=True)
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
parser.add_argument('--test_time_start_states_pickle', type=str, default=None)
parser.add_argument('--test_time_goal_states_pickle', type=str, default=None)
parser.add_argument('--intervention', type=str, default=None)

args = parser.parse_args()

################################################################################
# PLOTTING AND PICKLING FUNCTIONS
################################################################################

def visualize_ddpg_replay_buffer(solver, solver_number, directory):
    print('*' * 80)
    print("Plotting value function...")
    print('*' * 80)
    states = np.array([exp[0] for exp in solver.replay_buffer.memory])
    actions = np.array([exp[1] for exp in solver.replay_buffer.memory])
    states_tensor = torch.from_numpy(states).float().to(solver.device)
    actions_tensor = torch.from_numpy(actions).float().to(solver.device)
    qvalues = solver.get_qvalues(states_tensor, actions_tensor).cpu().numpy().squeeze(1)
    plt.scatter(states[:, 0], states[:, 1], c=qvalues)

    plt.xlim(-10, 10)
    plt.ylim(-10, 10)

    plt.xlabel("x position")
    plt.ylabel("y position")

    cbar = plt.colorbar()
    cbar.set_label("predicted value")

    plt.title(f"DDPG {solver_number}, unconditioned")

    plt.savefig(directory / f'ddpg_value_function_{solver_number}.png')
    plt.close()

def visualize_gc_ddpg_replay_buffer(solver, goal_state, solver_number, directory):
    print('*' * 80)
    print("Plotting value function...")
    print('*' * 80)
    def condition_on_goal(state):
        gc_state = state
        gc_state[-2] = goal_state[0]
        gc_state[-1] = goal_state[1]

        return gc_state

    states = np.array([condition_on_goal(exp[0]) for exp in solver.replay_buffer.memory])
    actions = np.array([exp[1] for exp in solver.replay_buffer.memory])

    states_tensor = torch.from_numpy(states).float().to(solver.device)
    actions_tensor = torch.from_numpy(actions).float().to(solver.device)
    qvalues = solver.get_qvalues(states_tensor, actions_tensor).cpu().numpy().squeeze(1)
    plt.scatter(states[:, 0], states[:, 1], c=qvalues)

    plt.xlim(-10, 10)
    plt.ylim(-10, 10)

    plt.xlabel("x position")
    plt.ylabel("y position")

    cbar = plt.colorbar()
    cbar.set_label("predicted value")

    plt.title(f"DDPG {solver_number}, conditioned on ({goal_state[0]},{goal_state[1]})")

    plt.savefig(directory / f'ddpg_gc_value_function_{solver_number}.png')
    plt.close()

# def moving_average(a, n):
#     ret = np.cumsum(a, dtype=float)
#     ret[n:] = ret[n:] - ret[:-n]
#     return ret[n-1:] / n

if __name__ == '__main__':
    directory = Path.cwd() / f'{args.experiment_name}_{args.seed}'
    os.mkdir(directory)

    #ipdb.set_trace()

    random.seed(args.seed)
    np.random.seed(args.seed)

    pickle.dump(args, open(directory / 'args.pkl', 'wb'))

    if args.env == "d4rl-ant-maze":
        from simple_rl.tasks.d4rl_ant_maze.D4RLAntMazeMDPClass import D4RLAntMazeMDP
        mdp = D4RLAntMazeMDP(maze_size="medium", seed=args.seed, render=args.render)
    elif args.env == "d4rl-medium-point-maze":
        from simple_rl.tasks.d4rl_point_maze.D4RLPointMazeMDPClass import D4RLPointMazeMDP
        mdp = D4RLPointMazeMDP(seed=args.seed,
                                       render=args.render,
                                       use_hard_coded_events=False,
                                       difficulty="medium",
                                       goal_directed=True)
    elif args.env == "d4rl-hard-point-maze":
        from simple_rl.tasks.d4rl_point_maze.D4RLPointMazeMDPClass import D4RLPointMazeMDP
        mdp = D4RLPointMazeMDP(seed=args.seed,
                                       render=args.render,
                                       use_hard_coded_events=False,
                                       difficulty="hard",
                                       goal_directed=True)
    elif args.env == "ant-reacher":
        from simple_rl.tasks.ant_reacher.AntReacherMDPClass import AntReacherMDP
        mdp = AntReacherMDP(seed=args.seed,
                            goal_reward=args.goal_reward)
    else:
        raise NotImplementedError(args.env)

    # (i) Maybe pretrain our DDPG agent using HER

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
            pes_i, ped_i, pesuccesses_i, _ = her_train(
                solver,
                mdp, 
                args.num_pretrain_episodes, 
                args.num_steps, 
                goal_state=None, 
                sampling_strategy="diverse",
                dense_reward=args.dense_reward
                )
        
            pickle.dump(pes_i, open(directory / 'pretraining_scores.pkl', 'wb'))
            pickle.dump(ped_i, open(directory / 'pretraining_durations.pkl', 'wb'))
            pickle.dump(pesuccesses_i, open(directory / 'pretraining_successes.pkl', 'wb'))
            pickle.dump(solver, open(directory / f'ddpg_{solver.name}_-1.pkl', 'wb'))
        else:
            raise ValueError("Need to specify number of pretrain episodes")
    else:
        solver = pickle.load(open(args.preload_solver_path, 'rb'))

    if args.test_time_start_states_pickle is not None:
        test_time_start_states = pickle.load(open(args.test_time_start_states_pickle, 'rb'))
    else:
        test_time_start_states = [np.array([0,0])]
    
    if args.test_time_goal_states_pickle is not None:
        test_time_goal_states = pickle.load(open(args.test_time_goal_states_pickle, 'rb'))
    else:
        test_time_goal_states = [np.array([1.5,1.5])]

    #ipdb.set_trace()

    for n, (start_state, goal_state) in enumerate(zip(test_time_start_states, test_time_goal_states)):    
        # Each time we do a unique intervention, so have to copy
        test_time_solver = deepcopy(solver)
        test_time_solver.name = "test_time"

        # Run our intervention here
        if args.intervention is not None:
            if args.intervention == "empty_buffer":
                test_time_solver.replay_buffer.clear()

                print("Fully emptied the replay buffer")

            elif args.intervention == "filter_buffer":
                old_buffer = solver.replay_buffer
                test_time_solver.replay_buffer.clear()
                for i in range(old_buffer.num_exp):
                    state, action, reward, next_state, terminal = old_buffer[i]
                    FILTER_THRESHOLD = 1

                    # We are going to filter the replay buffer for states where
                    # The conditioned goal state e.g. (state[-2],state[-1])
                    # is within some threshold of our test-time goal state
                    gc_state = state[-2:]
                    if np.linalg.norm(gc_state - goal_state) <= FILTER_THRESHOLD:
                        test_time_solver.replay_buffer.add(state, action, reward, next_state, terminal)
                    
                print(f"Filtered {test_time_solver.replay_buffer.num_exp} experiences out of {old_buffer.num_exp} with a threshold of {FILTER_THRESHOLD}")

            elif args.intervention == "train_subnetwork":
                # Here, we train a non-gc network on the fixed goal
                test_time_solver = DDPGAgent(
                    mdp.state_space_size(),
                    mdp.action_space_size(),
                    args.seed,
                    args.device,
                    name="test_time"
                    )
                
                def condition_on_goal(state):
                    gc_state = state
                    gc_state[-2:] = goal_state

                    return gc_state

                old_buffer = solver.replay_buffer
                # Even though we will strip the goal off this state,
                # We need it to get the goal-conditioned predicted rewards
                states = np.array([condition_on_goal(exp[0]) for exp in old_buffer.memory])
                actions = np.array([exp[1] for exp in old_buffer.memory])

                # Here, we use the q value of the old solver as our ground truth
                states_tensor = torch.from_numpy(states).float().to(args.device)
                actions_tensor = torch.from_numpy(actions).float().to(args.device)
                qvalues = solver.get_qvalues(states_tensor, actions_tensor).cpu().numpy().squeeze(1)

                # Now, do the thing for next_state,next_action pairs
                next_states = np.array([condition_on_goal(exp[3]) for exp in old_buffer.memory])
                next_actions = np.array([solver.act(next_state) for next_state in next_states])
                next_states_tensor = torch.from_numpy(next_states).float().to(args.device)
                next_actions_tensor = torch.from_numpy(next_actions).float().to(args.device)

                next_qvalues = solver.get_qvalues(next_states_tensor, next_actions_tensor).cpu().numpy().squeeze(1)
                discounted_next_qvalues = solver.gamma * next_qvalues

                rewards = qvalues - discounted_next_qvalues

                def is_terminal(state):
                    return np.linalg.norm(state[:2] - goal_state) <= args.goal_threshold
                terminals = [is_terminal(next_state) for next_state in next_states]

                # We can then train on these q values
                for state, action, reward, next_state, terminal in zip(states, actions, rewards, next_states, terminals):
                    # !!! important, our new solver is NOT goal conditioned
                    state = state[:-2]
                    next_state = next_state[:-2]

                    reward = float(reward)
                    terminal = bool(terminal)

                    ipdb.set_trace()
                    test_time_solver.step(state, action, reward, next_state, terminal)


            else:
                raise NotImplementedError(f"{args.intervention} is not a valid intervention")
        ipdb.set_trace()
        # TODO: Change this to not use HER if our intervention
        # was train subnetwork -- in that case, we now have a 
        # fixed-goal policy

        # (ii) Test on a fixed-goal domain, maybe pretrained
        pes_ii, ped_ii, pesuccesses_ii, _ = her_train(
            test_time_solver,
            mdp,
            args.num_episodes,
            args.num_steps,
            start_state = start_state,
            goal_state = goal_state,
            sampling_strategy="test",
            dense_reward=args.dense_reward,
            use_her=args.her_at_test_time
            )

        pickle.dump(pes_ii, open(directory / f'test_time_scores_{n}.pkl', 'wb'))
        pickle.dump(ped_ii, open(directory / f'test_time_durations_{n}.pkl', 'wb'))
        pickle.dump(pesuccesses_ii, open(directory / f'test_time_successes_{n}.pkl', 'wb'))
        pickle.dump(test_time_solver, open(directory / f'ddpg_{test_time_solver.name}_{n}.pkl', 'wb'))

    ipdb.set_trace()


    