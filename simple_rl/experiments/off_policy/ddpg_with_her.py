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
#parser.add_argument('--test_time_start_states_pickle', type=str, default=None)
#parser.add_argument('--test_time_goal_states_pickle', type=str, default=None)
parser.add_argument('--intervention', type=str, default=None)

args = parser.parse_args()

################################################################################
# PLOTTING AND PICKLING FUNCTIONS
################################################################################

# def moving_average(a, n):
#     ret = np.cumsum(a, dtype=float)
#     ret[n:] = ret[n:] - ret[:-n]
#     return ret[n-1:] / n

if __name__ == '__main__':
    ipdb.set_trace()

    directory = Path.cwd() / f'{args.experiment_name}_{args.seed}'
    os.mkdir(directory)

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
                torch.device(args.device),
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
        
            ipdb.set_trace()
            pickle.dump(pes_i, open(directory / 'pretraining_scores.pkl', 'wb'))
            pickle.dump(ped_i, open(directory / 'pretraining_durations.pkl', 'wb'))
            pickle.dump(pesuccesses_i, open(directory / 'pretraining_successes.pkl', 'wb'))
            pickle.dump(solver, open(directory / f'ddpg_{solver.name}_-1.pkl', 'wb'))
        else:
            raise ValueError("Need to specify number of pretrain episodes")
    else:
        solver = pickle.load(open(args.preload_solver_path, 'rb'))

        new_device = torch.device(args.device)
        solver.device = new_device
        solver.actor = solver.actor.to(new_device)
        solver.critic = solver.critic.to(new_device)
        solver.target_actor = solver.target_actor.to(new_device)
        solver.target_critic = solver.target_critic.to(new_device)

        # # Initialize actor target network
        # def copy_to_new_solver(new_params, old_params):
        #     for new_param, param in zip(new_params, old_params):
        #         target_param.data.copy_(param.data.to())

        # # Initialize critic target network
        # for target_param, param in zip(self.target_critic.parameters(), self.critic.parameters()):
        #     target_param.data.copy_(param.data)
        
    # Got rid of this cos it's all good hai
    # ipdb.set_trace()

    # if args.test_time_start_states_pickle is not None:
    #     test_time_start_states = pickle.load(open(args.test_time_start_states_pickle, 'rb'))
    # else:
    #     test_time_start_states = [np.array([0,0])]
    
    # if args.test_time_goal_states_pickle is not None:
    #     test_time_goal_states = pickle.load(open(args.test_time_goal_states_pickle, 'rb'))
    # else:
    #     test_time_goal_states = [np.array([2,2])]

    #for n, (start_state, goal_state) in enumerate(zip(test_time_start_states, test_time_goal_states)):    

    # Each time we do a unique intervention, so have to copy
    #test_time_solver = deepcopy(solver)
    #test_time_solver.name = "test_time"

    start_state = np.array([0,0])
    goal_state = np.array([3.5,3.5])

    # Just bad code right heeer
    use_her = True

    # Run our intervention here
    if args.intervention is not None:
        if args.intervention == "empty_buffer":
            solver.replay_buffer.clear()

            print("Fully emptied the replay buffer")

        elif args.intervention == "filter_buffer":
            old_buffer = deepcopy(solver.replay_buffer)
            solver.replay_buffer.clear()
            FILTER_THRESHOLD = 1
            for state, action, reward, next_state, terminal in old_buffer:
                # We are going to filter the replay buffer for states where
                # The conditioned goal state e.g. (state[-2],state[-1])
                # is within some threshold of our test-time goal state
                gc_state = state[-2:]
                if np.linalg.norm(gc_state - goal_state) <= FILTER_THRESHOLD:
                    solver.replay_buffer.add(state, action, reward, next_state, terminal)
                
            ipdb.set_trace()
            print(f"Filtered {solver.replay_buffer.num_exp} experiences out of {old_buffer.num_exp} with a threshold of {FILTER_THRESHOLD}")

        elif args.intervention == "train_subnetwork":
            # Only time we do this
            use_her = False

            # Here, we train a non-gc network on the fixed goal
            subnetwork = DDPGAgent(
                mdp.state_space_size(),
                mdp.action_space_size(),
                args.seed,
                torch.device(args.device),
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
            states_tensor = torch.from_numpy(states).float().to(torch.device(args.device))
            actions_tensor = torch.from_numpy(actions).float().to(torch.device(args.device))
            qvalues = solver.get_qvalues(states_tensor, actions_tensor).cpu().numpy().squeeze(1)

            # Now, do the thing for next_state,next_action pairs
            next_states = np.array([condition_on_goal(exp[3]) for exp in old_buffer.memory])
            next_actions = np.array([solver.act(next_state) for next_state in next_states])
            next_states_tensor = torch.from_numpy(next_states).float().to(torch.device(args.device))
            next_actions_tensor = torch.from_numpy(next_actions).float().to(torch.device(args.device))

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

                subnetwork.step(state, action, reward, next_state, terminal)

            # Here we drop the teacher
            solver = subnetwork

        else:
            raise NotImplementedError(f"{args.intervention} is not a valid intervention")
    # TODO: Change this to not use HER if our intervention
    # was train subnetwork -- in that case, we now have a 
    # fixed-goal policy

    # (ii) Test on a fixed-goal domain, maybe pretrained
    pes_ii, ped_ii, pesuccesses_ii, _ = her_train(
        solver,
        mdp,
        args.num_episodes,
        args.num_steps,
        start_state = start_state,
        goal_state = goal_state,
        sampling_strategy="test",
        dense_reward=args.dense_reward,
        use_her=use_her
        )

    pickle.dump(pes_ii, open(directory / f'test_time_scores_{n}.pkl', 'wb'))
    pickle.dump(ped_ii, open(directory / f'test_time_durations_{n}.pkl', 'wb'))
    pickle.dump(pesuccesses_ii, open(directory / f'test_time_successes_{n}.pkl', 'wb'))
    pickle.dump(solver, open(directory / f'ddpg_{solver.name}_{n}.pkl', 'wb'))

    ipdb.set_trace()


    