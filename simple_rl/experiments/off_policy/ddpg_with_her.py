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
parser.add_argument('--test_time_start_states_pickle', type=str)
parser.add_argument('--test_time_goal_states_pickle', type=str)

args = parser.parse_args()

################################################################################
# PLOTTING AND PICKLING FUNCTIONS
################################################################################

# def reset(self, episode, start_state=None):
#     """ Reset the MDP to either the default start state or to the specified one. """
#     print("*" * 80)
#     if start_state is None:
#         self.mdp.reset()
#         print(f"[DeepSkillGraphAgentClass] Episode {episode}: Resetting MDP to start state")
#     else:
#         start_position = start_state.position if isinstance(start_state, State) else start_state[:2]
#         print(f"[DeepSkillGraphAgentClass] Episode {episode}: Resetting MDP to manual state {start_position}")
#         self.mdp.set_xy(start_position)
#     print("*" * 80)

if __name__ == '__main__':
    directory = Path.cwd() / f'{args.experiment_name}_{args.seed}'
    os.mkdir(directory)

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


    test_time_start_states = pickle.load(open(args.test_time_start_states_pickle, 'rb'))
    test_time_goal_states = pickle.load(open(args.test_time_goal_states_pickle, 'rb'))

    for n, (start_state, goal_state) in enumerate(zip(test_time_start_states, test_time_goal_states)):    
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
            use_her=args.her_at_test_time
            )

        pickle.dump(pes_ii, open(directory / f'test_time_scores_{n}.pkl', 'wb'))
        pickle.dump(ped_ii, open(directory / f'test_time_durations_{n}.pkl', 'wb'))
        pickle.dump(pesuccesses_ii, open(directory / f'test_time_successes_{n}.pkl', 'wb'))
        pickle.dump(solver, open(directory / f'ddpg_{solver.name}_{n}.pkl', 'wb'))

    ipdb.set_trace()


    