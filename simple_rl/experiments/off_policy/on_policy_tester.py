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

################################################################################
# TRAIN OUR DDPG
################################################################################

def reward_metric(
    state: np.ndarray,
    goal_dimension: int,
    goal_threshold: float,
    goal_state: np.ndarray, # removed @HER
    dense_reward: bool
    ) -> Tuple[float, bool]:

    distance = np.linalg.norm(state[:goal_dimension] - goal_state)
    is_terminal = distance <= goal_threshold
    if dense_reward:
        reward = distance * -1
    else:
        reward = 10 if is_terminal else -1
    
    return reward, is_terminal

# For getting transitions @HER
def get_augmented_transition(
        state: np.ndarray,
        action: np.ndarray,
        next_state: np.ndarray,
        goal_dimension: int,
        goal_state: np.ndarray,
        goal_threshold: float,
        dense_reward: bool,
        ) -> Tuple[np.ndarray, np.ndarray, float, np.ndarray, bool]:

        # First, get our reward for the given conditioned goal
        reward, is_terminal = reward_metric(
            next_state, 
            goal_dimension, 
            goal_threshold, 
            goal_state,
            dense_reward
            )
        
        # Then, augment our state vectors with that goal
        state = np.concatenate((state, goal_state))
        next_state = np.concatenate((next_state, goal_state))

        # Finally, return the full transition tuple
        return (state, action, reward, next_state, is_terminal)  


def train_solver(
    solver: DDPGAgent,
    environment, # Gym Environment
    goal_dimension: int,
    goal_threshold: float,
    goal_state: np.ndarray,
    her_goals: np.ndarray, # Added @HER
    dense_reward: bool,
    fixed_epsilon: float,
    num_episodes: int, 
    num_steps: int, 
    render: bool,
    ) -> Tuple[DDPGAgent, np.ndarray]:
    
    results = np.zeros((num_episodes, 2))
    for episode in range(num_episodes):
        print(f'Executing episode {episode}/{num_episodes}')

        # We need to augment our state with our desired goal @HER
        state = environment.reset()

        for _ in range(num_steps):
            # (i) Render the current state
            if render:
                environment.render()

            # (ii) Get our next action
            if fixed_epsilon < np.random.random():
                action = environment.action_space.sample()
            else:
                # Act on the goal-conditioned state @HER
                augmented_state = np.concatenate((state, goal_state))
                action = solver.act(augmented_state) 
            
            # (iii) Take a step and get our rewards
            next_state, _, _, _ = environment.step(action)

            # Augment with (and record in results) our main goal @HER
            transition = get_augmented_transition(
                state, 
                action, 
                next_state, 
                goal_dimension, 
                goal_state, 
                goal_threshold, 
                dense_reward
                )
            
            # (iv) Train on and log our main goal transition
            solver.step(*transition)
            results[episode] += 1, transition[2]

            # Now, augment with and add to our sample buffer all our other HER goals @HER
            for her_goal in her_goals:
                her_transition = get_augmented_transition(
                    state, 
                    action, 
                    next_state, 
                    goal_dimension, 
                    her_goal, 
                    goal_threshold, 
                    dense_reward
                    )
                
                solver.step(*her_transition)

            # Iterate to the next state, not yet augmented @HER
            state = next_state
            if transition[4]:
                break
            
        print(f'Episode terminated in {results[episode,0]} steps with reward {results[episode,1]}')
    
    return solver, results

################################################################################
# OFF-POLICY PRE-TRAINING
################################################################################

def unpickle_combined_buffer(directory: Path, source_goals):
    combined_buffer = []
    for i in source_goals:
        if isinstance(i, int):
            buffer_path = directory / f'goal_{i}' / 'buffer.pkl'
        else:
            buffer_path = i
            
        if os.path.exists(buffer_path):
            buffer = pickle.load(open(buffer_path, 'rb'))
            combined_buffer += buffer
        else:
            print(f'Invalid goal {i}')
    
    return combined_buffer

def pretrain_solver(
    solver: DDPGAgent,
    directory: Path, 
    num_samples: int, 
    source_goals, 
    goal_dimension: int,
    goal_threshold: float,
    goal_state: np.ndarray,
    dense_reward: bool
    ) -> None:

    combined_buffer = unpickle_combined_buffer(directory, source_goals)
    num_samples = min(num_samples, len(combined_buffer))
    
    for n in num_samples:
        print(f'Pretraining on sample {n}/{num_samples}')
        transition = np.random.choice(combined_buffer, replace=False)
        solver.step(*transition)


################################################################################
# MAIN FUNCTION
################################################################################

# def rotate_file_name(file_path):
#     if os.path.isdir(file_path):
#         suffix = 1
#         file_path += "_" + str(suffix)
#         while os.path.isdir(file_path):
#             suffix += 1
#             file_path = file_path.rsplit("_", 1)[0] + "_" + str(suffix)
#         return file_path
#     else:
#         return file_path

# # save command used to run experiments
# f = open(os.path.join(self.path, "run_command.txt"), 'w')
# f.write(' '.join(str(arg) for arg in sys.argv))
# f.close()

def main():
    directory = Path.cwd() / f'{args.experiment_name}_{args.seed}'
    os.mkdir(directory)

    # TODO: Compat for non gym args
    environment = gym.make(args.environment)
    np.random.seed(args.seed)
    environment.seed(args.seed)

    # TODO: Right now, I assume all other goals are HER goals, which might not be right
    goals = [parse_goal(goal) for goal in args.add_goal]
    goal_states = [goal_state for (_, goal_state, _, _, _) in goals]

    for i, goal in enumerate(args.add_goal):
        # (i) Parse info about this goal
        (dense_reward, 
        goal_state, 
        pretrain, 
        num_samples, 
        source_goals
        ) = goal

        # TODO: Figure out what our HER goals are @HER
        her_goals = np.vstack(goal_states.remove(i))

        # (ii) Set up our DDPG
        solver = DDPGAgent(
            environment.observation_space.shape[0],
            environment.action_space.shape[0],
            args.seed,
            args.device
            )

        # *Optional* pretrain our DDPG
        if pretrain:
            pretrain_solver(
                solver, 
                directory, 
                num_samples, 
                source_goals,
                args.goal_dimension,
                args.goal_threshold,
                goal_state,
                dense_reward
                )
        
        # (iii) Run our training
        solver, results = train_solver(
            solver,
            environment,
            args.goal_dimension,
            args.goal_threshold,
            goal_state,
            her_goals,
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
                args.goal_threshold, 
                goal_state,
                dense_reward
                )
        
        if args.save_pickles:
            pickle_run(subdirectory, solver, results)


if __name__ == '__main__':
    main()


    