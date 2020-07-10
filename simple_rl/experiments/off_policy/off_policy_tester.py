import argparse
import os
import random
import sys
from collections import deque
from copy import deepcopy
import numpy as np
import ipdb
import torch
import matplotlib.pyplot as plt
import pickle

from matplotlib.lines import Line2D
from scipy.ndimage.filters import uniform_filter1d
from tqdm import tqdm
import datetime

from simple_rl.agents.func_approx.ddpg.DDPGAgentClass import DDPGAgent
from simple_rl.agents.func_approx.ddpg.replay_buffer import ReplayBuffer
from simple_rl.agents.func_approx.ddpg.utils import create_log_dir
from simple_rl.mdp.MDPPlotterClass import rotate_file_name

plt.style.use('default')


class TrainOffPolicy:
    def __init__(self, mdp_name, render, dense_reward, seeds, device, algorithm, experiment_name, off_policy_targets):
        self.device = device
        self.algorithm = algorithm
        self.path = rotate_file_name(os.path.join("plots", "off_policy", experiment_name))
        self.off_policy_targets = off_policy_targets
        for subdirectory in ['pickles', 'replay_buffers', 'learning_curves']:
            create_log_dir(os.path.join(self.path, subdirectory))

        # save command used to run experiments
        f = open(os.path.join(self.path, "run_command.txt"), 'w')
        f.write(' '.join(str(arg) for arg in sys.argv))
        f.close()

        if mdp_name == "point-reacher":
            from simple_rl.tasks.point_reacher.PointReacherMDPClass import PointReacherMDP
            self.on_policy_goal = (8, 8)
            self.tolerance = 0.5
            self.mdp = PointReacherMDP(seed=seeds[0],
                                       render=render,
                                       dense_reward=dense_reward,
                                       goal_pos=self.on_policy_goal,
                                       tolerance=self.tolerance)

    def _make_solvers(self, num_seeds, off_policy=False):
        if self.algorithm == 'DDPG':
            return [DDPGAgent(self.mdp.state_space_size(),
                              self.mdp.action_space_size(),
                              seed,
                              device=torch.device(self.device),
                              name=f"DDPG_{seed}_off_policy={off_policy}",
                              exploration=None) for seed in num_seeds]

    def train_solvers(self, solvers, episodes, steps, generate_plots, goal_pos):
        per_episode_scores = []
        per_episode_durations = []
        self.mdp.goal_pos = goal_pos

        for solver in solvers:
            # variables to save reward
            solver_per_episode_scores = []
            per_episode_scores.append(solver_per_episode_scores)
            solver_per_episode_durations = []
            per_episode_durations.append(solver_per_episode_durations)
            last_50_scores = deque(maxlen=50)
            last_50_durations = deque(maxlen=50)

            for episode in range(episodes):
                self.mdp.reset()
                state = deepcopy(self.mdp.init_state)
                score = 0
                step = 0
                for step in range(steps):
                    action = solver.act(state.features())
                    reward, next_state = self.mdp.execute_agent_action(action)
                    solver.step(state.features(), action, reward, next_state.features(), next_state.is_terminal())
                    solver.update_epsilon()
                    state = next_state
                    score += reward
                    if state.is_terminal():
                        break

                last_50_scores.append(score)
                solver_per_episode_scores.append(score)
                last_50_durations.append(step)
                solver_per_episode_durations.append(step)

                print(f"\rSolver: {solver.name}\tEpisode {episode}\tAverage Duration:{np.round(np.mean(last_50_durations), 2)}\tEpsilon: {round(solver.epsilon, 2)}")
            if generate_plots:
                self._plot_buffer(solver.replay_buffer, goal_pos=goal_pos)
        return per_episode_scores

    def plot_learning_curves(self, scores, labels, episodes, file_name="learning_curves"):
        print('*' * 80)
        print("Plotting learning curves...")
        print('*' * 80)

        fig, ax = plt.subplots()
        ax.set_xlim(0, episodes)
        for label, goal_scores in zip(labels, scores):
            mean = np.mean(goal_scores, axis=0, dtype=float)
            std_err = np.std(goal_scores, axis=0, dtype=float)
            smooth_mean = uniform_filter1d(mean, 10, mode='nearest')
            smooth_std_err = uniform_filter1d(std_err, 10, mode='nearest')
            ax.plot(range(episodes), smooth_mean, '-', label=label)
            ax.fill_between(range(episodes),
                            smooth_mean - smooth_std_err,
                            smooth_mean + smooth_std_err,
                            alpha=0.3)
        ax.legend()
        plt.savefig(os.path.join(self.path, 'learning_curves', file_name + '.png'))
        plt.close()

    @staticmethod
    def _get_states(replay_buffer):
        return np.array(replay_buffer.memory)[:, 0]

    def _plot_buffer(self, replay_buffer, goal_pos=None):
        if type(replay_buffer) is ReplayBuffer:
            states = self._get_states(replay_buffer)
            file_name = f"{replay_buffer.name}_{goal_pos}.png"
        else:
            states = np.array(replay_buffer)[:, 0]
            file_name = f"combined_replay_buffer_{goal_pos}.png"

        positions = np.array([(state[0], state[1]) for state in states])
        # set up plots
        fig, ax = plt.subplots(figsize=(4, 4))
        ax.set_aspect = 'equal'
        ax.set_xlim(-10, 10)
        ax.set_ylim(-10, 10)

        # plot scatter
        ax.scatter(x=positions[:, 0], y=positions[:, 1], color='b', alpha=0.16)

        # plot goal
        handles = []
        on_policy_goal_marker = Line2D([], [], marker="o", linestyle="none", color='k', markersize=12, label="original on-policy goal")
        handles.append(on_policy_goal_marker)
        on_policy_goal = plt.Circle(self.on_policy_goal, self.tolerance, alpha=0.7, color='k')
        ax.add_patch(on_policy_goal)

        if goal_pos is not None:
            goal_state = plt.Circle(goal_pos, self.tolerance, alpha=0.7, color='gold')
            ax.add_patch(goal_state)
        else:
            for goal_pos in self.off_policy_targets:
                goal_state = plt.Circle(goal_pos, self.tolerance, alpha=0.7, color='gold')
                ax.add_patch(goal_state)

        curr_goal_marker = Line2D([], [], marker="o", linestyle="none", color='gold', markersize=12, label="current goal")
        handles.append(curr_goal_marker)
        ax.legend(handles=handles)

        # save file
        plt.savefig(os.path.join(self.path, 'replay_buffers', file_name))
        plt.close()

    def _train_off_policy_on_data(self, num_seeds, new_goal, experiences):
        self.mdp.goal_pos = new_goal
        off_policy_solvers = self._make_solvers(num_seeds, off_policy=True)
        for solver in off_policy_solvers:
            for (state, action, _, next_state, _) in tqdm(experiences, desc=f"Pretraining {solver.name}"):
                is_terminal = self.mdp.is_goal_state(next_state)
                reward = 1 if is_terminal else -1
                solver.step(state, action, reward, next_state, is_terminal)
        return off_policy_solvers

    def _save_combined_replay_buffers(self, solvers, num_training_examples=None):
        replay_buffers = [solver.replay_buffer for solver in solvers]
        shared_experiences = []
        for replay_buffer in replay_buffers:
            shared_experiences += replay_buffer.memory
        if num_training_examples is None:
            training_times = [len(replay_buffer.memory) for replay_buffer in replay_buffers]
            num_training_examples = int(sum(training_times) / len(training_times))
        combined_replay_buffer = random.sample(shared_experiences, num_training_examples)
        self._plot_buffer(combined_replay_buffer)

        print('*' * 80)
        print("Saving combined replay buffer...")
        print('*' * 80)

        with open(os.path.join(self.path, 'pickles', "combined_replay_buffers.pkl"), "wb") as f:
            pickle.dump(combined_replay_buffer, f)

    @staticmethod
    def _get_replay_buffer(filename):
        with open(filename, 'rb') as file:
            return pickle.load(file)

    def generate_on_policy_pickled_buffers(self, num_on_policy_seeds, episodes, steps, generate_plots):
        # Train on-policy solvers for n episodes and pickle the replay buffers
        on_policy_solvers = self._make_solvers(num_on_policy_seeds)
        scores = self.train_solvers(on_policy_solvers, episodes, steps, generate_plots, self.on_policy_goal)
        self.plot_learning_curves(scores=[scores], labels=["on policy reference"], episodes=episodes,
                                  file_name="original_goal_on_policy_learning_curves")

        # save all original replay buffers and also combine all replay buffers into 1 and save the sampled replay buffer
        for solver in on_policy_solvers:
            self._save_solver(solver)
        self._save_combined_replay_buffers(on_policy_solvers)

    def test_off_policy_training(self, pickled_buffers_dir, num_off_policy_seeds, episodes, steps, generate_plots):
        # collect off policy training data, pretrain policies, and then train normally (to compare to baseline)

        on_policy_training_data = self._get_replay_buffer(
            os.path.join("plots", pickled_buffers_dir, "pickles", "combined_replay_buffers.pkl"))

        for new_goal in self.off_policy_targets:
            initialized_off_policy_solvers = self._train_off_policy_on_data(num_off_policy_seeds, new_goal, on_policy_training_data)
            off_policy_episode_scores = self.train_solvers(initialized_off_policy_solvers, episodes, steps, generate_plots, new_goal)

            # train baseline on policy solver
            baseline_on_policy_solvers = self._make_solvers(num_off_policy_seeds)
            baseline_on_policy_episode_scores = self.train_solvers(baseline_on_policy_solvers, episodes, steps, generate_plots, new_goal)
            self.plot_learning_curves(
                [off_policy_episode_scores, baseline_on_policy_episode_scores],
                ["off policy", "on policy baseline"],
                episodes,
                file_name=f"off_policy_learning_curves_{new_goal}")

    def _save_solver(self, solver):
        with open(os.path.join(self.path, "pickles", f"{solver.name}_replay_buffer.pkl"), "wb") as f:
            pickle.dump(solver.replay_buffer, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dense_reward", help="Whether to use dense/sparse rewards", action="store_true", default=False)
    parser.add_argument("--env", type=str, help="name of gym environment", default="point-env")
    parser.add_argument("--experiment_name", type=str, help="name of experiment", default=datetime.datetime.now().strftime("%I:%M%p_%B_%d_%Y"))
    parser.add_argument("--render", help="render environment training", action="store_true", default=False)
    parser.add_argument("--episodes", type=int, help="number of training episodes")
    parser.add_argument("--steps", type=int, help="number of steps per episode")
    parser.add_argument("--device", type=str, help="cuda/cpu", default="cpu")
    parser.add_argument("--plot_replay_buffers", help="save pickled files", action="store_true", default=False)
    parser.add_argument("--num_seeds", type=int, help="number of seeds to run", default=5)
    parser.add_argument("--preload_buffer_experiment_name", type=str, help="path to solver", default="")
    args = parser.parse_args()

    train_off_policy = TrainOffPolicy(mdp_name=args.env,
                                      render=args.render,
                                      dense_reward=args.dense_reward,
                                      seeds=range(args.num_seeds),
                                      device=args.device,
                                      algorithm="DDPG",
                                      experiment_name=args.experiment_name,
                                      off_policy_targets=[(5, 8), (8, 5), (5, 5), (9.5, 9.5), (0, 8), (8, 0)])
    if args.preload_buffer_experiment_name == "":
        train_off_policy.generate_on_policy_pickled_buffers(range(args.num_seeds), args.episodes, args.steps, args.plot_replay_buffers)
        file_dir = train_off_policy.path
    else:
        file_dir = args.preload_buffer_experiment_name

    train_off_policy.test_off_policy_training(file_dir, range(args.num_seeds), args.episodes, args.steps, args.plot_replay_buffers)
    ipdb.set_trace()
