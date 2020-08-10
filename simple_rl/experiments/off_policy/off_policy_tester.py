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
import matplotlib.cm as cm
from matplotlib.colors import Normalize
import pickle

from matplotlib.lines import Line2D
from scipy.ndimage.filters import uniform_filter1d
from tqdm import tqdm
import datetime

from simple_rl.agents.func_approx.ddpg.DDPGAgentClass import DDPGAgent
from simple_rl.agents.func_approx.ddpg.replay_buffer import ReplayBuffer
from simple_rl.agents.func_approx.ddpg.utils import create_log_dir
from simple_rl.agents.func_approx.dsc.utils import visualize_ddpg_replay_buffer
from simple_rl.mdp.MDPPlotterClass import rotate_file_name
from simple_rl.tasks import GymMDP
from simple_rl.tasks.ant_reacher.AntReacherMDPClass import AntReacherMDP
from simple_rl.tasks.off_policy_swimmer.SwimmerMDPClass import SwimmerMDP

plt.style.use('default')


class TrainOffPolicy:
    def __init__(self, mdp_name, render, dense_reward, seeds, device, algorithm, experiment_name, off_policy_targets, fixed_epsilon,
                 plot_vf, plot_buffers, save_pickles):
        self.mdp_name = mdp_name
        self.device = device
        self.algorithm = algorithm
        self.path = rotate_file_name(os.path.join("plots", "off_policy", experiment_name))
        self.off_policy_targets = off_policy_targets
        self.fixed_epsilon = fixed_epsilon
        self.plot_vf = plot_vf
        self.plot_buffers = plot_buffers
        self.save_pickles = save_pickles
        self.plot_freq = 500

        for subdirectory in ['pickles', 'replay_buffers', 'learning_curves', 'value_functions']:
            create_log_dir(os.path.join(self.path, subdirectory))

        # save command used to run experiments
        f = open(os.path.join(self.path, "run_command.txt"), 'w')
        f.write(' '.join(str(arg) for arg in sys.argv))
        f.close()

        if mdp_name == "point-reacher":
            from simple_rl.tasks.point_reacher.PointReacherMDPClass import PointReacherMDP
            self.on_policy_goal = (8, 8)
            self.tolerance = 0.5
            self.xlim = (-10, 10)
            self.ylim = (-10, 10)
            self.mdp = PointReacherMDP(seed=seeds[0],
                                       render=render,
                                       dense_reward=dense_reward,
                                       goal_pos=self.on_policy_goal,
                                       tolerance=self.tolerance)
        elif mdp_name == "ant-reacher":
            self.on_policy_goal = (1.5, 1.5)
            self.tolerance = 0.6
            self.xlim = (-5, 5)
            self.ylim = (-5, 5)
            self.mdp = AntReacherMDP(goal_pos=self.on_policy_goal,
                                     seed=seeds[0],
                                     render=render,
                                     tolerance=self.tolerance,
                                     dense_reward=dense_reward)
        elif mdp_name == "swimmer":
            self.on_policy_goal = (1.5, 1.5)
            self.tolerance = 0.6
            self.xlim = (-5, 5)
            self.ylim = (-5, 5)
            self.mdp = SwimmerMDP(goal_pos=self.on_policy_goal,
                                  seed=seeds[0],
                                  render=render,
                                  tolerance=self.tolerance,
                                  dense_reward=dense_reward)
        else:
            raise NotImplementedError

    def _make_solvers(self, num_seeds, name_suffix=""):
        if self.algorithm == 'DDPG':
            return [DDPGAgent(self.mdp.state_space_size(),
                              self.mdp.action_space_size(),
                              seed,
                              device=torch.device(self.device),
                              name=f"DDPG_{seed}_{name_suffix}",
                              exploration=None) for seed in num_seeds]

    def train_solvers(self, solvers, episodes, steps, goal_pos):
        per_episode_scores = []
        per_episode_durations = []
        self.mdp.goal_pos = goal_pos

        for solver in solvers:
            # variables to save reward
            solver_per_episode_scores = []
            per_episode_scores.append(solver_per_episode_scores)
            solver_per_episode_durations = []
            per_episode_durations.append(solver_per_episode_durations)

            for episode in range(episodes):
                if episode % self.plot_freq == 0:
                    if self.plot_vf:
                        self._plot_value_function(solver, goal_pos, episode)
                    if self.plot_buffers:
                        self._plot_buffer(solver.replay_buffer, goal_pos, episode)

                self.mdp.reset()
                state = deepcopy(self.mdp.init_state)
                score = 0
                step = 0
                for step in range(steps):
                    if self.fixed_epsilon and np.random.uniform() < 0.4:
                        action_dim = self.mdp.action_space_size()
                        action = np.random.uniform([-1] * action_dim, [1] * action_dim)
                    else:
                        action = solver.act(state.features())
                    
                    reward, next_state = self.mdp.execute_agent_action(action)
                    solver.step(state.features(), action, reward, next_state.features(), next_state.is_terminal())
                    solver.update_epsilon()
                    state = next_state
                    score += reward
                    if state.is_terminal():
                        break

                solver_per_episode_scores.append(score)
                solver_per_episode_durations.append(step)

                print(
                    f"\rSolver: {solver.name}\tEpisode {episode}\tDuration:{np.round(score, 2)}\tEpsilon: {round(solver.epsilon, 2)}")
            if self.plot_vf:
                self._plot_value_function(solver, goal_pos, episodes, success_rate=np.mean(solver_per_episode_scores))
            if self.plot_buffers:
                self._plot_buffer(solver.replay_buffer, goal_pos, episodes, success_rate=np.mean(solver_per_episode_scores))
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
            smooth_mean = uniform_filter1d(mean, 20, mode='nearest')
            smooth_std_err = uniform_filter1d(std_err, 20, mode='nearest')
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

    def _plot_features(self, ax, goal_pos=None):
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

    def _plot_buffer(self, replay_buffer, goal_pos, episode=None, success_rate=None):
        suffix = ''
        if episode is not None:
            suffix += f"_episode={episode}"
        if success_rate is not None:
            suffix += f"_success_rate={success_rate}"

        if type(replay_buffer) is ReplayBuffer:
            states = self._get_states(replay_buffer)
            file_name = f"{replay_buffer.name}_{goal_pos}{suffix}.png"
        else:
            states = np.array(replay_buffer)[:, 0]
            file_name = f"combined_replay_buffer_{goal_pos}{suffix}.png"

        positions = np.array([(state[0], state[1]) for state in states])
        # set up plots
        fig, ax = plt.subplots(figsize=(4, 4))
        ax.set_aspect = 'equal'
        ax.set_xlim(self.xlim[0], self.xlim[1])
        ax.set_ylim(self.ylim[0], self.ylim[1])

        # plot scatter
        ax.scatter(x=positions[:, 0], y=positions[:, 1], color='b', alpha=0.16)
        self._plot_features(ax, goal_pos)
        fig.suptitle(replay_buffer.name if type(replay_buffer) is ReplayBuffer else "combined_replay_buffer")

        # save file
        plt.savefig(os.path.join(self.path, 'replay_buffers', file_name))
        plt.close()

    def _train_off_policy_on_data(self, num_seeds, new_goal, experiences):
        self.mdp.goal_pos = new_goal
        off_policy_solvers = self._make_solvers(num_seeds, name_suffix=f"off_policy_targeting_{new_goal}")
        for solver in off_policy_solvers:
            for (state, action, _, next_state, _) in tqdm(experiences, desc=f"Pretraining {solver.name}"):
                is_terminal = self.mdp.is_goal_state(next_state)
                reward = 10 if is_terminal else -1
                solver.step(state, action, reward, next_state, is_terminal)
        return off_policy_solvers

    @staticmethod
    def _generate_combined_replay_buffers(solvers, num_training_examples=None):
        replay_buffers = [solver.replay_buffer for solver in solvers]
        shared_experiences = []
        for replay_buffer in replay_buffers:
            shared_experiences += replay_buffer.memory
        if num_training_examples is None:
            training_times = [len(replay_buffer.memory) for replay_buffer in replay_buffers]
            num_training_examples = int(sum(training_times) / len(training_times))
        combined_replay_buffer = random.sample(shared_experiences, num_training_examples)
        return combined_replay_buffer

    @staticmethod
    def _get_replay_buffer(filename):
        with open(filename, 'rb') as file:
            return pickle.load(file)

    def collect_on_policy_data(self, num_on_policy_seeds, episodes, steps):
        # Train on-policy solvers for n episodes and pickle the replay buffers
        name_suffix = f"original_on_policy_targeting_original_goal_{self.on_policy_goal}"
        on_policy_solvers = self._make_solvers(num_on_policy_seeds, name_suffix=name_suffix)
        scores = self.train_solvers(on_policy_solvers, episodes, steps, self.on_policy_goal)
        self.plot_learning_curves(scores=[scores], labels=["on policy reference"], episodes=episodes,
                                  file_name="original_goal_on_policy_learning_curves")

        # save all original replay buffers and also combine all replay buffers into 1 and save the sampled replay buffer
        combined_replay_buffer = self._generate_combined_replay_buffers(on_policy_solvers)
        self._plot_buffer(combined_replay_buffer, self.on_policy_goal)

        if self.save_pickles:
            for solver in on_policy_solvers:
                self._save_solver(solver)

            with open(os.path.join(self.path, 'pickles', "combined_replay_buffers.pkl"), "wb") as f:
                pickle.dump(combined_replay_buffer, f)

    def test_off_policy_training(self, pickled_buffers_dir, num_off_policy_seeds, episodes, steps):
        # collect off policy training data, pretrain policies, and then train normally (to compare to baseline)
        full_pickled_buffers_dir = os.path.join(pickled_buffers_dir, "pickles", "combined_replay_buffers.pkl")
        on_policy_training_data = self._get_replay_buffer(full_pickled_buffers_dir)

        for new_goal in self.off_policy_targets:
            initialized_off_policy_solvers = self._train_off_policy_on_data(num_off_policy_seeds, new_goal, on_policy_training_data)
            off_policy_episode_scores = self.train_solvers(initialized_off_policy_solvers, episodes, steps, new_goal)

            # train baseline on policy solver
            baseline_on_policy_solvers = self._make_solvers(num_off_policy_seeds, name_suffix=f"on_policy_baseline_targeting_{new_goal}")
            baseline_on_policy_episode_scores = self.train_solvers(baseline_on_policy_solvers, episodes, steps, new_goal)
            self.plot_learning_curves(
                [off_policy_episode_scores, baseline_on_policy_episode_scores],
                ["off policy", "on policy baseline"],
                episodes,
                file_name=f"off_policy_learning_curves_{new_goal}")

    def _save_solver(self, solver):
        with open(os.path.join(self.path, "pickles", f"{solver.name}_replay_buffer.pkl"), "wb") as f:
            pickle.dump(solver.replay_buffer, f)

    def _plot_value_function(self, solver, goal_pos, episode, success_rate=None):
        def get_value_function_values():
            CHUNK_SIZE = 250

            # Chunk up the inputs so as to conserve GPU memory
            num_chunks = int(np.ceil(states.shape[0] / CHUNK_SIZE))
            state_chunks = np.array_split(np.array(states), num_chunks, axis=0)
            action_chunks = np.array_split(np.array(actions), num_chunks, axis=0)
            values = np.zeros(len(states))
            current_idx = 0

            # get values for each state
            for state_chunk, action_chunk in zip(state_chunks, action_chunks):
                current_chunk_size = len(state_chunk)
                state_tensor = torch.from_numpy(state_chunk).float().to(solver.device)
                action_tensor = torch.from_numpy(action_chunk).float().to(solver.device)
                value_chunk = solver.get_qvalues(state_tensor, action_tensor).cpu().numpy().squeeze(1)
                values[current_idx:current_idx + current_chunk_size] = value_chunk
                current_idx += current_chunk_size

            return values

        if len(solver.replay_buffer.memory) == 0:
            return

        states = np.array([exp[0] for exp in solver.replay_buffer.memory])
        actions = np.array([exp[1] for exp in solver.replay_buffer.memory])
        qvalues = get_value_function_values()
        fig, ax = plt.subplots()
        ax.scatter(states[:, 0], states[:, 1], c=qvalues)
        ax.set_xlim(self.xlim)
        ax.set_ylim(self.ylim)
        self._plot_features(ax, goal_pos)
        suffix = f"_score={str(success_rate)}" if success_rate is not None else ''
        fig.suptitle(solver.name)
        fig.colorbar(cm.ScalarMappable(norm=Normalize(vmin=min(qvalues), vmax=max(qvalues)), cmap="inferno"), ax=ax, aspect=40)
        file_name = f"{solver.name}_value_function_episode={episode}{suffix}.png"
        plt.savefig(os.path.join(self.path, "value_functions", file_name))
        plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dense_reward", help="Whether to use dense/sparse rewards", action="store_true", default=False)
    parser.add_argument("--env", type=str, help="name of gym environment", default="point-env")
    parser.add_argument("--experiment_name", type=str, help="name of experiment",
                        default=datetime.datetime.now().strftime("%I:%M%p_%B_%d_%Y"))
    parser.add_argument("--render", help="render environment training", action="store_true", default=False)
    parser.add_argument("--episodes", type=int, help="number of training episodes")
    parser.add_argument("--steps", type=int, help="number of steps per episode")
    parser.add_argument("--device", type=str, help="cuda/cpu", default="cpu")
    parser.add_argument("--plot_buffers", help="plot replay buffers (combined buffer always plotted)", action="store_true", default=False)
    parser.add_argument("--plot_vf", help="plot value functions", action="store_true", default=False)
    parser.add_argument("--save_pickles", help="save pickled files", action="store_true", default=False)
    parser.add_argument("--num_seeds", type=int, help="number of seeds to run", default=5)
    parser.add_argument("--preload_buffer_experiment_name", type=str, help="path to solver", default="")
    parser.add_argument("--skip_off_policy", help="only train on policy DDPG (for verifying env works)", default=False, action="store_true")
    parser.add_argument("--fixed_epsilon", help="whether to use a fixed or decaying epsilon", default=False, action="store_true")
    # parser.add_argument("--")
    args = parser.parse_args()
    
    assert not (args.skip_off_policy and len(args.preload_buffer_experiment_name) > 0)

    # ant-reacher:
    task_off_policy_targets = [(1.5, -1.5), (-1.5, -1.5), (2.5, 2.5), (3, 3), (1.21, 1.74)]
    # task_off_policy_targets = [(1.9, 0.95), (2.03, 0.61), (0.61, 2.03), (0.95, 1.9), (3, 3)]


    # point reacher:
    # task_off_policy_targets = [(4.5, 4.5), (-4.5, 4.5), (4.5, -4.5), (-4.5, -4.5), (0, 4.5), (4.5, 0), (-4.5, 0), (0, -4.5),
    #                            (-9, -9), (-9, 9), (9, -9), (9, 9), (0, 9), (9, 0), (-9, 0), (0, -9)]
    train_off_policy = TrainOffPolicy(mdp_name=args.env,
                                      render=args.render,
                                      dense_reward=args.dense_reward,
                                      seeds=range(args.num_seeds),
                                      device=args.device,
                                      algorithm="DDPG",
                                      experiment_name=args.experiment_name,
                                      off_policy_targets=task_off_policy_targets,
                                      fixed_epsilon=args.fixed_epsilon,
                                      plot_vf=args.plot_vf,
                                      plot_buffers=args.plot_buffers,
                                      save_pickles=args.save_pickles)
    
    if args.preload_buffer_experiment_name == "":
        train_off_policy.collect_on_policy_data(range(args.num_seeds), args.episodes, args.steps)
        file_dir = train_off_policy.path
    else:
        file_dir = os.path.join("plots", "off_policy", args.preload_buffer_experiment_name)
    
    if not args.skip_off_policy:
        train_off_policy.test_off_policy_training(file_dir, range(args.num_seeds), args.episodes, args.steps)
    ipdb.set_trace()
    #
    # mdp = GymMDP("Swimmer-v2", True)
    # for episode in range(10):
    #     mdp.reset()
    #     for step in range(150):
    #         action = np.random.uniform([-1] * 2, [1] * 2)
    #         ipdb.set_trace()
    #         reward, next_state = mdp.execute_agent_action(action)
