import argparse
import os
import random
from collections import deque
from copy import deepcopy
import numpy as np
import ipdb
import torch
import matplotlib.pyplot as plt
import pickle

from simple_rl.agents.func_approx.ddpg.DDPGAgentClass import DDPGAgent
from simple_rl.agents.func_approx.ddpg.utils import save_model, create_log_dir

plt.style.use('default')


class TrainOffPolicy:
    def __init__(self, mdp_name, render, dense_reward, seeds, device, algorithm):
        self.device = device
        self.algorithm = algorithm

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
                self._plot_buffer(solver.replay_buffer, goal_pos)
        return per_episode_scores

    @staticmethod
    def plot_learning_curves(scores, labels, episodes):
        def moving_average(arr, window_size):
            return np.convolve(arr, np.ones(window_size), 'same') / window_size

        print('*' * 80)
        print("Plotting learning curves...")
        print('*' * 80)

        fig, ax = plt.subplots()
        ax.set_xlim(0, episodes)
        for label, goal_scores in zip(labels, scores):
            mean = np.mean(goal_scores, axis=0)
            std_err = np.std(goal_scores, axis=0)
            smooth_mean = moving_average(mean, 10)
            smooth_std_err = moving_average(std_err, 10)
            ax.plot(range(episodes), smooth_mean, '-', label=label)
            ax.fill_between(range(episodes),
                            smooth_mean - smooth_std_err,
                            smooth_mean + smooth_std_err,
                            alpha=0.3)
        ax.legend()
        file_name = "learning_curves.png"
        plt.savefig(os.path.join("plots", "off_policy", file_name))
        plt.close()

    @staticmethod
    def _get_states(replay_buffer):
        return np.array(replay_buffer.memory)[:, 0]

    def _plot_buffer(self, replay_buffer, goal_pos):
        states = self._get_states(replay_buffer)
        positions = np.array([(state[0], state[1]) for state in states])

        # set up plots
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.set_aspect = 'equal'
        ax.set_xlim(-10, 10)
        ax.set_ylim(-10, 10)

        # plot scatter
        ax.scatter(x=positions[:, 0], y=positions[:, 1], color='b', alpha=0.2)

        # plot goal
        goal_state = plt.Circle(goal_pos, self.tolerance, alpha=1.0, color='g')
        ax.add_patch(goal_state)

        # save file
        file_name = f"{replay_buffer.name}.png"
        plt.savefig(os.path.join("plots", "off_policy", file_name))
        plt.close()

    def _train_off_policy_on_data(self, num_seeds, new_goal, experiences):
        self.mdp.goal_pos = new_goal
        off_policy_solvers = self._make_solvers(num_seeds)
        for solver in off_policy_solvers:
            for (state, action, _, next_state, _) in experiences:
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

        print('*' * 80)
        print("Saving combined replay buffer...")
        print('*' * 80)

        with open(os.path.join("plots", "off_policy", "combined_replay_buffers.pkl"), "wb") as f:
            pickle.dump(combined_replay_buffer, f)

    @staticmethod
    def _get_replay_buffer(filename):
        with open(filename, 'rb') as file:
            return pickle.load(file)

    def generate_on_policy_pickled_buffers(self, num_on_policy_seeds, episodes, steps, generate_plots):
        # Train on policy solvers for n episodes and pickle the replay buffers
        on_policy_solvers = self._make_solvers(num_on_policy_seeds)
        self.train_solvers(on_policy_solvers, episodes, steps, generate_plots, self.on_policy_goal)

        # save all original replay buffers and also combine all replay buffers into 1 and save the sampled replay buffer
        for solver in on_policy_solvers:
            save_model(solver, episodes, "plots", best=False, save_ddpg=False)
        self._save_combined_replay_buffers(on_policy_solvers)

    def test_off_policy_training(self, pickled_buffers_dir, num_off_policy_seeds, episodes, steps, generate_plots, new_goal):
        # collect off policy training data, pretrain policies, and then train normally (to compare to baseline)
        on_policy_training_data = self._get_replay_buffer(pickled_buffers_dir)  # type: []
        initialized_off_policy_solvers = self._train_off_policy_on_data(num_off_policy_seeds, new_goal, on_policy_training_data)
        off_policy_episode_scores = self.train_solvers(initialized_off_policy_solvers, episodes, steps, generate_plots, new_goal)

        # train baseline on policy solver
        baseline_on_policy_solvers = self._make_solvers(num_off_policy_seeds, off_policy=True)
        baseline_on_policy_episode_scores = self.train_solvers(baseline_on_policy_solvers, episodes, steps, generate_plots, new_goal)
        ipdb.set_trace()
        self.plot_learning_curves(
            [off_policy_episode_scores, baseline_on_policy_episode_scores],
            ["off policy", "on policy baseline"],
            episodes)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dense_reward", help="Whether to use dense/sparse rewards", action="store_true", default=False)
    parser.add_argument("--env", type=str, help="name of gym environment", default="point-env")
    parser.add_argument("--render", help="render environment training", action="store_true", default=False)
    parser.add_argument("--episodes", type=int, help="number of training episodes")
    parser.add_argument("--steps", type=int, help="number of steps per episode")
    parser.add_argument("--device", type=str, help="cuda/cpu", default="cpu")
    parser.add_argument("--generate_plots", help="save pickled files", action="store_true", default=False)
    parser.add_argument("--num_seeds", type=int, help="number of seeds to run", default=5)
    args = parser.parse_args()

    create_log_dir("plots/off_policy")
    train_off_policy = TrainOffPolicy(mdp_name=args.env,
                                      render=args.render,
                                      dense_reward=args.dense_reward,
                                      seeds=range(args.num_seeds),
                                      device=args.device,
                                      algorithm="DDPG")
    train_off_policy.generate_on_policy_pickled_buffers(range(args.num_seeds), args.episodes, args.steps, args.generate_plots)

    filename = os.path.join("plots", "off_policy", "combined_replay_buffers.pkl")
    train_off_policy.test_off_policy_training(filename, range(args.num_seeds), args.episodes, args.steps, args.generate_plots, (5, 8))
    ipdb.set_trace()
