import argparse
import os
from collections import deque
from copy import deepcopy
import numpy as np
import ipdb
import torch
import matplotlib.pyplot as plt

from simple_rl.agents.func_approx.ddpg.DDPGAgentClass import DDPGAgent
from simple_rl.agents.func_approx.ddpg.utils import save_model


class OffPolicyExperiment:
    def __init__(self, mdp_name, render, dense_reward, seeds, device, algorithm):
        if mdp_name == "point-reacher":
            from simple_rl.tasks.point_reacher.PointReacherMDPClass import PointReacherMDP
            goal_pos = (8, 8)
            self.mdp = PointReacherMDP(seed=seeds[0], render=render, dense_reward=dense_reward, goal_pos=goal_pos)

        if algorithm == 'DDPG':
            self.solvers = [DDPGAgent(self.mdp.state_space_size(),
                                      self.mdp.action_space_size(),
                                      seed,
                                      device=torch.device(device),
                                      name=f"DDPG_{seed}",
                                      exploration=None) for seed in seeds]

        self.secondary_goal_states = []

    def train_solvers(self, episodes, steps, generate_plots):
        per_episode_scores = []
        per_episode_durations = []

        for solver in self.solvers:
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
                score = 0.
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

                print(f"\rEpisode {episode}\tAverage Duration:{np.round(np.mean(last_50_durations), 2)}\tEpsilon: {round(solver.epsilon, 2)}")
            if generate_plots:
                save_model(solver, episodes, "plots", best=False, save_ddpg=False)

        fig, axs = plt.subplots()
        mean = np.mean(per_episode_scores, axis=0)
        std_err = np.std(per_episode_scores, axis=0)

        # plot learning curves
        print('*' * 80)
        print("Plotting learning curves...")
        print('*' * 80)
        fig, ax = plt.subplots()
        ax.plot(range(episodes), mean, '-')
        ax.fill_between(range(episodes), np.maximum(mean - std_err, 0), np.minimum(mean + std_err, 1), alpha=0.2)
        ax.set_xlim(0, episodes)

        file_name = "learning_curves.png"
        plt.savefig(os.path.join("plots", "saved_runs", file_name))
        plt.close()
        ipdb.set_trace()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dense_reward", help="Whether to use dense/sparse rewards", action="store_true", default=False)
    parser.add_argument("--env", type=str, help="name of gym environment", default="point-env")
    parser.add_argument("--render", help="render environment training", action="store_true", default=False)
    parser.add_argument("--episodes", type=int, help="number of training episodes")
    parser.add_argument("--steps", type=int, help="number of steps per episode")
    parser.add_argument("--device", type=str, help="cuda/cpu", default="cpu")
    parser.add_argument("--generate_plots", help="save pickled files", action="store_true", default=False)
    args = parser.parse_args()
    # make MDP with five seeds
    # train MDP for X episodes and pickle the training data
    # select new nearby goal states
    # pretrain using pickled data on those goal states
    off_policy_experiment = OffPolicyExperiment(mdp_name=args.env,
                                                render=args.render,
                                                dense_reward=args.dense_reward,
                                                seeds=range(5),
                                                device=args.device,
                                                algorithm="DDPG")
    off_policy_experiment.train_solvers(args.episodes, args.steps, args.generate_plots)
