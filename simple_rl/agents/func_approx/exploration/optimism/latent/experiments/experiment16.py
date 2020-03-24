from collections import deque
from copy import deepcopy
import numpy as np
import argparse
import torch
import matplotlib.pyplot as plt

from simple_rl.agents.func_approx.dqn.DQNAgentClass import DQNAgent
from simple_rl.tasks.gym.GymMDPClass import GymMDP
from simple_rl.agents.func_approx.exploration.optimism.latent.utils import create_log_dir, save_scores


class Experiment16:
    """
    Visual Mountain car -- DQN on pixel observations, OC-SVM on x, xdot features.
    Goal is to understand if DQN can learn (and how long it takes to learn) when getting images of mountain car.
    """
    def __init__(self, seed, *, episodes, experiment_name, tensor_log, device):
        self.mdp = GymMDP(env_name="MountainCar-v0", pixel_observation=True, seed=seed, control_problem=True, num_stack=4)
        state_dim = self.mdp.state_dim
        print(state_dim)
        self.agent = DQNAgent(state_size=state_dim, action_size=len(self.mdp.actions),
                              trained_options=[], seed=seed, device=device, name="GlobalDQN", pixel_observation=True,
                              lr=1e-3, use_double_dqn=False, exploration_method="oc-svm", tensor_log=tensor_log,
                              experiment_name=experiment_name, novelty_during_regression=True, normalize_states=False)
        self.exploration_method = "oc-svm"
        self.episodes = episodes
        self.seed = seed
        self.experiment_name = experiment_name

    def run_experiment(self):
        training_scores, training_durations = self.train_dqn_agent(self.agent, self.mdp, self.episodes, 200)
        return training_scores, training_durations

    def make_value_plot(self, agent, episode):
        states = np.array([transition.state for transition in agent.replay_buffer])
        positions = np.array([transition.position for transition in agent.replay_buffer])

        states_tensor = torch.from_numpy(states).float().to(agent.device)

        qvalues = agent.get_batched_qvalues(states_tensor, None).cpu().numpy()

        plt.figure(figsize=(14, 10))
        for action in self.agent.actions:
            plt.subplot(1, len(self.agent.actions), action + 1)
            plt.scatter(positions[:, 0], positions[:, 1], c=qvalues[:, action])#, norm=matplotlib.colors.LogNorm())
            plt.colorbar()

        plt.suptitle("Q-functions after episode {}".format(episode))
        plt.savefig(f"{self.experiment_name}/qf_plots/vf_{episode}_seed_{self.seed}.png")
        plt.close()

    def plot_one_class_initiation_classifiers(self, agent, episode):

        def make_meshgrid(x, y, h=.02):
            x_min, x_max = x.min() - 1, x.max() + 1
            y_min, y_max = y.min() - 1, y.max() + 1
            xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                                 np.arange(y_min, y_max, h))
            return xx, yy

        def make_plot(X, clf):
            X0, X1 = X[:, 0], X[:, 1]
            xx, yy = make_meshgrid(X0, X1)
            Z1 = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
            Z1 = Z1.reshape(xx.shape)
            plt.contour(xx, yy, Z1, levels=[0], linewidths=2, cmap=plt.cm.bone)
            plt.scatter(X0, X1, alpha=0.3)
            plt.ylim((-0.07, 0.07))

        plt.figure(figsize=(14, 10))
        for i, action in enumerate(self.agent.actions):
            plt.subplot(1, len(self.agent.actions), i + 1)
            visited_states = np.array(agent.visited_state_action_pairs[action])
            if self.agent.one_class_classifiers[action] is not None:
                make_plot(visited_states, self.agent.one_class_classifiers[action])

        plt.xlabel("x")
        plt.ylabel("xdot")
        plt.savefig(f"{self.experiment_name}/novelty_plots/ocsvm_{episode}_seed_{self.seed}.png")
        plt.close()

    def train_dqn_agent(self, agent, mdp, episodes, steps):
        per_episode_scores = []
        per_episode_durations = []

        last_10_scores = deque(maxlen=10)
        last_10_durations = deque(maxlen=10)
        iteration_counter = 0

        for episode in range(episodes):
            mdp.reset()
            state = deepcopy(mdp.cur_state)

            score = 0.

            for step in range(steps):
                iteration_counter += 1

                position = mdp.get_position()

                action = agent.act(state.features(), train_mode=True, position=position, use_novelty=True)
                reward, next_state = mdp.execute_agent_action(action)

                reward = 1. if next_state.is_terminal() else 0.
                next_position = mdp.get_position()

                agent.step(state.features(), position, action, reward, next_state.features(), next_position,
                           next_state.is_terminal(), num_steps=1)
                agent.update_epsilon()

                state = next_state
                score += reward

                if state.is_terminal():
                    break

            agent.train_novelty_detector()

            if agent.tensor_log:
                agent.writer.add_scalar("Score", score, iteration_counter)

            if args.make_plots:
                # self.make_value_plot(agent, episode)

                self.plot_one_class_initiation_classifiers(agent, episode)

            last_10_scores.append(score)
            last_10_durations.append(step + 1)
            per_episode_scores.append(score)
            per_episode_durations.append(step + 1)


            print(
                '\rEpisode {}\tAverage Score: {:.2f}\tAverage Duration: {:.2f}\tEpsilon: {:.2f}'.format(
                    episode,
                    np.mean(last_10_scores),
                    np.mean(last_10_durations),
                    agent.epsilon)
            )

        return per_episode_scores, per_episode_durations


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment_name", type=str, required=True)
    parser.add_argument("--run_title", type=str, required=True)  # This is the subdir that we'll be saving in.

    parser.add_argument("--seed", type=int, help="Random seed for this run (default=0)", default=0)
    parser.add_argument("--episodes", type=int, help="# episodes", default=100)
    parser.add_argument("--make_plots", action="store_true", default=False)
    parser.add_argument("--tensor_log", action="store_true", default=False)

    args = parser.parse_args()

    full_experiment_name = f"{args.experiment_name}/{args.run_title}"

    create_log_dir(args.experiment_name)
    create_log_dir(full_experiment_name)
    create_log_dir(f"{full_experiment_name}/qf_plots")
    create_log_dir(f"{full_experiment_name}/novelty_plots")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    exp = Experiment16(args.seed, episodes=args.episodes, experiment_name=full_experiment_name,
                       tensor_log=args.tensor_log, device=device)
    episodic_scores, episodic_durations = exp.run_experiment()

    save_scores(episodic_durations, args.experiment_name, args.seed, run_title=args.run_title)
