# Python imports.
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from collections import defaultdict, Counter
import itertools
import ipdb
import random
import argparse
from copy import deepcopy
from collections import deque

# Other imports.
from simple_rl.tasks.gridworld.gridworld import GridWorld
from simple_rl.tasks.gridworld.sensors import SensorChain, ResampleSensor, ImageSensor, NoisySensor
from simple_rl.agents.func_approx.exploration.optimism.latent.CountingLatentSpaceClass import CountingLatentSpace
from simple_rl.agents.func_approx.dqn.DQNAgentClass import DQNAgent
from simple_rl.tasks.gym.GymMDPClass import GymMDP
from simple_rl.agents.func_approx.exploration.optimism.latent.utils import create_log_dir, save_scores


class Experiment12:
    """ RL on mountain car using the learned exploration bonus. """

    def __init__(self, seed, *, pixel_observation,
                 eval_eps, exploration_method, num_episodes, num_steps, device, experiment_name):
        self.mdp = GymMDP("MountainCar-v0", pixel_observation=pixel_observation, seed=seed)
        state_dim = self.mdp.state_dim
        self.agent = DQNAgent(state_size=state_dim, action_size=len(self.mdp.actions),
                          trained_options=[], seed=seed, device=device,
                          name="GlobalDDQN", lr=1e-3, use_double_dqn=False, # TODO: Changed learning rate
                          exploration_method=exploration_method, pixel_observation=pixel_observation,
                          evaluation_epsilon=eval_eps, tensor_log=True, experiment_name=experiment_name)
        self.exploration_method = exploration_method
        self.episodes = num_episodes
        self.num_steps = num_steps
        self.seed = seed
        self.experiment_name = experiment_name

    def run_experiment(self):
        training_scores = self.train_dqn_agent(self.agent, self.mdp, self.episodes, self.num_steps)
        return training_scores

    def make_latent_plot(self, agent, episode):

        # Normalize the data before asking for its embeddings
        normalized_sns_buffer = agent.novelty_tracker.get_sns_buffer(normalized=True)

        # phi(s)
        states = np.array([sns[0] for sns in normalized_sns_buffer])
        states_repr = self.agent.novelty_tracker.counting_space.extract_features(states)

        # Plotting
        plt.scatter(states_repr[:, 0], states_repr[:, 1], alpha=0.3)
        plt.xlabel("x")
        plt.ylabel("y")

        plt.title("Latent Space after Episode {}".format(episode))
        plt.savefig(f"{self.experiment_name}/latent_plots/latents_{episode}_seed_{self.seed}.png")
        plt.close()

    def make_bonus_plot(self, agent, episode):

        # Normalize the data before asking for its embeddings
        un_normalized_sns_buffer = agent.novelty_tracker.get_sns_buffer()

        states = np.array([sns[0] for sns in un_normalized_sns_buffer])
        bonuses = agent.novelty_tracker.get_batched_exploration_bonus(states)

        plt.figure(figsize=(14, 10))
        for i, action in enumerate(self.mdp.actions):
            plt.subplot(1, len(self.mdp.actions), i + 1)
            plt.scatter(states[:, 0], states[:, 1], c=bonuses[:, action])
            plt.colorbar()

        plt.suptitle("Exploration Bonuses after Episode {}".format(episode))
        plt.savefig(f"{self.experiment_name}/bonus_plots/bonuses_{episode}_seed_{self.seed}.png")
        plt.close()

    def make_value_plot(self, agent, episode):
        states = np.array([transition.state for transition in agent.replay_buffer])

        states_tensor = torch.from_numpy(states).float().to(agent.device)

        qvalues = agent.get_batched_qvalues(states_tensor, None).cpu().numpy()

        plt.figure(figsize=(14, 10))
        for action in self.agent.actions:
            plt.subplot(1, len(self.agent.actions), action + 1)
            plt.scatter(states[:, 0], states[:, 1], c=qvalues[:, action])
            plt.colorbar()

        plt.suptitle("Q-functions after episode {}".format(episode))
        plt.savefig(f"{self.experiment_name}/qf_plots/vf_{episode}_seed_{self.seed}.png")
        plt.close()

    def train_dqn_agent(self, agent, mdp, episodes, steps):
        per_episode_scores = []
        last_10_scores = deque(maxlen=10)
        iteration_counter = 0

        for episode in range(episodes):
            mdp.reset()
            state = deepcopy(mdp.cur_state)

            score = 0.
            for _ in range(steps):
                agent.writer.add_scalar("TestingItOut", iteration_counter, iteration_counter)
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

            if agent.tensor_log:
                agent.writer.add_scalar("Score", score, iteration_counter)

            if agent.exploration_method == "count-phi" and episode >= 1:
                agent.train_novelty_detector()

            if args.make_plots:
                if self.exploration_method == "count-phi":
                    self.make_latent_plot(agent, episode)
                    self.make_bonus_plot(agent, episode)
                    self.make_value_plot(agent, episode)

            last_10_scores.append(score)
            per_episode_scores.append(score)

            sns_size = len(agent.novelty_tracker.un_normalized_sns_buffer)
            lam = agent.novelty_tracker.counting_space.lam
            print('\rEpisode {}\tAverage Score: {:.2f}\tEpsilon: {:.2f}\tSNS Size: {}\tLam {}'.format(episode,
                                                                                             np.mean(last_10_scores),
                                                                                             agent.epsilon, sns_size, lam))
        return per_episode_scores


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment_name", type=str, help="Experiment Name", required=True)
    parser.add_argument("--run_title", type=str, required=True) # This is the subdir that we'll be saving in.

    parser.add_argument("--seed", type=int, help="Random seed for this run (default=0)", default=0)
    parser.add_argument("--episodes", type=int, help="# episodes", default=100)
    parser.add_argument("--steps", type=int, help="# steps", default=25)
    parser.add_argument("--render", type=bool, help="Render the mdp env", default=False)
    parser.add_argument("--pixel_observation", action="store_true", help="Images / Dense input", default=False)
    parser.add_argument("--exploration_method", type=str, default="eps-decay")
    parser.add_argument("--use_bonus_during_action_selection", type=bool, default=False)
    parser.add_argument("--eval_eps", type=float, default=0.05)
    parser.add_argument("--make_plots", action="store_true", default=False)

    args = parser.parse_args()

    full_experiment_name = f"{args.experiment_name}/{args.run_title}"

    create_log_dir(args.experiment_name)
    create_log_dir(full_experiment_name)
    create_log_dir(f"{full_experiment_name}/bonus_plots")
    create_log_dir(f"{full_experiment_name}/latent_plots")
    create_log_dir(f"{full_experiment_name}/qf_plots")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    exp = Experiment12(args.seed, pixel_observation=args.pixel_observation,
                      eval_eps=args.eval_eps, exploration_method=args.exploration_method, num_episodes=args.episodes,
                      num_steps=args.steps, device=device, experiment_name=full_experiment_name)

    episodic_scores = exp.run_experiment()

    save_scores(episodic_scores, args.experiment_name, args.seed, run_title=args.run_title)
