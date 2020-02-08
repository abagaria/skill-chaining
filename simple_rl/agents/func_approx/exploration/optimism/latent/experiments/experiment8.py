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
from simple_rl.tasks.gridworld.VisualGridWorldMDPClass import VisualGridWorldMDP
from simple_rl.agents.func_approx.exploration.optimism.latent.utils import create_log_dir, save_scores


class Experiment8:
    """ Visual Grid-World Learning with DQN in different configurations on (x, y) position obs. """

    def __init__(self, seed, *, pixel_observation, grid_size, noise_level,
                 eval_eps, exploration_method, num_episodes, num_steps, device, experiment_name):
        self.mdp = VisualGridWorldMDP(pixel_observation, grid_size, noise_level, seed)
        state_dim = self.mdp.state_dim
        self.agent = DQNAgent(state_size=state_dim, action_size=len(self.mdp.actions),
                          trained_options=[], seed=seed, device=device,
                          name="GlobalDDQN", lr=1e-4, use_double_dqn=False,
                          exploration_method=exploration_method, pixel_observation=pixel_observation,
                          evaluation_epsilon=eval_eps)
        self.exploration_method = exploration_method
        self.episodes = num_episodes
        self.num_steps = num_steps
        self.seed = seed
        self.experiment_name = experiment_name

    def run_experiment(self):
        training_scores = self.train_dqn_agent(self.agent, self.mdp, self.episodes, self.num_steps)
        return training_scores

    def extract_states(self):
        states = [list(self.gt_state_action_counts[a].keys()) for a in self.mdp.actions]
        return itertools.chain.from_iterable(states)

    def get_combined_state_obs(self):
        state_observation_map = defaultdict(list)
        for action in self.gt_state_observation_map:
            for state in self.gt_state_observation_map[action]:
                state_observation_map[state] += self.gt_state_observation_map[action][state]

        return state_observation_map

    @staticmethod
    def _get_state_colormap(state_buffer):
        unique_states = set(state_buffer)
        s_to_n = {state: i for i, state in enumerate(sorted(unique_states))}

        color_map = [s_to_n[s] for s in state_buffer]
        return color_map


    def make_value_function_plot(self, episode):
        state_value_pairs = []

        combined_state_obs = self.get_combined_state_obs()

        for state in combined_state_obs:
            all_observations = np.array(combined_state_obs[state])
            # all_observations = np.expand_dims(all_observations, axis=1) # SLOPPY
            # ipdb.set_trace()
            value = self.agent.get_batched_values(all_observations).mean()
            state_value_pairs.append((state, value))

        x = [sb[0][0] for sb in state_value_pairs]
        y = [sb[0][1] for sb in state_value_pairs]
        v = [sb[1] for sb in state_value_pairs]

        plt.scatter(x, y, c=v)
        plt.xlabel("x")
        plt.ylabel("y")
        plt.colorbar()
        plt.xlim((-1, 4))
        plt.ylim((-1, 4))
        plt.title("Value function after episode {}".format(episode))
        plt.savefig(f"{self.experiment_name}/vf_plots/vf_episode_{episode}_seed_{self.seed}.png")
        plt.close()

    def make_bonus_plot(self, episode):
        # state_space = [np.array([x, y]) for x in range(4) for y in range(4)]
        state_bonus_pairs = []

        combined_state_obs = self.get_combined_state_obs()
        for state in combined_state_obs:
            all_observations = np.array(combined_state_obs[state])

            # Average exploration bonuses over all actions and the batch dimension
            bonus = self.agent.novelty_tracker.get_batched_exploration_bonus(all_observations).mean()
            state_bonus_pairs.append((state, bonus))

        x = [sb[0][0] for sb in state_bonus_pairs]
        y = [sb[0][1] for sb in state_bonus_pairs]
        b = [sb[1] for sb in state_bonus_pairs]

        plt.scatter(x, y, c=b)
        plt.xlabel("x")
        plt.ylabel("y")
        plt.colorbar()
        plt.xlim((-1, 4))
        plt.ylim((-1, 4))
        plt.title("Average Exploration Bonuses after Episode {}".format(episode))
        plt.savefig(f"{self.experiment_name}/bonus_plots/bonuses_{episode}_seed_{self.seed}.png")
        plt.close()

    def make_count_plot(self, episode):

        for buffer_idx in self.mdp.actions:

            # for the current action, create a list of (count, [o1, ..., oN]) tuples
            true_vs_est_counts = []
            for sa in self.gt_state_action_counts[buffer_idx]:
                true_count = self.gt_state_action_counts[buffer_idx][sa]
                obs_list = self.gt_state_observation_map[buffer_idx][sa]
                obs_np = np.array(obs_list)
                estimated_counts = self.agent.novelty_tracker.counting_space.get_counts(obs_np, buffer_idx)
                for ec in estimated_counts:
                    true_vs_est_counts.append((true_count, ec))

            if len(true_vs_est_counts) > 0:
                true_counts, est_counts = list(zip(*true_vs_est_counts))

                plt.subplot(1, len(self.mdp.actions), buffer_idx + 1)
                plt.plot(np.arange(0, max(true_counts) + 2), np.arange(0, max(true_counts) + 2), "--")
                plt.scatter(true_counts, est_counts, alpha=0.3)
                plt.xlabel("True counts")

        plt.ylabel("Estimated counts")
        plt.suptitle("How well counts match with true counts")
        plt.savefig(f"{self.experiment_name}/count_plots/true_vs_estimated_counts_episode_{episode}_seed_{self.seed}.png")
        plt.close()

    def make_latent_plot(self, episode):
        # state_latent_pairs = []
        latent_states = []

        combined_state_obs = self.get_combined_state_obs()
        states = list(sorted(combined_state_obs.keys()))

        for_cmap = []
        for state in states:
            all_observations = np.array(combined_state_obs[state])

            # Average exploration bonuses over all actions and the batch dimension
            states_repr = self.agent.novelty_tracker.counting_space.extract_features(all_observations)
            # ipdb.set_trace()

            latent_states += states_repr.tolist()
            for _ in range(len(all_observations)):
                for_cmap.append(state)

        color_map = self._get_state_colormap(for_cmap)


        latent_states = np.asarray(latent_states)

        plt.scatter(latent_states[:, 0], latent_states[:, 1], c=color_map, alpha=0.3)

        plt.xlabel("x")
        plt.ylabel("y")
        plt.colorbar()
        plt.xlim((-1.2, 1.2))
        plt.ylim((-1.2, 1.2))
        plt.title("Latent Space after Episode {}".format(episode))
        plt.savefig(f"{self.experiment_name}/latent_plots/latents_{episode}_seed_{self.seed}.png")
        plt.close()

    def _update_count(self, *, obs, position, action):
        if not hasattr(self, 'gt_state_action_counts'):
            self.gt_state_action_counts = defaultdict(lambda : defaultdict(int))
        if not hasattr(self, 'gt_state_observation_map'):
            self.gt_state_observation_map = defaultdict(lambda : defaultdict(list))

        self.gt_state_action_counts[action][tuple(position)] += 1
        self.gt_state_observation_map[action][tuple(position)].append(obs)

    def train_dqn_agent(self, agent, mdp, episodes, steps):
        per_episode_scores = []
        last_10_scores = deque(maxlen=10)
        iteration_counter = 0

        for episode in range(episodes):
            mdp.reset()
            state = deepcopy(mdp.init_state)

            score = 0.
            for _ in range(steps):
                iteration_counter += 1

                position = mdp.get_position()
                action = agent.act(state.features(), train_mode=True, position=position, use_novelty=True)
                reward, next_state = mdp.execute_agent_action(action)
                next_position = mdp.get_position()

                agent.step(state.features(), position, action, reward, next_state.features(), next_position,
                           next_state.is_terminal(), num_steps=1)
                agent.update_epsilon()

                self._update_count(obs=state.features(), position=position, action=action)

                state = next_state
                score += reward
                if agent.tensor_log:
                    agent.writer.add_scalar("Score", score, iteration_counter)

                if state.is_terminal():
                    break

            if agent.exploration_method == "count-phi":
                agent.train_novelty_detector()

            if args.make_plots:
                if self.exploration_method == "count-phi":
                    self.make_count_plot(episode)
                    self.make_latent_plot(episode)
                self.make_bonus_plot(episode)
                self.make_value_function_plot(episode)

            last_10_scores.append(score)
            per_episode_scores.append(score)

            sns_size = len(agent.novelty_tracker.sns_buffer)
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
    parser.add_argument("--pixel_observation", type=bool, help="Images / Dense input", default=False)
    parser.add_argument("--exploration_method", type=str, default="eps-decay")
    parser.add_argument("--use_bonus_during_action_selection", type=bool, default=False)
    parser.add_argument("--eval_eps", type=float, default=0.05)
    parser.add_argument("--make_plots", action="store_true", default=False)
    parser.add_argument("--grid_size", choices=("small", "medium", "large"), default="medium")
    parser.add_argument("--noise_level", type=float, default=0.)

    args = parser.parse_args()

    full_experiment_name = f"{args.experiment_name}/{args.run_title}"

    create_log_dir(args.experiment_name)
    create_log_dir(full_experiment_name)
    create_log_dir(f"{full_experiment_name}/count_plots")
    create_log_dir(f"{full_experiment_name}/bonus_plots")
    create_log_dir(f"{full_experiment_name}/latent_plots")
    create_log_dir(f"{full_experiment_name}/vf_plots")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    exp = Experiment8(args.seed, pixel_observation=args.pixel_observation, grid_size=args.grid_size, noise_level=args.noise_level,
                      eval_eps=args.eval_eps, exploration_method=args.exploration_method, num_episodes=args.episodes,
                      num_steps=args.steps, device=device, experiment_name=full_experiment_name)

    episodic_scores = exp.run_experiment()

    save_scores(episodic_scores, args.experiment_name, args.seed, run_title=args.run_title)
