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
import time
import pickle

# Other imports.
from simple_rl.agents.func_approx.exploration.optimism.latent.CountingLatentSpaceClass import CountingLatentSpace
from simple_rl.tasks.gridworld.VisualGridWorldMDPClass import VisualGridWorldMDP
from simple_rl.agents.func_approx.exploration.optimism.latent.utils import create_log_dir


class Experiment13:
    """ Visual Grid-World Learning with Approx Dataset, only counting class """

    def __init__(self, seed, *, grid_size, noise_level, num_steps, device, experiment_name):
        self.mdp = VisualGridWorldMDP(True, grid_size, noise_level, seed)
        state_dim = self.mdp.state_dim

        self.counting_space = CountingLatentSpace(
            state_dim=state_dim, action_dim=4, latent_dim=2, epsilon=0.1, phi_type="function",
            experiment_name="exp13", pixel_observations=True, lam=1,
            optimization_quantity="approx-count", device=device
        )

        self.num_steps = num_steps
        self.seed = seed
        self.experiment_name = experiment_name

    def generate_data(self, num_steps=None):
        if num_steps is None:
            num_steps = self.num_steps

        obs_next_obs_buffer = []
        state_buffer = []
        state_action_dict = defaultdict(list)

        obs = self.mdp.cur_state

        for _ in tqdm(range(num_steps), desc="Collecting data"):
            state = self.mdp.get_position()

            action = random.choice(self.mdp.actions)
            reward, next_obs = self.mdp.execute_agent_action(action)

            obs_next_obs_buffer.append((obs.features(), next_obs.features()))
            state_buffer.append(tuple(state))
            state_action_dict[action].append(obs.features())

            self._update_count(obs=obs, position=state, action=action)

            obs = next_obs
            if obs.is_terminal():
                self.mdp.reset()
                obs = self.mdp.cur_state

        action_buffers = self._get_action_buffers(state_action_dict)
        print("Collected {} state-next_state pairs".format(len(obs_next_obs_buffer)))
        return obs_next_obs_buffer, action_buffers

    @staticmethod
    def _get_action_buffers(state_action_dict):
        action_buffers = []
        sorted_keys = sorted(state_action_dict.keys())
        for action in sorted_keys:
            states = state_action_dict[action]
            states_array = np.vstack(states)
            action_buffers.append(states_array)
        return action_buffers

    def make_time_training_plots(self):
        epochs = 10
        steps_range = [5000]

        for steps in steps_range:
            obs_next_obs_buffer, action_buffers = self.generate_data(num_steps=steps)

            chunk_sizes = [500, 1000, 2000, 4000]
            time_to_train = []

            for chunk_size in chunk_sizes:
                self.counting_space.approx_chunk_size = chunk_size
                start = time.time()
                self.counting_space.train(action_buffers=action_buffers, state_next_state_buffer=obs_next_obs_buffer, epochs=epochs)
                total_time = time.time() - start
                time_to_train.append(total_time)

            plt.plot(chunk_sizes, time_to_train, label=f"{steps} steps")

            with open(f"{self.experiment_name}/time_plots/times_{steps}_steps.pkl", "wb+") as f:
                pickle.dump(time_to_train, f)

        plt.legend()
        plt.xlabel("Chunk Size")
        plt.ylabel(f"Time in seconds ({epochs} epochs)")
        plt.xscale("log")
        plt.yscale("log")
        plt.savefig(f"{self.experiment_name}/time_plots/log_log_times_seed_{self.seed}.png")
        plt.close()

    def run_experiment(self, timing_experiment=False, epochs=100):
        if timing_experiment:
            self.make_time_training_plots()
        else:
            obs_next_obs_buffer, action_buffers = self.generate_data()

            for epoch in range(epochs):
                self.make_count_plot(epoch)
                self.make_latent_plot(epoch)
                self.counting_space.train(action_buffers=action_buffers, state_next_state_buffer=obs_next_obs_buffer, epochs=1)

            self.make_count_plot(epoch + 1)
            self.make_latent_plot(epoch + 1)

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

    def make_count_plot(self, epoch):

        for buffer_idx in self.mdp.actions:

            # for the current action, create a list of (count, [o1, ..., oN]) tuples
            true_vs_est_counts = []
            for sa in self.gt_state_action_counts[buffer_idx]:
                true_count = self.gt_state_action_counts[buffer_idx][sa]
                obs_list = self.gt_state_observation_map[buffer_idx][sa]
                obs_np = np.vstack(obs_list)

                estimated_counts = self.counting_space.get_counts(obs_np, buffer_idx)
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
        plt.savefig(f"{self.experiment_name}/count_plots/true_vs_estimated_counts_seed_{self.seed}_epoch_{epoch}.png")
        plt.close()

    def make_latent_plot(self, epoch):
        # state_latent_pairs = []
        latent_states = []

        combined_state_obs = self.get_combined_state_obs()
        states = list(sorted(combined_state_obs.keys()))

        for_cmap = []
        for state in states:
            all_observations = np.vstack(combined_state_obs[state])

            # Average exploration bonuses over all actions and the batch dimension
            states_repr = self.counting_space.extract_features(all_observations)

            latent_states += states_repr.tolist()
            for _ in range(len(all_observations)):
                for_cmap.append(state)

        color_map = self._get_state_colormap(for_cmap)


        latent_states = np.asarray(latent_states)

        plt.scatter(latent_states[:, 0], latent_states[:, 1], c=color_map, alpha=0.3)
        plt.colorbar()

        plt.title("Latent Space")
        plt.savefig(f"{self.experiment_name}/latent_plots/latents_seed_{self.seed}_epoch_{epoch}.png")
        plt.close()

    def _update_count(self, *, obs, position, action):
        if not hasattr(self, 'gt_state_action_counts'):
            self.gt_state_action_counts = defaultdict(lambda : defaultdict(int))
        if not hasattr(self, 'gt_state_observation_map'):
            self.gt_state_observation_map = defaultdict(lambda : defaultdict(list))

        self.gt_state_action_counts[action][tuple(position)] += 1
        self.gt_state_observation_map[action][tuple(position)].append(obs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment_name", type=str, help="Experiment Name", required=True)
    parser.add_argument("--run_title", type=str, required=True) # This is the subdir that we'll be saving in.

    parser.add_argument("--seed", type=int, help="Random seed for this run (default=0)", default=0)
    parser.add_argument("--steps", type=int, help="# steps", default=200)
    parser.add_argument("--render", type=bool, help="Render the mdp env", default=False)
    parser.add_argument("--grid_size", choices=("small", "medium", "large"), default="medium")
    parser.add_argument("--noise_level", type=float, default=0.)
    parser.add_argument("--timing_plots", action="store_true", default=False)
    parser.add_argument("--epochs", type=int, default=100)

    args = parser.parse_args()

    full_experiment_name = f"{args.experiment_name}/{args.run_title}"

    create_log_dir(args.experiment_name)
    create_log_dir(full_experiment_name)
    create_log_dir(f"{full_experiment_name}/count_plots")
    create_log_dir(f"{full_experiment_name}/bonus_plots")
    create_log_dir(f"{full_experiment_name}/latent_plots")
    create_log_dir(f"{full_experiment_name}/vf_plots")
    create_log_dir(f"{full_experiment_name}/time_plots")


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    exp = Experiment13(args.seed, grid_size=args.grid_size, noise_level=args.noise_level,
                       num_steps=args.steps, device=device, experiment_name=full_experiment_name)

    exp.run_experiment(timing_experiment=args.timing_plots, epochs=args.epochs)
