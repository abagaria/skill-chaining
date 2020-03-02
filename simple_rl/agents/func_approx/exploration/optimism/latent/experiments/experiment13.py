# Python imports.
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from collections import defaultdict
import itertools
import ipdb
import random
import argparse
import time
import pickle
import matplotlib

# Other imports.
from simple_rl.agents.func_approx.exploration.optimism.latent.CountingLatentSpaceClass import CountingLatentSpace
from simple_rl.tasks.gridworld.VisualGridWorldMDPClass import VisualGridWorldMDP
from simple_rl.tasks.gym.GymMDPClass import GymMDP
from simple_rl.agents.func_approx.exploration.optimism.latent.utils import create_log_dir, normalize, get_mean_std


class Experiment13:
    """
    Testing out non-RL with chunked dataset. On visual/not GridWorld/MountainCar.
    Also analyze the impact of num_steps, lambda and repulsive loss scaling on the chunked bonus loss function.
    """

    def __init__(self, seed, *, grid_size, noise_level, num_steps, device, lam,
                 experiment_name, optimization_quantity, env_name, pixel_observation, scale_images,
                 bonus_scaling_term, lam_scaling_term):

        assert optimization_quantity in ("chunked-count", "chunked-bonus", "chunked-log"), optimization_quantity

        if "grid" in env_name:
            self.mdp = VisualGridWorldMDP(True, grid_size, noise_level, seed)
        elif "mountain" in env_name.lower():
            self.mdp = GymMDP(env_name="MountainCar-v0", pixel_observation=pixel_observation,
                              seed=seed, control_problem=True, num_stack=4)
        else:
            raise IOError(f"{env_name}")

        state_dim = self.mdp.state_dim
        action_dim = len(self.mdp.actions)

        self.counting_space = CountingLatentSpace(
            state_dim=state_dim, action_dim=action_dim, latent_dim=2, epsilon=0.1, phi_type="function",
            experiment_name=experiment_name, pixel_observations=pixel_observation, lam=lam,
            optimization_quantity=optimization_quantity, device=device, approx_chunk_size=1000,
            bonus_scaling_term=bonus_scaling_term, lam_scaling_term=lam_scaling_term
        )

        self.num_steps = num_steps
        self.seed = seed
        self.experiment_name = experiment_name
        self.scale_images = scale_images

    def _pre_process_data(self, obs_next_obs_buffer, action_buffers):

        states = np.array([ss[0] for ss in obs_next_obs_buffer])
        next_states = np.array([ss[1] for ss in obs_next_obs_buffer])

        mean, std = get_mean_std(states, flat_std=self.mdp.pixel_observation)

        normalized_states = normalize(states, mean, std)
        normalized_next_states = normalize(next_states, mean, std)
        normalized_action_buffers = [normalize(action_buffer, mean, std) for action_buffer in action_buffers]
        normalized_state_next_state_buffer = list(zip(normalized_states, normalized_next_states))

        return normalized_state_next_state_buffer, normalized_action_buffers


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

            # obs_next_obs_buffer.append((obs.features(), next_obs.features()))
            obs_next_obs_buffer.append((np.array(obs.features()), np.array(next_obs.features())))
            state_buffer.append(tuple(state))
            state_action_dict[action].append(np.array(obs.features()))
            # state_action_dict[action].append(obs.features())

            self._update_count(obs=obs, position=state, action=action)

            obs = next_obs
            if obs.is_terminal():
                self.mdp.reset()
                obs = self.mdp.cur_state

        action_buffers = self._get_action_buffers(state_action_dict)
        print("Collected {} state-next_state pairs".format(len(obs_next_obs_buffer)))

        # if self.mdp.env_name == "MountainCar-v0" and not self.mdp.pixel_observation:
        if self.mdp.env_name == "MountainCar-v0":
            if self.scale_images or not self.mdp.pixel_observation:
                obs_next_obs_buffer, action_buffers = self._pre_process_data(obs_next_obs_buffer, action_buffers)
        # if self.mdp.env_name == "MountainCar-v0" and not self.mdp.pixel_observation:
        #     obs_next_obs_buffer, action_buffers = self._pre_process_data(obs_next_obs_buffer, action_buffers)

        return obs_next_obs_buffer, action_buffers, state_buffer

    def _get_action_buffers(self, state_action_dict):
        action_buffers = []
        sorted_keys = sorted(state_action_dict.keys())
        for action in sorted_keys:
            states = state_action_dict[action]
            states_array = np.array(states) if self.mdp.pixel_observation else np.vstack(states)
            action_buffers.append(states_array)
        return action_buffers

    def make_time_training_plots(self):
        epochs = 10
        steps_range = [5000]

        for steps in steps_range:
            obs_next_obs_buffer, action_buffers, _ = self.generate_data(num_steps=steps)

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
            obs_next_obs_buffer, action_buffers, state_buffer = self.generate_data()

            print("Making plots")
            for epoch in tqdm(range(epochs)):
                if "grid" in self.mdp.env_name:
                    self.make_count_plot(epoch)

                # self.make_latent_plot(epoch)

                if self.mdp.env_name == "MountainCar-v0":
                    phi_s = self.mcar_latent_and_counts_plot(state_buffer, obs_next_obs_buffer, epoch)

                self.counting_space.train(action_buffers=action_buffers, state_next_state_buffer=obs_next_obs_buffer, epochs=1, verbose=False)

            self.make_count_plot(epoch + 1)
            self.make_latent_plot(epoch + 1)

            # Measure and log the spread of the latent space
            spread = self.compute_latent_spread(phi_s)
            self.log_latent_spread(*spread)
            print(spread)

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

    def mcar_latent_and_counts_plot(self, state_buffer, obs_next_obs_buffer, epoch):
        observations = np.array([ono[0] for ono in obs_next_obs_buffer])
        states = np.array(state_buffer)
        obs_repr = self.counting_space.extract_features(observations)

        plt.figure(figsize=(14, 10))
        for buffer_idx in self.mdp.actions:
            counts = self.counting_space.get_counts(observations, buffer_idx)
            plt.subplot(2, 3, buffer_idx + 1)
            if self.mdp.pixel_observation:
                plt.scatter(states[:, 0], states[:, 1], c=counts, alpha=0.3)
            else:
                plt.scatter(observations[:, 0], observations[:, 1], c=counts, alpha=0.3)
            plt.colorbar()
            plt.subplot(2, 3, buffer_idx + 4)
            plt.scatter(obs_repr[:, 0], obs_repr[:, 1], c=range(len(observations)), alpha=0.3, norm=matplotlib.colors.LogNorm())

            plt.colorbar()
        plt.savefig(f"{self.experiment_name}/count_scatter_plots/latents_seed_{self.seed}_epoch_{epoch}.png")
        plt.close()

        return obs_repr

    def make_count_plot(self, epoch):

        for buffer_idx in self.mdp.actions:

            # for the current action, create a list of (count, [o1, ..., oN]) tuples
            true_vs_est_counts = []
            for sa in self.gt_state_action_counts[buffer_idx]:
                true_count = self.gt_state_action_counts[buffer_idx][sa]
                obs_list = self.gt_state_observation_map[buffer_idx][sa]

                # We need to add a dimension, with vstack it was just putting it in the channels dim
                obs_np = np.stack(obs_list)

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
            # Again with the vstack, we really want stack.
            all_observations = np.stack(combined_state_obs[state])

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
        self.gt_state_observation_map[action][tuple(position)].append(np.array(obs.features()))

    def compute_latent_spread(self, phi_s):
        """
        Compute different metrics that heuristically measure the spread of the latent space.
        Args:
            phi_s (np.ndarray): observations projected into latent space

        Returns:
            x_range (float)
            y_range (float)
            x_std (float)
            y_std (float)
            volume (float)
            volume_per_point (float)
        """
        x_spread = phi_s[:, 0].max() - phi_s[:, 0].min()
        y_spread = phi_s[:, 1].max() - phi_s[:, 1].min()
        x_std = phi_s[:, 0].std()
        y_std = phi_s[:, 1].std()
        determinant = np.linalg.det(np.cov(phi_s.T))
        volume = np.sqrt(determinant)
        volume_per_point = volume / phi_s.shape[0]
        return x_spread, y_spread, x_std, y_std, volume, volume_per_point

    def log_latent_spread(self, x_range, y_range, x_std, y_std, vol, vol_per_point):
        with open(f"{self.experiment_name}/latent_range_logs/x_range_seed_{self.seed}.pkl", "wb+") as f:
            pickle.dump(x_range, f)
        with open(f"{self.experiment_name}/latent_range_logs/y_range_seed_{self.seed}.pkl", "wb+") as f:
            pickle.dump(y_range, f)
        with open(f"{self.experiment_name}/latent_range_logs/x_std_seed_{self.seed}.pkl", "wb+") as f:
            pickle.dump(x_std, f)
        with open(f"{self.experiment_name}/latent_range_logs/y_std_seed_{self.seed}.pkl", "wb+") as f:
            pickle.dump(y_std, f)
        with open(f"{self.experiment_name}/latent_range_logs/vol_seed_{self.seed}.pkl", "wb+") as f:
            pickle.dump(vol, f)
        with open(f"{self.experiment_name}/latent_range_logs/vol_pp_seed_{self.seed}.pkl", "wb+") as f:
            pickle.dump(vol_per_point, f)


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
    parser.add_argument("--optimization_quantity", type=str, default="chunked-count")
    parser.add_argument("--env_name", type=str)
    parser.add_argument("--pixel_observation", action="store_true", default=False)
    parser.add_argument("--lam", type=float, default=0.1)
    parser.add_argument("--bonus_scaling_term", type=str, default="none")
    parser.add_argument("--lam_scaling_term", type=str, default="fit")

    args = parser.parse_args()

    full_experiment_name = f"{args.experiment_name}/{args.run_title}"

    create_log_dir(args.experiment_name)
    create_log_dir(full_experiment_name)
    create_log_dir(f"{full_experiment_name}/count_plots")
    create_log_dir(f"{full_experiment_name}/count_scatter_plots")
    create_log_dir(f"{full_experiment_name}/bonus_plots")
    create_log_dir(f"{full_experiment_name}/latent_plots")
    create_log_dir(f"{full_experiment_name}/vf_plots")
    create_log_dir(f"{full_experiment_name}/time_plots")
    create_log_dir(f"{full_experiment_name}/latent_range_logs")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    exp = Experiment13(args.seed, grid_size=args.grid_size, noise_level=args.noise_level,
                       num_steps=args.steps, device=device, experiment_name=full_experiment_name,
                       optimization_quantity=args.optimization_quantity, env_name=args.env_name,
                       pixel_observation=args.pixel_observation, scale_images=True, lam=args.lam,
                       bonus_scaling_term=args.bonus_scaling_term, lam_scaling_term=args.lam_scaling_term)

    exp.run_experiment(timing_experiment=args.timing_plots, epochs=args.epochs)
