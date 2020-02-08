# Python imports.
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from collections import defaultdict, Counter
import itertools
import pdb
import random

# Other imports.
from simple_rl.tasks.gridworld.gridworld import GridWorld
from simple_rl.tasks.gridworld.sensors import SensorChain, ResampleSensor, ImageSensor, NoisySensor
from simple_rl.agents.func_approx.exploration.optimism.latent.CountingLatentSpaceClass import CountingLatentSpace


class Experiment9:
    def __init__(self, epsilon, seed=0, lam=0.1, use_noise=False, use_small_grid=False, num_epochs=1):
        np.random.seed(seed)
        self.num_epochs = num_epochs

        if use_small_grid:
            self.env = GridWorld(2, 2)
            scale = 14
            grid_size = 2
        else:
            self.env = GridWorld(4, 4)
            scale = 7
            grid_size = 4

        sensors = [
            ImageSensor(range=((0, self.env._rows), (0, self.env._cols)), pixel_density=1),
            ResampleSensor(scale=scale),
        ]

        if use_noise:
            sensors.append(NoisySensor(sigma=0.1))

        self.sensor = SensorChain(sensors)

        state_dim = (scale * grid_size, scale * grid_size)

        self.counting_space = CountingLatentSpace(state_dim=state_dim, action_dim=4, latent_dim=4, epsilon=epsilon, phi_type="function",
                                                  experiment_name="exp5", pixel_observations=True, lam=lam,
                                                  optimization_quantity="bonus")

    @staticmethod
    def _get_action_buffers(state_action_dict):
        action_buffers = []
        sorted_keys = sorted(state_action_dict.keys())
        for action in sorted_keys:
            states = state_action_dict[action]
            states_array = np.array(states)
            action_buffers.append(states_array)
        return action_buffers

    def generate_data(self, num_steps):
        def unsqueeze(s):
            return s[None, ...]

        # Init state
        state = self.env.agent.position

        # Return buffers
        obs_next_obs_buffer = []
        state_buffer = []
        state_action_dict = defaultdict(list)
        gt_state_action_counts = defaultdict(lambda : defaultdict(int))
        gt_state_observation_map = defaultdict(lambda : defaultdict(list))

        for _ in tqdm(range(num_steps)):
            action = random.choice(self.env.actions)
            next_state, _, done = self.env.step(action)

            obs = self.sensor.observe(state)
            next_obs = self.sensor.observe(next_state)

            obs_next_obs_buffer.append((unsqueeze(obs), unsqueeze(next_obs)))
            state_buffer.append(tuple(state))
            state_action_dict[action].append(unsqueeze(obs))
            gt_state_action_counts[action][tuple(state)] += 1
            gt_state_observation_map[action][tuple(state)].append(unsqueeze(obs))

            state = next_state
            if done:
                state = self.env.reset_agent()

        action_buffers = self._get_action_buffers(state_action_dict)
        print("Collected {} state-next_state pairs".format(len(obs_next_obs_buffer)))
        return obs_next_obs_buffer, action_buffers, gt_state_action_counts, gt_state_observation_map, state_buffer

    @staticmethod
    def get_most_similar_state_idx(buffer):
        """
        Args:
            buffer (np.ndarray): size num_states x 1 x 28 x 28 for MNIST

        Returns:
            similar_idx (int): state most similar to buffer[0]
        """
        assert isinstance(buffer, np.ndarray)
        query_image = buffer[0]
        sq_distances = ((query_image - buffer) ** 2).sum(axis=1)
        similar_idx = np.argmin(sq_distances[1:]) + 1
        return similar_idx

    @staticmethod
    def get_most_dissimilar_state_idx(buffer):
        assert isinstance(buffer, np.ndarray)
        query_image = buffer[0]
        sq_distances = ((query_image - buffer) ** 2).sum(axis=1)
        dissimilar_idx = np.argmax(sq_distances)
        return dissimilar_idx

    @staticmethod
    def _get_state_colormap(state_buffer):
        unique_states = set(state_buffer)
        s_to_n = {state: i for i, state in enumerate(sorted(unique_states))}

        color_map = [s_to_n[s] for s in state_buffer]
        return color_map

    def get_combined_state_obs(self, gt_state_action_counts, gt_state_observation_map):
        state_observation_map = defaultdict(list)
        for action in gt_state_observation_map:
            for state in gt_state_observation_map[action]:
                state_observation_map[state] += gt_state_observation_map[action][state]

        return state_observation_map

    def _get_counting_error(self, gt_state_action_counts, gt_state_observation_map, action):
        errors = []
        for state in gt_state_observation_map[action]:
            true_count = gt_state_action_counts[action][state]
            obs_arr = np.array(gt_state_observation_map[action][state])
            est_counts = self.counting_space.get_counts(obs_arr, action)
            average_count = est_counts.mean()

            errors.append((true_count - average_count) / true_count) # relative error

        return errors

    def _get_mean_counting_error(self, gt_state_action_counts, gt_state_observation_map):
        counting_errors = [self._get_counting_error(gt_state_action_counts, gt_state_observation_map, a) for a in self.env.actions]
        flattened = [elem for ce in counting_errors for elem in ce]
        mean_counting_error = np.array(flattened).mean()
        return mean_counting_error

    def _get_loss_ratio(self, obs_next_obs_buffer, action_buffers):
        # PREPARE THE DATA!
        all_obs = np.array([ono[0] for ono in obs_next_obs_buffer])




        # attractive_loss = self._counting_loss(state_batch_transformed, next_state_batch_transformed,
        #                                       loss_type=self.attractive_loss_type)
        # bonus_loss = self._bonus_loss(phi_s=support_batch_transformed, buffers=buffers)



        pass


    def _single_run(self, num_steps):
        obs_next_obs_buffer, action_buffers, gt_state_action_counts, gt_state_observation_map, state_buffer = self.generate_data(num_steps)
        self.counting_space.train(action_buffers=action_buffers, state_next_state_buffer=obs_next_obs_buffer, epochs=self.num_epochs)

        mean_counting_error = self._get_mean_counting_error(gt_state_action_counts, gt_state_observation_map)


        return mean_counting_error

    def run_experiment(self):
        num_steps_list = list(range(100, 2600, 100))
        counting_errors = []
        for num_steps in num_steps_list:
            mean_counting_error = self._single_run(num_steps)
            counting_errors.append(mean_counting_error)

        plt.plot(num_steps_list, counting_errors)
        plt.plot(num_steps_list, counting_errors, "kx")
        plt.title(f"Counting Errors as a function of num samples, with lam={self.counting_space.lam}")
        plt.show()

        # return gt_state_action_counts

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--lam", type=float, help="lambda", default=1.0)
    parser.add_argument("--epsilon", type=float, help="Epsilon", default=0.1)
    parser.add_argument("--use_noise", action="store_true", default=False)
    parser.add_argument("--small_grid", action="store_true", default=False)
    parser.add_argument("--num_epochs", type=int, default=100)

    args = parser.parse_args()
    exp = Experiment9(args.epsilon, seed=0, lam=args.lam, use_noise=args.use_noise, use_small_grid=args.small_grid,
                      num_epochs=args.num_epochs)
    exp.run_experiment()
