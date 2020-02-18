# Python imports.
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from collections import defaultdict, Counter
import itertools
import ipdb
import random

# Other imports.
from simple_rl.tasks.gridworld.gridworld import GridWorld
from simple_rl.tasks.gridworld.sensors import SensorChain, ResampleSensor, ImageSensor, NoisySensor, UnsqueezeSensor
from simple_rl.agents.func_approx.exploration.optimism.latent.CountingLatentSpaceClass import CountingLatentSpace


class Experiment10:
    """
    This is our first attempt at the TC-loss on Visual GridWorld. No other loss terms for now.

    Results:
        It makes really good clusters. But sometimes they live on top of each other.
    """
    def __init__(self, epsilon, num_steps=200, seed=0, lam=0.1,
                 use_noise=False, use_small_grid=False, num_epochs=1, weighted_actions=False):
        self.num_steps = num_steps
        np.random.seed(seed)
        self.num_epochs = num_epochs
        self.weighted_actions = weighted_actions

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
            UnsqueezeSensor(dim=0),
        ]

        if use_noise:
            sensors.append(NoisySensor(sigma=0.1))



        self.sensor = SensorChain(sensors)

        state_dim = (1, scale * grid_size, scale * grid_size)

        self.counting_space = CountingLatentSpace(state_dim=state_dim, action_dim=4, latent_dim=2, epsilon=epsilon, phi_type="function",
                                                  experiment_name="exp10", pixel_observations=True, lam=lam,
                                                  optimization_quantity="count-tc")

    @staticmethod
    def _get_action_buffers(state_action_dict):
        action_buffers = []
        sorted_keys = sorted(state_action_dict.keys())
        for action in sorted_keys:
            states = state_action_dict[action]
            states_array = np.array(states)
            action_buffers.append(states_array)
        return action_buffers

    def _get_weighted_random_action(self):
        if not self.weighted_actions:
            return random.choice(self.env.actions)

        probabilities = [0.3, 0.2, 0.3, 0.2]
        return np.random.choice(self.env.actions, p=probabilities)

    def generate_data(self):
        # Init state
        state = self.env.agent.position

        # Return buffers
        state_buffer = []
        obs_buffer = []
        state_action_dict = defaultdict(list)
        gt_state_action_counts = defaultdict(lambda : defaultdict(int))
        gt_state_observation_map = defaultdict(lambda : defaultdict(list))

        obs_next_obs_dict = {action: [] for action in self.env.actions}

        action_dict = defaultdict(int)

        for _ in tqdm(range(self.num_steps)):
            # action = random.choice(self.env.actions)
            action = self._get_weighted_random_action()
            action_dict[action] += 1
            next_state, _, done = self.env.step(action)

            obs = self.sensor.observe(state)
            next_obs = self.sensor.observe(next_state)

            obs_next_obs_dict[action].append((np.copy(obs), np.copy(next_obs))) # np.copy is just for memory bug prevention

            obs_buffer.append(np.copy(obs))
            state_buffer.append(tuple(state))
            state_action_dict[action].append(obs)
            gt_state_action_counts[action][tuple(state)] += 1
            gt_state_observation_map[action][tuple(state)].append(obs)

            state = next_state
            if done:
                state = self.env.reset_agent()

        # action_buffers = self._get_action_buffers(state_action_dict)
        print(action_dict)
        return obs_next_obs_dict, gt_state_action_counts, gt_state_observation_map, state_buffer, obs_buffer

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

    def run_experiment(self):
        obs_next_obs_dict, gt_state_action_counts, gt_state_observation_map, state_buffer, obs_buffer = self.generate_data()

        action_buffers = [obs_next_obs_dict[action] for action in self.env.actions]

        self.counting_space.train(tc_action_buffers=action_buffers, epochs=self.num_epochs)

        observation_representations = self.counting_space.extract_features(np.array(obs_buffer))
        color_map = self._get_state_colormap(state_buffer)

        for buffer_idx in range(len(action_buffers)):

            # for the current action, create a list of (count, [o1, ..., oN]) tuples
            true_vs_est_counts = []
            for sa in gt_state_action_counts[buffer_idx]:
                true_count = gt_state_action_counts[buffer_idx][sa]
                obs_list = gt_state_observation_map[buffer_idx][sa]
                obs_np = np.array(obs_list)
                estimated_counts = self.counting_space.get_counts(obs_np, buffer_idx)
                for ec in estimated_counts:
                    true_vs_est_counts.append((true_count, ec))

            true_counts, est_counts = list(zip(*true_vs_est_counts))

            plt.subplot(1, len(action_buffers), buffer_idx + 1)
            plt.plot(np.arange(0, max(true_counts) + 2), np.arange(0, max(true_counts) + 2), "--")
            plt.scatter(true_counts, est_counts, alpha=0.3)
            plt.xlabel("True counts")

        plt.ylabel("Estimated counts")
        plt.suptitle("How well counts match with true counts")
        plt.show()



        for buffer_idx in range(len(action_buffers)):
            plt.subplot(1, len(action_buffers), buffer_idx + 1)
            plt.scatter(observation_representations[:, 0], observation_representations[:, 1], c=color_map, alpha=0.3)
            plt.colorbar()
        plt.suptitle("Learned latent representations in visual grid-world")
        plt.show()


        # observations = np.array([ss[0] for ss in obs_next_obs_buffer])
        # # gt_states = self.get_ground_truth_states(gt_state_observation_map)
        #
        # self.counting_space.train(action_buffers=action_buffers, state_next_state_buffer=obs_next_obs_buffer, epochs=self.num_epochs)
        #
        # observation_representations = self.counting_space.extract_features(observations)
        #
        # color_map = self._get_state_colormap(state_buffer)
        #
        # for buffer_idx in range(len(action_buffers)):
        #
        #     # for the current action, create a list of (count, [o1, ..., oN]) tuples
        #     true_vs_est_counts = []
        #     for sa in gt_state_action_counts[buffer_idx]:
        #         true_count = gt_state_action_counts[buffer_idx][sa]
        #         obs_list = gt_state_observation_map[buffer_idx][sa]
        #         obs_np = np.array(obs_list)
        #         estimated_counts = self.counting_space.get_counts(obs_np, buffer_idx)
        #         for ec in estimated_counts:
        #             true_vs_est_counts.append((true_count, ec))
        #
        #     true_counts, est_counts = list(zip(*true_vs_est_counts))
        #
        #     plt.subplot(1, len(action_buffers), buffer_idx + 1)
        #     plt.plot(np.arange(0, max(true_counts) + 2), np.arange(0, max(true_counts) + 2), "--")
        #     plt.scatter(true_counts, est_counts, alpha=0.3)
        #     plt.xlabel("True counts")
        #
        # plt.ylabel("Estimated counts")
        # plt.suptitle("How well counts match with true counts")
        # plt.show()
        #
        # for buffer_idx in range(len(action_buffers)):
        #     plt.subplot(1, len(action_buffers), buffer_idx + 1)
        #     plt.scatter(observation_representations[:, 0], observation_representations[:, 1], c=color_map, alpha=0.3)
        #     plt.colorbar()
        # plt.suptitle("Learned latent representations in visual grid-world")
        # plt.show()
        #
        # most_similar_idx = self.get_most_similar_state_idx(observation_representations)
        # most_dissimilar_idx = self.get_most_dissimilar_state_idx(observation_representations)
        #
        # most_similar_image = observations[most_similar_idx]
        # most_dissimilar_image = observations[most_dissimilar_idx]
        #
        # plt.subplot(1, 3, 1)
        # plt.imshow(observations[0].squeeze(0))
        # plt.title("Query image")
        # plt.xticks([])
        # plt.yticks([])
        #
        # plt.subplot(1, 3, 2)
        # plt.imshow(most_similar_image.squeeze(0))
        # plt.title("Most similar image")
        # plt.xticks([])
        # plt.yticks([])
        #
        # plt.subplot(1, 3, 3)
        # plt.imshow(most_dissimilar_image.squeeze(0))
        # plt.title("Most dissimilar image")
        # plt.xticks([])
        # plt.yticks([])
        #
        # plt.show()

        return gt_state_action_counts

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--lam", type=float, help="lambda", default=1.0)
    parser.add_argument("--epsilon", type=float, help="Epsilon", default=0.1)
    parser.add_argument("--use_noise", action="store_true", default=False)
    parser.add_argument("--small_grid", action="store_true", default=False)
    parser.add_argument("--num_steps", type=int, default=200)
    parser.add_argument("--num_epochs", type=int, default=1)
    parser.add_argument("--weighted_actions", action="store_true", default=False)

    args = parser.parse_args()
    exp = Experiment10(args.epsilon, seed=0, lam=args.lam, use_noise=args.use_noise, use_small_grid=args.small_grid,
                      num_steps=args.num_steps, num_epochs=args.num_epochs, weighted_actions=args.weighted_actions)
    gt_sa_counts = exp.run_experiment()

    """
    Where did I leave off? What did I learn?

    The good news is, for some reason, even if scaling is 0, it really does learn amazing clusters. And that's just with counts, not even with bonuses! That's super satisfying.
    Although I really don't know why this clusters so well, it just seems like magic as of now.
    
    If we set scaling=1.0, the easiest way to get the right answer is just to collapse. If there's weight decay, that's the only good answer.
    But it's confusing why the clusters stay so close to each other even when scaling=0.0. You would think that there would be nothing special about this method then.
    
    I think I know why: We see that the only way things are pushed apart is if they're both the child of the same action. But, the upper left is NEVER the child of DOWN or RIGHT.
    And bottom right is NEVER the child of UP or LEFT. So, there will be absolutely no repulsion between the two. That accounts for some of the cluster-overlap.
    In general, the edges and the corners have very few times they can be repelled, which makes it hard to learn that they're distinct.
    We can confirm this on a 2x2 grid. If you run it a few times (esp with weight decay) 1 and 4, and 2 and 3, are super close to each other. That's because nothing makes
    them do otherwise.
    
    What about two squares in the center? They should both be in the same action buffers at least. BUT, their parents may be close together because of that same problem.
    Which would make them unlikely to separate as well.
    
    All in all, I'd say it's very promising, but probably needs another term.

    """
