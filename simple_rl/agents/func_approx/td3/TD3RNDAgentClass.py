import ipdb
import argparse
import numpy as np
from copy import deepcopy
from collections import deque

import torch
import torch.nn.functional as F

from simple_rl.agents.func_approx.td3.replay_buffer_rnd import ReplayBuffer
from simple_rl.tasks.d4rl_ant_maze.D4RLAntMazeMDPClass import D4RLAntMazeMDP
from simple_rl.agents.func_approx.td3.model import Actor, DualHeadCritic, RNDModel
from simple_rl.agents.func_approx.td3.utils import *


# Adapted author implementation of Twin Delayed Deep Deterministic Policy Gradients (TD3)
# Paper: https://arxiv.org/abs/1802.09477


class TD3(object):
    def __init__(
            self,
            state_dim,
            action_dim,
            max_action,
            discount=0.99,
            tau=0.005,
            policy_noise=0.2,
            noise_clip=0.5,
            policy_freq=2,
            batch_size=256,
            exploration_noise=0.1,
            lr_c=3e-4, lr_a=3e-4, lr_rnd=1e-4,
            device=torch.device("cuda"),
            name="Global-TD3-Agent",
            augment_with_rewards=False,
            use_obs_normalization=False
    ):

        self.critic_learning_rate = lr_c
        self.actor_learning_rate = lr_a
        self.rnd_learning_rate = lr_rnd
        self.augment_with_rewards = augment_with_rewards
        self.use_obs_normalization = use_obs_normalization

        self.td3_state_dim = state_dim + 1 if augment_with_rewards else state_dim
        self.rnd_state_dim = state_dim

        self.actor = Actor(self.td3_state_dim, action_dim, max_action).to(device)

        self.target_actor = copy.deepcopy(self.actor)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.actor_learning_rate)

        self.critic = DualHeadCritic(self.td3_state_dim, action_dim).to(device)
        self.target_critic = copy.deepcopy(self.critic)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.critic_learning_rate)

        # RND Parameters
        self.rnd = RNDModel(self.rnd_state_dim).to(device)
        self.rnd_optimizer = torch.optim.Adam(self.rnd.parameters(), lr=self.rnd_learning_rate)
        self.reward_rms = RunningMeanStd()
        self.obs_rms = RunningMeanStd(shape=(1, self.rnd_state_dim))

        self.replay_buffer = ReplayBuffer(self.td3_state_dim, action_dim, device=device)

        self.max_action = max_action
        self.action_dim = action_dim
        self.gamma = discount
        self.tau = tau
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.policy_freq = policy_freq
        self.batch_size = batch_size
        self.epsilon = exploration_noise
        self.device = device
        self.name = name

        self.trained_options = []

        self.total_it = 0

    def act(self, state, evaluation_mode=False):
        state = torch.FloatTensor(state.reshape(1, -1)).to(self.device)
        selected_action = self.actor(state)

        selected_action = selected_action.cpu().data.numpy().flatten()
        noise = np.random.normal(0, self.max_action * self.epsilon, size=self.action_dim)
        if not evaluation_mode:
            selected_action += noise
        return selected_action.clip(-self.max_action, self.max_action)

    def get_intrinsic_reward(self, next_state):
        def _normalize(s):
            return (s - self.obs_rms.mean) / np.sqrt(self.obs_rms.var)

        normalized_next_state = _normalize(next_state) if self.use_obs_normalization else next_state[None, ...]
        intrinsic_reward = self._compute_intrinsic_reward(normalized_next_state).item()
        self.reward_rms.update(np.array([intrinsic_reward]))
        return intrinsic_reward

    def batched_get_intrinsic_reward(self, states):
        assert isinstance(states, (np.ndarray, torch.tensor)), type(states)
        intrinsic_reward = self._compute_intrinsic_reward(states).mean().item()
        return intrinsic_reward

    def step(self, state, action, intrinsic_reward, extrinsic_reward, next_state, is_terminal):
        self.replay_buffer.add(state, action, intrinsic_reward, extrinsic_reward, next_state, is_terminal)

        if len(self.replay_buffer) > self.batch_size:
            self.train(self.replay_buffer, self.batch_size)

    def train(self, replay_buffer, batch_size=100):
        self.total_it += 1

        # Sample replay buffer - result is tensors
        state, action, next_state, intrinsic_reward, extrinsic_reward, done = replay_buffer.sample(batch_size)

        with torch.no_grad():
            # Select action according to policy and add clipped noise
            noise = (
                    torch.randn_like(action) * self.policy_noise
            ).clamp(-self.noise_clip, self.noise_clip)

            target_actions = self.target_actor(next_state)

            next_action = (
                    target_actions + noise
            ).clamp(-self.max_action, self.max_action)

            # Compute the target Q value
            target_Q1_E, target_Q1_I, target_Q2_E, target_Q2_I = self.target_critic(next_state, next_action)
            target_Q_E = torch.min(target_Q1_E, target_Q2_E)
            target_Q_I = torch.min(target_Q1_I, target_Q2_I)

            # Normalize the intrinsic rewards before using them
            intrinsic_reward -= self.reward_rms.mean
            intrinsic_reward /= np.sqrt(self.reward_rms.var)
            
            # One-step Bellman targets - note that intrinsic rewards are always non-terminal
            target_Q_E = extrinsic_reward + (1. - done) * self.gamma * target_Q_E
            target_Q_I = intrinsic_reward + (1. - 0.00) * self.gamma * target_Q_I

        # Get current Q estimates
        current_Q1_E, current_Q1_I, current_Q2_E, current_Q2_I = self.critic(state, action)

        # Compute critic loss
        critic_loss = F.mse_loss(current_Q1_E, target_Q_E) + F.mse_loss(current_Q2_E, target_Q_E) \
                    + F.mse_loss(current_Q1_I, target_Q_I) + F.mse_loss(current_Q2_I, target_Q_I)

        # Update the observation normalization
        if self.use_obs_normalization:
            self.obs_rms.update(self.get_features_for_rnd(next_state.detach().cpu().numpy()))

        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Delayed policy updates
        if self.total_it % self.policy_freq == 0:

            # Compute actor loss
            actor_loss = -self.critic.Q1(state, self.actor(state)).mean()

            # Optimize the actor
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # Update the frozen target models
            for param, target_param in zip(self.critic.parameters(), self.target_critic.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

            for param, target_param in zip(self.actor.parameters(), self.target_actor.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        # RND Update
        predict_next_state_feature, target_next_state_feature = self.rnd(self.get_features_for_rnd(next_state))
        rnd_loss = F.mse_loss(predict_next_state_feature, target_next_state_feature.detach())

        self.rnd_optimizer.zero_grad()
        rnd_loss.backward()
        self.rnd_optimizer.step()

    def _compute_intrinsic_reward(self, next_obs):
        """ Given a normalized observation, return the intrinsic reward associated with it. """

        if isinstance(next_obs, np.ndarray):
            next_obs = torch.FloatTensor(self.get_features_for_rnd(next_obs)).to(self.device)

        with torch.no_grad():
            target_features = self.rnd.target(next_obs)
            predicted_features = self.rnd.predictor(next_obs)
            r_int = (target_features - predicted_features).pow(2).sum(1)
        return r_int

    def update_epsilon(self):
        """ We are using fixed (default) epsilons for TD3 because tuning it is hard. """
        pass

    def get_qvalues(self, states, actions):
        """ Get the values associated with the input state-action pairs. """

        self.critic.eval()
        with torch.no_grad():
            q1e, q1i, q2e, q2i = self.critic(states, actions)
            q1 = q1e + q1i
            q2 = q2e + q2i
            q = torch.min(q1, q2)
        self.critic.train()

        return q

    def get_values(self, states):
        """ Get the values associated with the input states. """

        if isinstance(states, np.ndarray):
            states = torch.as_tensor(states).float().to(self.device)

        with torch.no_grad():
            actions = self.actor(states)
            q_values = self.get_qvalues(states, actions)
        return q_values.cpu().numpy()

    def get_features_for_rnd(self, state):
        """ Feature extractor for the exploration module. """

        if len(state.shape) == 1:
            return state[:self.rnd_state_dim]
        if len(state.shape) == 2:
            return state[:, :self.rnd_state_dim]
        ipdb.set_trace()

    def get_augmented_state(self, state, reward):
        """ Feature extractor for the policy learning module. """

        state_features = state.features()
        if self.augment_with_rewards:
            reward_features = np.array([reward])
            state_features = np.concatenate((state_features, reward_features), axis=0)
        return state_features


def pre_train(agent, mdp, episodes, steps, experiment_name):
    """ Random rollouts to warmstart the observation normalizer. """

    assert isinstance(agent, TD3)

    print(f"Starting pre-train with mean {agent.obs_rms.mean} and variance {agent.obs_rms.var}")

    for _ in range(episodes):
        mdp.reset()
        for _ in range(steps):
            a = mdp.sample_random_action()
            _, next_state = mdp.execute_agent_action(a)
            
            next_state = next_state.features()[None, ...]
            agent.obs_rms.update(next_state)

    print(f"Ending pre-train with mean {agent.obs_rms.mean} and variance {agent.obs_rms.var}")


def train(agent, mdp, episodes, steps, experiment_name, starting_episode=0):
    per_episode_scores = []
    per_episode_durations = []
    per_episode_intrinsic_rewards = []
    
    for episode in range(starting_episode, starting_episode+episodes):
        mdp.reset()
        
        state = deepcopy(mdp.init_state)
        intrinsic_reward = 0.
        
        score = 0.
        intrinsic_score = 0.

        for step in range(steps):

            augmented_state = agent.get_augmented_state(state, intrinsic_reward)
            action = agent.act(augmented_state)
            
            reward, next_state = mdp.execute_agent_action(action)
            intrinsic_reward = agent.get_intrinsic_reward(next_state.features())
            augmented_next_state = agent.get_augmented_state(next_state, intrinsic_reward)

            agent.step(augmented_state, action, intrinsic_reward, reward, augmented_next_state, next_state.is_terminal())
            
            score += reward
            state = next_state
            intrinsic_score += intrinsic_reward

            if state.is_terminal():
                break

        per_episode_scores.append(score)
        per_episode_durations.append(step)
        per_episode_intrinsic_rewards.append(intrinsic_score)

        print(f"Episode: {episode} | Intrinsic Reward: {intrinsic_score} | Extrinsic Reward: {score}")

        if episode % 100 == 0:
            time = visualize_next_state_intrinsic_reward_heat_map(agent, mdp, episode, experiment_name)
            print(f"Took {time}s to make the intrinsic reward plot on {len(agent.replay_buffer)} states")

            # Make reward heat map on d4rl dataset
            time = visualize_intrinsic_rewards_on_offline_dataset(agent, mdp, episode, experiment_name)
            print(f"Took {time}s to make the intrinsic reward plot on D4RL states")

            # Make value plots
            max_int, max_ext = make_chunked_value_function_plot(agent, episode, experiment_name)
            print(f"Max intrinsic value: {max_int} \t Max extrinsic value: {max_ext}")

    return per_episode_scores, per_episode_durations, per_episode_intrinsic_rewards


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment_name", type=str, help="Experiment Name")
    parser.add_argument("--env", type=str, help="name of gym environment", default="point-env")
    parser.add_argument("--render", type=bool, help="render environment training", default=False)
    parser.add_argument("--episodes", type=int, help="number of training episodes", default=200)
    parser.add_argument("--steps", type=int, help="number of steps per episode", default=200)
    parser.add_argument("--device", type=str, help="cuda/cpu", default="cpu")
    parser.add_argument("--augment_with_rewards", action="store_true", default=False)
    parser.add_argument("--use_obs_normalization", action="store_true", default=False)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--use_extrinsic_rewards", action="store_true", default=False)
    parser.add_argument("--dense_reward", action="store_true", default=False)

    args = parser.parse_args()

    log_dir = create_log_dir(args.experiment_name)
    create_log_dir("saved_runs")
    create_log_dir("value_function_plots")
    create_log_dir("initiation_set_plots")
    create_log_dir("value_function_plots/{}".format(args.experiment_name))
    create_log_dir("initiation_set_plots/{}".format(args.experiment_name))

    goal_state = np.array((0, 8)) if args.use_extrinsic_rewards else None
    mdp = D4RLAntMazeMDP(seed=args.seed,
                         maze_size="umaze",
                         render=args.render,
                         goal_state=goal_state,
                         dense_reward=args.dense_reward)
    
    agent = TD3(mdp.state_space_size(),
                mdp.action_space_size(),
                max_action=1.,
                device=args.device,
                augment_with_rewards=args.augment_with_rewards,
                use_obs_normalization=args.use_obs_normalization)

    pre_train(agent, mdp, 10, args.steps, args.experiment_name)
    pes, ped, pei = train(agent, mdp, args.episodes, args.steps, args.experiment_name)
