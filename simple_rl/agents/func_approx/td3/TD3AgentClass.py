import ipdb
import numpy as np
import torch
import torch.nn.functional as F

from simple_rl.agents.func_approx.td3.replay_buffer import ReplayBuffer
from simple_rl.agents.func_approx.td3.model import Actor, Critic, NormActor
from simple_rl.agents.func_approx.td3.utils import *


# Adapted author implementation of Twin Delayed Deep Deterministic Policy Gradients (TD3)
# Paper: https://arxiv.org/abs/1802.09477


class TD3(object):
    def __init__(
            self,
            state_dim,
            action_dim,
            max_action,
            use_output_normalization=True,
            discount=0.99,
            tau=0.005,
            policy_noise=0.2,
            noise_clip=0.5,
            policy_freq=2,
            batch_size=256,
            exploration_noise=0.1,
            device=torch.device("cuda"),
            name="Global-TD3-Agent"
    ):

        self.critic_learning_rate = 3e-5
        self.actor_learning_rate = 3e-5

        if use_output_normalization:
            assert max_action == 1., "Haven't fixed max-action for output-norm yet"
            self.actor = NormActor(state_dim, action_dim).to(device)
        else:
            self.actor = Actor(state_dim, action_dim, max_action).to(device)

        self.target_actor = copy.deepcopy(self.actor)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.actor_learning_rate)

        self.critic = Critic(state_dim, action_dim).to(device)
        self.target_critic = copy.deepcopy(self.critic)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.critic_learning_rate)

        self.replay_buffer = ReplayBuffer(state_dim, action_dim, device=device)

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
        self.use_output_normalization = use_output_normalization

        self.trained_options = []

        self.total_it = 0

    def act(self, state, evaluation_mode=False):
        state = torch.FloatTensor(state.reshape(1, -1)).to(self.device)
        selected_action = self.actor(state)

        if self.use_output_normalization:
            selected_action = self.normalize_actions(selected_action)

        selected_action = selected_action.cpu().data.numpy().flatten()
        noise = np.random.normal(0, self.max_action * self.epsilon, size=self.action_dim)
        if not evaluation_mode:
            selected_action += noise
        return selected_action.clip(-self.max_action, self.max_action)

    def normalize_actions(self, actions):

        if len(actions.shape) == 1:
            actions = actions.unsqueeze(0)

        K = torch.tensor(self.action_dim).to(self.device)
        G = torch.sum(torch.abs(actions), dim=1).view(-1, 1)
        G = G / K

        ones = torch.ones(G.size()).to(self.device)
        G_mod = torch.where(G >= 1, G, ones)

        normalized_actions = actions / G_mod

        return normalized_actions

    def step(self, state, action, reward, next_state, is_terminal):
        self.replay_buffer.add(state, action, reward, next_state, is_terminal)

        if len(self.replay_buffer) > self.batch_size:
            self.train(self.replay_buffer, self.batch_size)

    def train(self, replay_buffer, batch_size=100):
        self.total_it += 1

        # Sample replay buffer - result is tensors
        state, action, next_state, reward, done = replay_buffer.sample(batch_size)

        with torch.no_grad():
            # Select action according to policy and add clipped noise
            noise = (
                    torch.randn_like(action) * self.policy_noise
            ).clamp(-self.noise_clip, self.noise_clip)

            target_actions = self.target_actor(next_state)

            if self.use_output_normalization:
                target_actions = self.normalize_actions(target_actions)

            next_action = (
                    target_actions + noise
            ).clamp(-self.max_action, self.max_action)

            # Compute the target Q value
            target_Q1, target_Q2 = self.target_critic(next_state, next_action)
            target_Q = torch.min(target_Q1, target_Q2)
            target_Q = reward + (1. - done) * self.gamma * target_Q

        # Get current Q estimates
        current_Q1, current_Q2 = self.critic(state, action)

        # Compute critic loss
        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)

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

    def update_epsilon(self):
        """ We are using fixed (default) epsilons for TD3 because tuning it is hard. """
        pass

    def get_qvalues(self, states, actions):
        """ Get the values associated with the input state-action pairs. """

        self.critic.eval()
        with torch.no_grad():
            q1, q2 = self.critic(states, actions)
        self.critic.train()
        return torch.min(q1, q2)

    def get_values(self, states):
        """ Get the values associated with the input states. """

        if isinstance(states, np.ndarray):
            states = torch.as_tensor(states).float().to(self.device)

        with torch.no_grad():
            actions = self.actor(states)
            if self.use_output_normalization:
                actions = self.normalize_actions(actions)
                actions = actions.clamp(-self.max_action, self.max_action)
            q_values = self.get_qvalues(states, actions)
        return q_values.cpu().numpy()
