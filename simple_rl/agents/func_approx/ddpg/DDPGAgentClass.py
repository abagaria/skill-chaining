# Python imports.
import random
import numpy as np
from copy import deepcopy
from collections import deque
import argparse

# PyTorch imports.
import torch
import torch.optim as optim
import torch.nn.functional as F
from tensorboardX import SummaryWriter

# Other imports.
from simple_rl.agents.AgentClass import Agent
from simple_rl.agents.func_approx.ddpg.model import Actor, Critic, OrnsteinUhlenbeckActionNoise
from simple_rl.agents.func_approx.ddpg.hyperparameters import *
from simple_rl.agents.func_approx.ddpg.replay_buffer import ReplayBuffer
from simple_rl.tasks.gym.GymMDPClass import GymMDP

class DDPGAgent(Agent):
    def __init__(self, state_size, action_size, seed, device, tensor_log=False, name="DDPG-Agent"):
        self.state_size = state_size
        self.action_size = action_size

        # TODO: Set the random seed for torch, random and numpy
        self.seed = random.seed(seed)
        
        self.device = device
        self.tensor_log = tensor_log
        self.name = name

        self.noise = OrnsteinUhlenbeckActionNoise(mu=np.zeros(action_size))
        self.actor = Actor(state_size, action_size, device=device)
        self.critic = Critic(state_size, action_size, device=device)

        self.target_actor = Actor(state_size, action_size, device=device)
        self.target_critic = Critic(state_size, action_size, device=device)

        # Initialize actor target network
        for target_param, param in zip(self.target_actor.parameters(), self.actor.parameters()):
            target_param.data.copy_(param.data)

        # Initialize critic target network
        for target_param, param in zip(self.target_critic.parameters(), self.critic.parameters()):
            target_param.data.copy_(param.data)

        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=LRC)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=LRA)

        self.replay_buffer = ReplayBuffer(buffer_size=BUFFER_SIZE, name_buffer="{}_replay_buffer".format(name))

        # Tensorboard logging
        self.writer = SummaryWriter() if tensor_log else None
        self.n_learning_iterations = 0

        Agent.__init__(self, name, [], gamma=GAMMA)

    def act(self, state, epsilon):
        action = self.actor.get_action(state)
        action += (self.noise() * max(0.0, epsilon))
        return np.clip(action, -1., 1.)

    def step(self, state, action, reward, next_state, done):
        self.replay_buffer.add(state, action, reward, next_state, done)

        if len(self.replay_buffer) > BATCH_SIZE:
            experiences = self.replay_buffer.sample(batch_size=BATCH_SIZE)
            self._learn(experiences, GAMMA)

    def _learn(self, experiences, gamma):
        states, actions, rewards, next_states, dones = experiences
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.FloatTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(np.float32(dones)).unsqueeze(1).to(self.device)

        next_actions = self.target_actor(next_states)
        Q_targets_next = self.target_critic(next_states, next_actions)

        Q_targets = rewards + (1.0 - dones) * GAMMA * Q_targets_next.detach()
        Q_expected = self.critic(states, actions)

        self.critic_optimizer.zero_grad()
        critic_loss = F.smooth_l1_loss(Q_expected, Q_targets)
        critic_loss.backward()
        self.critic_optimizer.step()

        self.actor_optimizer.zero_grad()
        actor_loss = -self.critic(states, self.actor(states))
        actor_loss = actor_loss.mean()
        actor_loss.backward()
        self.actor_optimizer.step()

        self.soft_update(self.actor, self.target_actor, tau=TAU)
        self.soft_update(self.critic, self.target_critic, tau=TAU)

        # Tensorboard logging
        if self.writer is not None:
            self.n_learning_iterations = self.n_learning_iterations + 1
            self.writer.add_scalar("critic_loss", critic_loss.item(), self.n_learning_iterations)
            self.writer.add_scalar("actor_loss", actor_loss.item(), self.n_learning_iterations)
            self.writer.add_scalar("critic_grad_norm", self._compute_gradient_norm(self.critic), self.n_learning_iterations)
            self.writer.add_scalar("actor_grad_norm", self._compute_gradient_norm(self.actor), self.n_learning_iterations)
            self.writer.add_scalar("sampled_q_values", Q_expected.mean().item(), self.n_learning_iterations)

    def soft_update(self, local_model, target_model, tau):
        """
        Soft update of target network from policy network.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Args:
            local_model (nn.Module): weights will be copied from
            target_model (nn.Module): weights will be copied to
            tau (float): interpolation parameter - usually small eg 0.0001
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)

    @staticmethod
    def _compute_gradient_norm(model):
        total_norm = 0.
        for p in model.parameters():
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
        total_norm = total_norm ** (1. / 2)
        return total_norm


def train(agent, mdp, episodes, steps):
    epsilon = 1.0
    best_episodic_reward = -np.inf
    per_episode_scores = []
    last_10_scores = deque(maxlen=50)

    for episode in range(episodes):
        mdp.reset()
        state = deepcopy(mdp.init_state)
        score = 0.
        for step in range(steps):
            epsilon -= epsilon_decay

            action = agent.act(state.features(), epsilon)
            reward, next_state = mdp.execute_agent_action(action)
            agent.step(state.features(), action, reward, next_state.features(), next_state.is_terminal())

            state = next_state
            score += reward

            if state.is_terminal():
                break

        last_10_scores.append(score)
        per_episode_scores.append(score)

        if score > best_episodic_reward:
            torch.save(agent.actor.state_dict(), "best_actor_{}_{}.pkl".format(mdp.env_name, episode))
            torch.save(agent.critic.state_dict(), "best_critic_{}_{}.pkl".format(mdp.env_name, episode))
            best_episodic_reward = score

        print('\rEpisode {}\tAverage Score: {:.2f}'.format(episode, np.mean(last_10_scores)), end="")
        if episode % PRINT_EVERY == 0:
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(episode, np.mean(last_10_scores)))

    return per_episode_scores


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, help="name of gym environment", default="Pendulum-v0")
    parser.add_argument("--render", type=bool, help="render environment training", default=False)
    parser.add_argument("--log", type=bool, help="enable tensorboard logging", default=True)
    parser.add_argument("--episodes", type=int, help="number of training episodes", default=200)
    parser.add_argument("--steps", type=int, help="number of steps per episode", default=200)
    parser.add_argument("--device", type=str, help="cuda/cpu", default="cpu")
    parser.add_argument("--seed", type=int, help="random seed", default=0)
    args = parser.parse_args()
    
    overall_mdp = GymMDP(args.env, render=args.render)
    state_dim = overall_mdp.env.observation_space.shape[0]
    action_dim = overall_mdp.env.action_space.shape[0]
    print("{}: State dim: {}, Action dim: {}".format(overall_mdp.env_name, state_dim, action_dim))
    ddpg_agent = DDPGAgent(state_dim, action_dim, args.seed, torch.device(args.device), tensor_log=args.log)
    episodic_scores = train(ddpg_agent, overall_mdp, args.episodes, args.steps)

