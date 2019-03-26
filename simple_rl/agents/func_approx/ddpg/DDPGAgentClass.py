# Python imports.
import random
import numpy as np
from copy import deepcopy
from collections import deque
import argparse
import os

# PyTorch imports.
import torch
import torch.optim as optim
import torch.nn.functional as F
from tensorboardX import SummaryWriter

# Other imports.
from simple_rl.agents.AgentClass import Agent
from simple_rl.agents.func_approx.ddpg.model import Actor, Critic, OrnsteinUhlenbeckActionNoise
from simple_rl.agents.func_approx.ddpg.replay_buffer import ReplayBuffer
from simple_rl.agents.func_approx.ddpg.hyperparameters import *
from simple_rl.agents.func_approx.ddpg.utils import *
# from simple_rl.tasks.gym.GymMDPClass import GymMDP
# from simple_rl.tasks.dm_fixed_reacher.FixedReacherMDPClass import FixedReacherMDP
from simple_rl.tasks.point_env.PointEnvMDPClass import PointEnvMDP


class DDPGAgent(Agent):
    def __init__(self, state_size, action_size, seed, device, tensor_log=False, name="DDPG-Agent"):
        self.state_size = state_size
        self.action_size = action_size

        self.seed = random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        
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
        self.epsilon = 1.0

        # Tensorboard logging
        self.writer = SummaryWriter() if tensor_log else None
        self.n_learning_iterations = 0

        Agent.__init__(self, name, [], gamma=GAMMA)

    def act(self, state, evaluation_mode=False):
        action = self.actor.get_action(state)
        if not evaluation_mode:
            action += (self.noise() * self.epsilon)
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

        Q_targets = rewards + (1.0 - dones) * gamma * Q_targets_next.detach()
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
            self.writer.add_scalar("critic_grad_norm", compute_gradient_norm(self.critic), self.n_learning_iterations)
            self.writer.add_scalar("actor_grad_norm", compute_gradient_norm(self.actor), self.n_learning_iterations)
            self.writer.add_scalar("sampled_q_values", Q_expected.mean().item(), self.n_learning_iterations)
            self.writer.add_scalar("epsilon", self.epsilon, self.n_learning_iterations)

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

    def update_epsilon(self):
        self.epsilon = max(0., self.epsilon - LINEAR_EPS_DECAY)

def trained_forward_pass(agent, mdp, steps, render=False):
    mdp.reset()
    state = deepcopy(mdp.init_state)
    overall_reward = 0.
    original_render = deepcopy(mdp.render)
    mdp.render = render

    for _ in range(steps):
        action = agent.act(state.features(), evaluation_mode=True)
        reward, next_state = mdp.execute_agent_action(action)
        overall_reward += reward
        state = next_state
        if state.is_terminal():
            break

    mdp.render = original_render
    return overall_reward


def train(agent, mdp, episodes, steps):
    best_episodic_reward = -np.inf
    per_episode_scores = []
    per_episode_durations = []
    last_10_scores = deque(maxlen=50)
    last_10_durations = deque(maxlen=50)

    for episode in range(episodes):
        mdp.reset()
        state = deepcopy(mdp.init_state)
        score = 0.
        for step in range(steps):
            action = agent.act(state.features())
            reward, next_state = mdp.execute_agent_action(action)
            agent.step(state.features(), action, reward, next_state.features(), next_state.is_terminal())
            agent.update_epsilon()
            state = next_state
            score += reward

            if state.is_terminal():
                break

        last_10_scores.append(score)
        per_episode_scores.append(score)
        last_10_durations.append(step)
        per_episode_durations.append(step)

        if score > best_episodic_reward:
            save_model(agent, episode)
            best_episodic_reward = score

        print('\rEpisode {}\tAverage Score: {:.2f}\tAverage Duration: {:.2f}\tEpsilon: {:.2f}'.format(
            episode, np.mean(last_10_scores), np.mean(last_10_durations), agent.epsilon), end="")
        if episode % PRINT_EVERY == 0:
            print('\rEpisode {}\tAverage Score: {:.2f}\tAverage Duration: {:.2f}\tEpsilon: {:.2f}'.format(
            episode, np.mean(last_10_scores), np.mean(last_10_durations), agent.epsilon))

    return per_episode_scores, per_episode_durations


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, help="name of gym environment", default="Pendulum-v0")
    parser.add_argument("--difficulty", type=str, help="Control suite env difficulty", default="easy")
    parser.add_argument("--render", type=bool, help="render environment training", default=False)
    parser.add_argument("--log", type=bool, help="enable tensorboard logging", default=False)
    parser.add_argument("--episodes", type=int, help="number of training episodes", default=200)
    parser.add_argument("--steps", type=int, help="number of steps per episode", default=200)
    parser.add_argument("--device", type=str, help="cuda/cpu", default="cpu")
    parser.add_argument("--seed", type=int, help="random seed", default=0)
    args = parser.parse_args()

    if "reacher" in args.env.lower():
        overall_mdp = FixedReacherMDP(seed=args.seed, difficulty=args.difficulty, render=args.render)
        state_dim = overall_mdp.init_state.features().shape[0]
        action_dim = overall_mdp.env.action_spec().minimum.shape[0]
    elif "point" in args.env.lower():
        overall_mdp = PointEnvMDP(render=args.render)
        state_dim = 4
        action_dim = 2
    else:
        overall_mdp = GymMDP(args.env, render=args.render)
        state_dim = overall_mdp.env.observation_space.shape[0]
        action_dim = overall_mdp.env.action_space.shape[0]
        overall_mdp.env.seed(args.seed)

    print("{}: State dim: {}, Action dim: {}".format(overall_mdp.env_name, state_dim, action_dim))

    agent_name = overall_mdp.env_name + "_ddpg_agent"
    ddpg_agent = DDPGAgent(state_dim, action_dim, args.seed, torch.device(args.device), tensor_log=args.log, name=agent_name)
    episodic_scores, episodic_durations = train(ddpg_agent, overall_mdp, args.episodes, args.steps)

    save_model(ddpg_agent, episode_number=args.episodes, best=False)

    best_ep, best_agent = load_model(ddpg_agent)
    print("loaded {} from episode {}".format(best_agent.name, best_ep))
    trained_forward_pass(best_agent, overall_mdp, args.steps)
