# Python imports.
import time
import random
import numpy as np
from copy import deepcopy
from collections import deque
import argparse
import pdb
from sklearn.neighbors import KernelDensity

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


class DensityModel(object):
    def __init__(self, bandwidth=0.2, use_full_state=False):
        self.bandwidth = bandwidth
        self.use_full_state = use_full_state
        self.model = KernelDensity(bandwidth=bandwidth, rtol=0.05)  # r_tol of 0.05 implies an error tolerance of 5%
        self.fitted = False

    def fit(self, state_buffer):
        """
        Args:
            state_buffer (np.ndarray): Array with each row representing the features of the state
        """
        if self.use_full_state:
            input_buffer = state_buffer
        else:  # Use only the position elements of the state vector
            input_buffer = state_buffer[:, :2]

        start_time = time.time()
        self.model.fit(input_buffer)
        end_time = time.time()

        fitting_time = end_time - start_time
        if fitting_time >= 1:
            print("\rDensity Model took {} seconds to fit".format(end_time - start_time))
        self.fitted = True

        return fitting_time

    def get_log_prob(self, states):
        """
        Args:
            states (np.ndarray)

        Returns:
            log_probabilities (np.ndarray): log probability of each state in `states`
        """
        if self.use_full_state:
            X = states
        else:  # Use only the position elements of the state vector
            X = states[:, :2]
        log_pdf = self.model.score_samples(X)
        return log_pdf


class DDPGAgent(Agent):
    def __init__(self, state_size, action_size, action_bound, seed, device, lr_actor=LRA, lr_critic=LRC,
                 batch_size=BATCH_SIZE, tensor_log=False, writer=None, name="DDPG-Agent"):
        self.state_size = state_size
        self.action_size = action_size
        self.action_bound = action_bound
        self.actor_learning_rate = lr_actor
        self.critic_learning_rate = lr_critic
        self.batch_size = batch_size

        self.seed = random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        self.device = device
        self.tensor_log = tensor_log
        self.name = name

        self.noise = OrnsteinUhlenbeckActionNoise(mu=np.zeros(action_size))
        self.actor = Actor(state_size, action_size, action_bound, device=device)
        self.critic = Critic(state_size, action_size, device=device)

        self.target_actor = Actor(state_size, action_size, action_bound, device=device)
        self.target_critic = Critic(state_size, action_size, device=device)

        # Initialize actor target network
        for target_param, param in zip(self.target_actor.parameters(), self.actor.parameters()):
            target_param.data.copy_(param.data)

        # Initialize critic target network
        for target_param, param in zip(self.target_critic.parameters(), self.critic.parameters()):
            target_param.data.copy_(param.data)

        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr_critic, weight_decay=1e-2)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr_actor)

        self.replay_buffer = ReplayBuffer(buffer_size=BUFFER_SIZE, name_buffer="{}_replay_buffer".format(name))
        self.epsilon = 0.2  # 0.2 for ant-maze and 0.15 for point-maze

        # Pseudo-count based exploration
        self.density_model = DensityModel(use_full_state=args.use_full_state)
        self.should_update_density_model = True
        self.n_density_fits = 0

        # Tensorboard logging
        self.writer = None
        if tensor_log: self.writer = writer if writer is not None else SummaryWriter()

        self.n_learning_iterations = 0
        self.n_acting_iterations = 0

        Agent.__init__(self, name, [], gamma=GAMMA)

    def act(self, state, evaluation_mode=False):
        if np.random.random() < 0.8:
            action = self.actor.get_action(state)
        else:
            action = np.random.uniform(-self.action_bound, self.action_bound, self.action_size)
        noise = self.noise()
        if not evaluation_mode:
            action += (noise * self.epsilon)
        action = np.clip(action, -1., 1.)

        if self.writer is not None:
            self.n_acting_iterations = self.n_acting_iterations + 1
            self.writer.add_scalar("{}_action_x".format(self.name), action[0], self.n_acting_iterations)
            self.writer.add_scalar("{}_action_y".format(self.name), action[1], self.n_acting_iterations)
            self.writer.add_scalar("{}_state_x".format(self.name), state[0], self.n_acting_iterations)
            self.writer.add_scalar("{}_state_y".format(self.name), state[1], self.n_acting_iterations)
            self.writer.add_scalar("{}_state_xdot".format(self.name), state[2], self.n_acting_iterations)
            self.writer.add_scalar("{}_state_ydot".format(self.name), state[3], self.n_acting_iterations)
            self.writer.add_scalar("{}_noise_x".format(self.name), noise[0], self.n_acting_iterations)
            self.writer.add_scalar("{}_noise_y".format(self.name), noise[1], self.n_acting_iterations)

        return action

    def step(self, state, action, reward, next_state, done):
        self.replay_buffer.add(state, action, reward, next_state, done)

        if len(self.replay_buffer) > self.batch_size:

            # Update the state-density model every `fitting_interval` number of steps
            # Stop updating after 100 episodes so that we can fix the target value function
            if len(self.replay_buffer) % args.fitting_interval == 0 and \
                    len(self.replay_buffer) > 0 and self.should_update_density_model:
                self.update_density_model()

            # Regular DDPG gradient step
            experiences = self.replay_buffer.sample(batch_size=self.batch_size)
            self._learn(experiences, GAMMA)

    def _learn(self, experiences, gamma):
        states, actions, rewards, next_states, dones = experiences

        # Compute exploration bonus before pushing experiences to the GPU
        exploration_bonus = self.exploration_bonus(states, next_states)
        augmented_rewards = rewards + exploration_bonus

        states = torch.FloatTensor(states).to(self.device)
        actions = torch.FloatTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        augmented_rewards = torch.FloatTensor(augmented_rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(np.float32(dones)).unsqueeze(1).to(self.device)

        next_actions = self.target_actor(next_states)
        Q_targets_next = self.target_critic(next_states, next_actions)

        Q_targets = augmented_rewards + (1.0 - dones) * gamma * Q_targets_next.detach()
        Q_expected = self.critic(states, actions)

        self.critic_optimizer.zero_grad()
        critic_loss = F.mse_loss(Q_expected, Q_targets)
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
            self.writer.add_scalar("{}_mean_raw_rewards".format(self.name), rewards.mean(), self.n_learning_iterations)
            self.writer.add_scalar("{}_mean_aug_rewards".format(self.name), augmented_rewards.mean(), self.n_learning_iterations)
            self.writer.add_scalar("{}_mean_exploration_bonus".format(self.name), exploration_bonus.mean(), self.n_learning_iterations)
            self.writer.add_scalar("{}_critic_loss".format(self.name), critic_loss.item(), self.n_learning_iterations)
            self.writer.add_scalar("{}_actor_loss".format(self.name), actor_loss.item(), self.n_learning_iterations)
            self.writer.add_scalar("{}_critic_grad_norm".format(self.name), compute_gradient_norm(self.critic), self.n_learning_iterations)
            self.writer.add_scalar("{}_actor_grad_norm".format(self.name), compute_gradient_norm(self.actor), self.n_learning_iterations)
            self.writer.add_scalar("{}_sampled_q_values".format(self.name), Q_expected.mean().item(), self.n_learning_iterations)
            self.writer.add_scalar("{}_epsilon".format(self.name), self.epsilon, self.n_learning_iterations)

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

    def exploration_bonus(self, states, next_states):
        if self.density_model.fitted:
            log_pdf = self.density_model.get_log_prob(states)
            log_pdf[log_pdf >= 0] = -0.01

            if args.exploration_strategy == "penalty":
                return self.get_penalty_score(log_pdf)
            elif args.exploration_strategy == "bonus":
                return self.get_bonus_score(log_pdf)
            elif args.exploration_strategy == "shaped":
                next_log_pdf = self.density_model.get_log_prob(next_states)
                return self.get_shaped_bonus_score(log_pdf, next_log_pdf)
            else:
                raise ValueError("Exploration strategy {} not supported".format(args.exploration_strategy))

        return np.zeros(states.shape[0])

    @staticmethod
    def get_penalty_score(log_pdf):
        assert (log_pdf <= 0.).all(), "Expected -ive log-probability, got {}".format(log_pdf)
        penalty = -2. / np.sqrt(-log_pdf)

        # Clipping the penalty at -2 is equivalent to clipping the log prob at -0.5
        return np.clip(penalty, -2., -0.01)

    @staticmethod
    def get_bonus_score(log_pdf):
        exploration_bonus = -1e-4 * log_pdf
        return np.clip(exploration_bonus, None, 100.)

    def get_shaped_bonus_score(self, current_state_log_pdf, next_state_log_pdf):
        current_state_bonus = -current_state_log_pdf
        next_state_bonus = -next_state_log_pdf
        exploration_bonus = (self.gamma * next_state_bonus) - current_state_bonus
        return np.clip(exploration_bonus, None, 100.)

    def update_density_model(self):
        stored_transitions = self.replay_buffer.memory
        stored_states = np.array([transition[0] for transition in stored_transitions])
        fitting_time = self.density_model.fit(stored_states)

        self.n_density_fits += 1
        if self.tensor_log:
            self.writer.add_scalar("{}_fitting_time".format(self.name), fitting_time, self.n_density_fits)

        # Plot the first few density fits as a sanity check
        # When the data set becomes too large, plotting is prohibitively slow
        if self.n_density_fits < 20:
            use_full_state = args.use_full_state
            make_kde_plot(self, stored_states, use_full_state, self.n_density_fits, args.experiment_name, args.seed)

    def update_epsilon(self):
        if args.epsilon_decay:
            if "global" in self.name.lower():
                self.epsilon = max(0., self.epsilon - GLOBAL_LINEAR_EPS_DECAY)
            else:
                self.epsilon = max(0., self.epsilon - OPTION_LINEAR_EPS_DECAY)

    def get_value(self, state):
        action = self.actor.get_action(state)
        return self.critic.get_q_value(state, action)

    def get_qvalues(self, states, actions):
        self.critic.eval()
        with torch.no_grad():
            q_values = self.critic(states, actions)
        self.critic.train()
        return q_values

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
    last_10_scores = deque(maxlen=10)
    last_10_durations = deque(maxlen=10)

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
            # render_sampled_value_function(agent, episode=episode, experiment_name=args.experiment_name)
        if episode == args.update_episodes - 1:
            print("\r Fixing Density Model \r")
            agent.should_update_density_model = False

    return per_episode_scores, per_episode_durations


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment_name", type=str, help="Experiment Name")
    parser.add_argument("--dense_reward", type=bool, help="Whether to use dense/sparse rewards", default=False)
    parser.add_argument("--env", type=str, help="name of gym environment", default="point-env")
    parser.add_argument("--difficulty", type=str, help="Control suite env difficulty", default="easy")
    parser.add_argument("--render", type=bool, help="render environment training", default=False)
    parser.add_argument("--log", type=bool, help="enable tensorboard logging", default=False)
    parser.add_argument("--episodes", type=int, help="number of training episodes", default=200)
    parser.add_argument("--steps", type=int, help="number of steps per episode", default=200)
    parser.add_argument("--device", type=str, help="cuda/cpu", default="cpu")
    parser.add_argument("--seed", type=int, help="random seed", default=0)

    parser.add_argument("--epsilon_decay", type=bool, help="Whether to use e-greedy decay", default=False)
    parser.add_argument("--fitting_interval", type=int, help="# steps b/w updates for density estimation", default=500)
    parser.add_argument("--update_episodes", type=int, help="# episodes after which we fix density model", default=120)
    parser.add_argument("--use_full_state", type=bool, help="(Baseline) Use full state for density model?", default=False)
    parser.add_argument("--exploration_strategy", type=str, help="penalty/bonus/shaped", default="penalty")
    args = parser.parse_args()

    log_dir = create_log_dir(args.experiment_name)

    create_log_dir("saved_runs")
    create_log_dir("value_function_plots")
    create_log_dir("initiation_set_plots")
    create_log_dir("kde_plots")
    create_log_dir("value_function_plots/{}".format(args.experiment_name))
    create_log_dir("initiation_set_plots/{}".format(args.experiment_name))
    create_log_dir("kde_plots/{}".format(args.experiment_name))

    if "reacher" in args.env.lower():
        from simple_rl.tasks.dm_fixed_reacher.FixedReacherMDPClass import FixedReacherMDP
        overall_mdp = FixedReacherMDP(seed=args.seed, difficulty=args.difficulty, render=args.render)
        state_dim = overall_mdp.init_state.features().shape[0]
        action_dim = overall_mdp.env.action_spec().minimum.shape[0]
    elif "ant" in args.env.lower():
        from simple_rl.tasks.ant_maze.AntMazeMDPClass import AntMazeMDP
        overall_mdp = AntMazeMDP(seed=args.seed, dense_reward=args.dense_reward, render=args.render)
        state_dim = overall_mdp.state_space_size()
        action_dim = overall_mdp.action_space_size()
    elif "maze" in args.env.lower():
        from simple_rl.tasks.point_maze.PointMazeMDPClass import PointMazeMDP
        overall_mdp = PointMazeMDP(dense_reward=args.dense_reward, seed=args.seed, render=args.render)
        state_dim = 6
        action_dim = 2
    else:
        from simple_rl.tasks.gym.GymMDPClass import GymMDP
        overall_mdp = GymMDP(args.env, render=args.render)
        state_dim = overall_mdp.env.observation_space.shape[0]
        action_dim = overall_mdp.env.action_space.shape[0]
        overall_mdp.env.seed(args.seed)

    print("{}: State dim: {}, Action dim: {}".format(overall_mdp.env_name, state_dim, action_dim))

    agent_name = overall_mdp.env_name + "_ddpg_agent"
    ddpg_agent = DDPGAgent(state_dim, action_dim, overall_mdp.action_bound, args.seed, torch.device(args.device),
                           tensor_log=args.log, name=agent_name)
    episodic_scores, episodic_durations = train(ddpg_agent, overall_mdp, args.episodes, args.steps)

    save_model(ddpg_agent, episode_number=args.episodes, best=False)
    save_all_scores(episodic_scores, episodic_durations, log_dir, args.seed)

    best_ep, best_agent = load_model(ddpg_agent)
    print("loaded {} from episode {}".format(best_agent.name, best_ep))
    # trained_forward_pass(best_agent, overall_mdp, args.steps)
