# Python imports.
import random
import numpy as np
from copy import deepcopy
from collections import deque
import argparse
import ipdb

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
from simple_rl.agents.func_approx.dsc.utils import *
from simple_rl.agents.func_approx.exploration.DiscreteCountExploration import CountBasedDensityModel


class DDPGAgent(Agent):
    def __init__(self, state_size, action_size, seed, device, lr_actor=LRA, lr_critic=LRC,
                 batch_size=BATCH_SIZE, tensor_log=False, writer=None, name="Global-DDPG-Agent", exploration="shaping",
                 trained_options=[], evaluation_epsilon=0.1, use_fixed_noise=True):
        self.state_size = state_size
        self.action_size = action_size
        self.actor_learning_rate = lr_actor
        self.critic_learning_rate = lr_critic
        self.batch_size = batch_size
        self.exploration_method = exploration
        self.trained_options = trained_options
        self.evaluation_epsilon = evaluation_epsilon
        self.use_fixed_noise = use_fixed_noise

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

        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr_critic, weight_decay=1e-2)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr_actor)

        self.replay_buffer = ReplayBuffer(buffer_size=BUFFER_SIZE, name_buffer="{}_replay_buffer".format(name))
        self.epsilon = 1.0

        if exploration == "counts":
            self.density_model = CountBasedDensityModel(state_rounding_decimals=2,
                                                        action_rounding_decimals=2,
                                                        use_position_only=True)

        # Tensorboard logging
        self.writer = None
        if tensor_log: self.writer = writer if writer is not None else SummaryWriter()

        self.n_learning_iterations = 0
        self.n_acting_iterations = 0

        print("Creating {} with exploration strategy of {}".format(self.name, self.exploration_method))

        Agent.__init__(self, name, [], gamma=GAMMA)

    def __getstate__(self):
        actor_state = self.actor.state_dict()
        actor_optimizer = self.actor_optimizer.state_dict()

        critic_state = self.critic.state_dict()
        critic_optimizer = self.critic_optimizer.state_dict()

        target_actor_state = self.target_actor.state_dict()
        target_critic_state = self.target_critic.state_dict()

        return {"name": self.name,
                "epsilon": self.epsilon,
                "actor_state": actor_state,
                "critic_state": critic_state,
                "target_actor_state": target_actor_state,
                "target_critic_state": target_critic_state,
                "actor_optimizer": actor_optimizer,
                "critic_optimizer": critic_optimizer,
                "replay_buffer": self.replay_buffer,
                "state_size": self.state_size,
                "action_size": self.action_size,
                "device": self.device,
                "lr_critic": self.critic_learning_rate,
                "lr_actor": self.actor_learning_rate}

    def __setstate__(self, state_dictionary):
        self.name = state_dictionary["name"]
        self.epsilon = state_dictionary["epsilon"]
        self.state_size = state_dictionary["state_size"]
        self.action_size = state_dictionary["action_size"]
        self.device = state_dictionary["device"]
        self.critic_learning_rate = state_dictionary["lr_critic"]
        self.actor_learning_rate = state_dictionary["lr_actor"]

        self.actor = Actor(self.state_size, self.action_size, device=self.device)
        self.critic = Critic(self.state_size, self.action_size, device=self.device)
        self.target_actor = Actor(self.state_size, self.action_size, device=self.device)
        self.target_critic = Critic(self.state_size, self.action_size, device=self.device)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=self.critic_learning_rate, weight_decay=1e-2)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.actor_learning_rate)

        self.actor.load_state_dict(state_dictionary["actor_state"])
        self.critic.load_state_dict(state_dictionary["critic_state"])
        self.target_actor.load_state_dict(state_dictionary["target_actor_state"])
        self.target_critic.load_state_dict(state_dictionary["target_critic_state"])

        self.actor_optimizer.load_state_dict(state_dictionary["actor_optimizer"])
        self.critic_optimizer.load_state_dict(state_dictionary["critic_optimizer"])

        self.replay_buffer = state_dictionary["replay_buffer"]

    def add_noise_to_action(self, action):

        if self.use_fixed_noise:
            noise = np.random.normal(0, self.evaluation_epsilon, size=(self.action_size,))
            action += noise
        else:  # OU Noise
            noise = self.noise()
            action += (noise * self.epsilon)

        # Adding noise could have taken us outside the action space bounds
        action = np.clip(action, -1., 1.)

        return action

    def act(self, state, evaluation_mode=False):
        action = self.actor.get_action(state)

        if not evaluation_mode:
            action = self.add_noise_to_action(action)

        if self.writer is not None:
            self.n_acting_iterations = self.n_acting_iterations + 1
            self.writer.add_scalar("{}_action_x".format(self.name), action[0], self.n_acting_iterations)
            self.writer.add_scalar("{}_action_y".format(self.name), action[1], self.n_acting_iterations)
            self.writer.add_scalar("{}_state_x".format(self.name), state[0], self.n_acting_iterations)
            self.writer.add_scalar("{}_state_y".format(self.name), state[1], self.n_acting_iterations)
            self.writer.add_scalar("{}_state_xdot".format(self.name), state[2], self.n_acting_iterations)
            self.writer.add_scalar("{}_state_ydot".format(self.name), state[3], self.n_acting_iterations)

        return action

    def step(self, state, action, reward, next_state, done):
        self.replay_buffer.add(state, action, reward, next_state, done)

        if len(self.replay_buffer) > self.batch_size:
            experiences = self.replay_buffer.sample(batch_size=self.batch_size)
            self._learn(experiences, GAMMA)

    def _learn(self, experiences, gamma):
        states, actions, rewards, next_states, dones = experiences

        np_states = np.copy(states)
        np_actions = np.copy(actions)
        np_next_states = np.copy(next_states)

        states = torch.FloatTensor(states).to(self.device)
        actions = torch.FloatTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(np.float32(dones)).unsqueeze(1).to(self.device)

        # Before pushing to the GPU, augment the reward with exploration bonus
        if self.exploration_method == "counts":
            bonuses = self.density_model.batched_get_exploration_bonus(np_states, np_actions)
            rewards += torch.FloatTensor(bonuses).unsqueeze(1).to(self.device)
        elif self.exploration_method == "shaping":
            if len(self.trained_options) > 0 and self.name == "global_option_ddpg_agent":
                for option in self.trained_options:
                    if option.should_target_with_bonus():
                        phi = option.batched_is_init_true
                        shaped_bonus = 1. * phi(np_next_states)
                        rewards = rewards + torch.FloatTensor(shaped_bonus).unsqueeze(1).to(self.device)

        next_actions = self.target_actor(next_states)
        Q_targets_next = self.target_critic(next_states, next_actions)

        Q_targets = rewards + (1.0 - dones) * gamma * Q_targets_next.detach()
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

    def update_epsilon(self):
        if "global" in self.name.lower():
            self.epsilon = max(self.evaluation_epsilon, self.epsilon - GLOBAL_LINEAR_EPS_DECAY)
        else:
            self.epsilon = max(self.evaluation_epsilon, self.epsilon - OPTION_LINEAR_EPS_DECAY)

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

    visualize_next_state_reward_heat_map(agent, args.episodes, args.experiment_name)

    return per_episode_scores, per_episode_durations


def her_rollout(agent, goal, mdp, steps):
    score = 0.
    mdp.reset()
    trajectory = []

    mdp.set_current_goal(goal)

    for step in range(steps):
        state = deepcopy(mdp.cur_state)
        aug_state = np.concatenate((state.features(), goal), axis=0)

        if np.random.uniform() < 0.25:
            action = mdp.sample_random_action()
        else:
            action = agent.act(aug_state)
            
        reward, next_state = mdp.execute_agent_action(action)
        agent.update_epsilon()
        score = score + reward
        trajectory.append((state, action, reward, next_state))
        if next_state.is_terminal():
            break

    return score, trajectory

# I totally butchered this function on this branch, def don't merge this back in to main -Kiran
def her_train(agent, mdp, episodes, steps, goal_state=None, sampling_strategy="fixed", dense_reward=False):

    assert sampling_strategy in ("fixed", "diverse", "test"), sampling_strategy
    if sampling_strategy == "test": assert goal_state is not None, goal_state

    trajectories = []
    per_episode_scores = []
    last_10_scores = deque(maxlen=10)

    for episode in range(episodes):

        # Set the goal to a feasible random state
        if sampling_strategy == "fixed":
            goal_state = np.array([0., 8.])
        if sampling_strategy == "diverse":
            goal_state = np.random.uniform([0,0], [4,4])

        # Roll-out current policy for one episode
        _, trajectory = her_rollout(agent, goal_state, mdp, steps, fixed_epsilon)

        # Debug log the trajectories
        trajectories.append(trajectory)

        # Regular Experience Replay
        score = 0
        for state, action, _, next_state in trajectory:

            reward_func = mdp.dense_gc_reward_function if dense_reward else mdp.sparse_gc_reward_function
            reward, done = reward_func(next_state, goal_state, {})

            augmented_state = np.concatenate((state.features(), goal_state), axis=0)
            augmented_next_state = np.concatenate((next_state.features(), goal_state), axis=0)
            agent.step(augmented_state, action, reward, augmented_next_state, done)

            score += reward

        # If traj is empty, we avoid doing hindsight experience replay
        if len(trajectory) == 0:
            continue

        # `final` strategy for picking up the hindsight goal
        reached_goal = trajectory[-1][-1].features()[:2]

        # Hindsight Experience Replay
        for state, action, _, next_state in trajectory:

            reward_func = mdp.dense_gc_reward_function if dense_reward else mdp.sparse_gc_reward_function
            reward, done = reward_func(next_state, reached_goal, {})

            augmented_state = np.concatenate((state.features(), reached_goal), axis=0)
            augmented_next_state = np.concatenate((next_state.features(), reached_goal), axis=0)
            agent.step(augmented_state, action, reward, augmented_next_state, done)

        # Logging
        per_episode_scores.append(score)
        last_10_scores.append(score)
        print(f"[Goal={goal_state}] Episode: {episode} \t Score: {score} \t Average Score: {np.mean(last_10_scores)}")

    return per_episode_scores, trajectories


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
    parser.add_argument("--goal_conditioned", action="store_true", default=False)
    parser.add_argument("--sampling_strategy", type=str, default="fixed")
    args = parser.parse_args()

    log_dir = create_log_dir(args.experiment_name)
    create_log_dir("saved_runs")
    create_log_dir("value_function_plots")
    create_log_dir("initiation_set_plots")
    create_log_dir("value_function_plots/{}".format(args.experiment_name))
    create_log_dir("initiation_set_plots/{}".format(args.experiment_name))

    if args.env == "d4rl-point-maze-easy":
        from simple_rl.tasks.d4rl_point_maze.D4RLPointMazeMDPClass import D4RLPointMazeMDP
        overall_mdp = D4RLPointMazeMDP(difficulty=args.difficulty,
                                       seed=args.seed, render=args.render, goal_directed=args.goal_conditioned)
        state_dim = overall_mdp.init_state.features().shape[0]
        if args.goal_conditioned:
            state_dim = overall_mdp.init_state.features().shape[0] + 2
        action_dim = overall_mdp.env.action_space.low.shape[0]
    elif "reacher" in args.env.lower():
        from simple_rl.tasks.dm_fixed_reacher.FixedReacherMDPClass import FixedReacherMDP
        overall_mdp = FixedReacherMDP(seed=args.seed, difficulty=args.difficulty, render=args.render)
        state_dim = overall_mdp.init_state.features().shape[0]
        action_dim = overall_mdp.env.action_spec().minimum.shape[0]
    elif "maze" in args.env.lower():
        from simple_rl.tasks.point_maze.PointMazeMDPClass import PointMazeMDP
        overall_mdp = PointMazeMDP(dense_reward=args.dense_reward, seed=args.seed, render=args.render)
        state_dim = 6
        action_dim = 2
    elif "point" in args.env.lower():
        from simple_rl.tasks.point_env.PointEnvMDPClass import PointEnvMDP
        overall_mdp = PointEnvMDP(dense_reward=args.dense_reward, render=args.render)
        state_dim = 4
        action_dim = 2
    else:
        from simple_rl.tasks.gym.GymMDPClass import GymMDP
        overall_mdp = GymMDP(args.env, render=args.render)
        state_dim = overall_mdp.env.observation_space.shape[0]
        action_dim = overall_mdp.env.action_space.shape[0]
        overall_mdp.env.seed(args.seed)

    print("{}: State dim: {}, Action dim: {}".format(overall_mdp.env_name, state_dim, action_dim))

    agent_name = overall_mdp.env_name + "_global_ddpg_agent"
    ddpg_agent = DDPGAgent(state_dim, action_dim, args.seed, torch.device(args.device), tensor_log=args.log,
                           name=agent_name, exploration="none")
    episodic_scores, trajs = her_train(ddpg_agent, overall_mdp, args.episodes, args.steps, goal_state=None,
                                       sampling_strategy=args.sampling_strategy)

    save_model(ddpg_agent, episode_number=args.episodes, best=False)
    save_all_scores(episodic_scores, episodic_scores, log_dir, args.seed)

    # best_ep, best_agent = load_model(ddpg_agent)
    # print("loaded {} from episode {}".format(best_agent.name, best_ep))
    # # trained_forward_pass(best_agent, overall_mdp, args.steps)
