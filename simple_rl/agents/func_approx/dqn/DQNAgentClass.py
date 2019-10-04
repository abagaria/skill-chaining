import numpy as np
import random
from collections import namedtuple, deque
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
import pdb
from copy import deepcopy
import shutil
import os
import time
import argparse
import pickle

import torch.optim as optim

import torch
import torch.nn as nn
import torch.nn.functional as F
from tensorboardX import SummaryWriter

from simple_rl.agents.AgentClass import Agent
from simple_rl.agents.func_approx.ddpg.utils import compute_gradient_norm
from simple_rl.agents.func_approx.dqn.replay_buffer import ReplayBuffer
from simple_rl.agents.func_approx.dqn.model import ConvQNetwork, DenseQNetwork
from simple_rl.agents.func_approx.dqn.epsilon_schedule import *
from simple_rl.tasks.gym.GymMDPClass import GymMDP
from simple_rl.tasks.lunar_lander.LunarLanderMDPClass import LunarLanderMDP
from simple_rl.agents.func_approx.dqn.RandomNetworkDistillationClass import RNDModel, RunningMeanStd


## Hyperparameters
BUFFER_SIZE = int(1e6)  # replay buffer size
BATCH_SIZE = 32  # minibatch size
GAMMA = 0.99  # discount factor
TAU = 1e-3  # for soft update of target parameters
LR = 1e-4  # learning rate
UPDATE_EVERY = 1  # how often to update the network
NUM_EPISODES = 3500
NUM_STEPS = 10000

class DQNAgent(Agent):
    """Interacts with and learns from the environment."""

    def __init__(self, state_size, action_size, trained_options, seed, device, name="DQN-Agent",
                 eps_start=1., tensor_log=False, lr=LR, use_double_dqn=True, gamma=GAMMA, loss_function="huber",
                 gradient_clip=None, evaluation_epsilon=0.05, exploration_method="eps-greedy",
                 pixel_observation=False, writer=None):
        self.state_size = state_size
        self.action_size = action_size
        self.trained_options = trained_options
        self.learning_rate = lr
        self.use_ddqn = use_double_dqn
        self.gamma = gamma
        self.loss_function = loss_function
        self.gradient_clip = gradient_clip
        self.evaluation_epsilon = evaluation_epsilon
        self.exploration_method = exploration_method
        self.pixel_observation = pixel_observation
        self.seed = random.seed(seed)
        np.random.seed(seed)
        self.tensor_log = tensor_log
        self.device = device

        # Q-Network
        if pixel_observation:
            self.policy_network = ConvQNetwork(in_channels=4, n_actions=action_size).to(self.device)
            self.target_network = ConvQNetwork(in_channels=4, n_actions=action_size).to(self.device)
        else:
            self.policy_network = DenseQNetwork(state_size, action_size, seed, fc1_units=32, fc2_units=16).to(self.device)
            self.target_network = DenseQNetwork(state_size, action_size, seed, fc1_units=32, fc2_units=16).to(self.device)

        self.optimizer = optim.Adam(self.policy_network.parameters(), lr=lr)

        # Replay memory
        self.replay_buffer = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, seed, self.device, pixel_observation)
        # Initialize time step (for updating every UPDATE_EVERY steps)
        self.t_step = 0

        # Epsilon strategy
        if exploration_method == "eps-greedy":
            self.epsilon_schedule = GlobalEpsilonSchedule(eps_start, evaluation_epsilon) if "global" in name.lower() else OptionEpsilonSchedule(eps_start)
            self.epsilon = eps_start
        elif exploration_method == "rnd":
            self.epsilon_schedule = GlobalEpsilonSchedule(eps_start, evaluation_epsilon)  # ConstantEpsilonSchedule(evaluation_epsilon)
            self.epsilon = eps_start
            self.rnd = RNDModel(device=device)
        else:
            raise NotImplementedError("{} not implemented", exploration_method)

        self.num_executions = 0 # Number of times act() is called (used for eps-decay)

        # Debugging attributes
        self.num_updates = 0
        self.num_epsilon_updates = 0

        if self.tensor_log:
            self.writer = SummaryWriter() if writer is None else writer

        print("\nCreating {} with lr={} and ddqn={} and buffer_sz={}\n".format(name, self.learning_rate,
                                                                               self.use_ddqn, BUFFER_SIZE))

        Agent.__init__(self, name, range(action_size), GAMMA)

    def get_impossible_option_idx(self, state):

        # Arg-max only over actions that can be executed from the current state
        # -- In general an option can be executed from s if s is in its initiation set and NOT in its termination set
        # -- However, in the case of the goal option we just need to ensure that we are in its initiation set since
        # -- its termination set is terminal anyway and we are thus not in the risk of executing og from its
        # -- termination set.

        impossible_option_idx = []
        for idx, option in enumerate(self.trained_options):
            np_state = state.cpu().data.numpy()[0] if not isinstance(state, np.ndarray) else state

            if option.parent is None:
                assert option.name == "overall_goal_policy" or option.name == "global_option"
                impossible = not option.is_init_true(np_state)
            else:
                impossible = (not option.is_init_true(np_state)) or option.is_term_true(np_state)

            if impossible:
                impossible_option_idx.append(idx)

        return impossible_option_idx

    def act(self, state, train_mode=True):
        """
        Interface to the DQN agent: state can be output of env.step() and returned action can be input into next step().
        Args:
            state (np.array): numpy array state from Gym env
            train_mode (bool): if training, use the internal epsilon. If evaluating, set epsilon to min epsilon

        Returns:
            action (int): integer representing the action to take in the Gym env
        """
        self.num_executions += 1
        epsilon = self.epsilon if train_mode else self.evaluation_epsilon

        state = np.array(state)  # Lazy Frame
        state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        self.policy_network.eval()
        with torch.no_grad():
            action_values = self.policy_network(state)
        self.policy_network.train()

        impossible_option_idx = self.get_impossible_option_idx(state)

        for impossible_idx in impossible_option_idx:
            action_values[0][impossible_idx] = torch.min(action_values, dim=1)[0] - 1.

        action_values = action_values.cpu().data.numpy()
        # Epsilon-greedy action selection
        if random.random() > epsilon:
            return np.argmax(action_values)

        all_option_idx = list(range(len(self.trained_options))) if len(self.trained_options) > 0 else self.actions
        possible_option_idx = list(set(all_option_idx).difference(impossible_option_idx))
        randomly_chosen_option = random.choice(possible_option_idx)

        return randomly_chosen_option

    def get_best_actions_batched(self, states):
        q_values = self.get_batched_qvalues(states)
        return torch.argmax(q_values, dim=1)

    def get_value(self, state):
        action_values = self.get_qvalues(state)

        # Argmax only over actions that can be implemented from the current state
        impossible_option_idx = self.get_impossible_option_idx(state)
        for impossible_idx in impossible_option_idx:
            action_values[0][impossible_idx] = torch.min(action_values).item() - 1.

        return np.max(action_values.cpu().data.numpy())

    def get_qvalue(self, state, action_idx):
        state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        self.policy_network.eval()
        with torch.no_grad():
            action_values = self.policy_network(state)
        self.policy_network.train()
        return action_values[0][action_idx]

    def get_qvalues(self, state):
        state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        self.policy_network.eval()
        with torch.no_grad():
            action_values = self.policy_network(state)
        self.policy_network.train()

        return action_values

    def get_batched_qvalues(self, states):
        """
        Q-values corresponding to `states` for all ** permissible ** actions/options given `states`.
        Args:
            states (torch.tensor) of shape (64 x 4)

        Returns:
            qvalues (torch.tensor) of shape (64 x |A|)
        """
        self.policy_network.eval()
        with torch.no_grad():
            action_values = self.policy_network(states)
        self.policy_network.train()

        if len(self.trained_options) > 0:
            # Move the states and action values to the cpu to allow numpy computations
            states = states.cpu().data.numpy()
            action_values = action_values.cpu().data.numpy()

            for idx, option in enumerate(self.trained_options): # type: Option
                try:
                    inits = option.batched_is_init_true(states)
                    terms = np.zeros(inits.shape) if option.parent is None else option.parent.batched_is_init_true(states)
                    action_values[(inits.squeeze() != 1) | (terms.squeeze() == 1), idx] = np.min(action_values) - 1.
                except:
                    pdb.set_trace()

            # Move the q-values back the GPU
            action_values = torch.from_numpy(action_values).float().to(self.device)

        return action_values

    def step(self, state, action, reward, next_state, done, num_steps=1):
        """
        Interface method to perform 1 step of learning/optimization during training.
        Args:
            state (np.array): state of the underlying gym env
            action (int)
            reward (float)
            next_state (np.array)
            done (bool): is_terminal
            num_steps (int): number of steps taken by the option to terminate
        """
        # Save experience in replay memory
        self.replay_buffer.add(state, action, reward, next_state, done, num_steps)

        # Learn every UPDATE_EVERY time steps.
        self.t_step = (self.t_step + 1) % UPDATE_EVERY
        if self.t_step == 0:
            # If enough samples are available in memory, get random subset and learn
            if len(self.replay_buffer) > BATCH_SIZE:
                experiences = self.replay_buffer.sample(batch_size=BATCH_SIZE)
                self._learn(experiences, GAMMA)
                if self.tensor_log:
                    self.writer.add_scalar("NumPositiveTransitions", self.replay_buffer.positive_transitions[-1], self.num_updates)
                self.num_updates += 1

    def _learn(self, experiences, gamma):
        """
        Update value parameters using given batch of experience tuples.
        Args:
            experiences (tuple<torch.Tensor>): tuple of (s, a, r, s', done, tau) tuples
            gamma (float): discount factor
        """
        states, actions, rewards, next_states, dones, steps = experiences

        if self.exploration_method == "rnd":
            observations = states[:, -1, :, :].unsqueeze(1)
            intrinsic_rewards = self.rnd(observations)
            self.rnd.update(intrinsic_rewards)

        # Get max predicted Q values (for next states) from target model
        if self.use_ddqn:

            if len(self.trained_options) == 0:
                self.policy_network.eval()
                with torch.no_grad():
                    selected_actions = self.policy_network(next_states).argmax(dim=1).unsqueeze(1)
                self.policy_network.train()
            else:
                selected_actions = self.get_best_actions_batched(next_states).unsqueeze(1)

            Q_targets_next = self.target_network(next_states).detach().gather(1, selected_actions)
        else:
            Q_targets_next = self.target_network(next_states).detach().max(1)[0].unsqueeze(1)
            raise NotImplementedError("I have not fixed the Q(s',a') problem for vanilla DQN yet")

        # Options in SMDPs can take multiple steps to terminate, the Q-value needs to be discounted appropriately
        discount_factors = gamma ** steps

        # Compute Q targets for current states
        # Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))
        Q_targets = rewards + (discount_factors * Q_targets_next * (1 - dones))

        # Get expected Q values from local model
        Q_expected = self.policy_network(states).gather(1, actions)

        # Compute loss
        if self.loss_function == "huber":
            loss = F.smooth_l1_loss(Q_expected, Q_targets)
        elif self.loss_function == "mse":
            loss = F.mse_loss(Q_expected, Q_targets)
        else:
            raise NotImplementedError("{} loss function type not implemented".format(self.loss_function))

        # Minimize the loss
        self.optimizer.zero_grad()
        loss.backward()

        # # Gradient clipping: tried but the results looked worse -- needs more testing
        if self.gradient_clip is not None:
            for param in self.policy_network.parameters():
                param.grad.data.clamp_(-self.gradient_clip, self.gradient_clip)

        self.optimizer.step()

        if self.tensor_log:
            self.writer.add_scalar("DQN-Loss", loss.item(), self.num_updates)
            self.writer.add_scalar("DQN-AverageTargetQvalue", Q_targets.mean().item(), self.num_updates)
            self.writer.add_scalar("DQN-AverageQValue", Q_expected.mean().item(), self.num_updates)
            self.writer.add_scalar("DQN-GradientNorm", compute_gradient_norm(self.policy_network), self.num_updates)

        # ------------------- update target network ------------------- #
        self.soft_update(self.policy_network, self.target_network, TAU)

    def soft_update(self, local_model, target_model, tau):
        """
        Args:
            local_model (nn.Module): weights will be copied from
            target_model (nn.Module): weights will be copied to
            tau (float): interpolation parameter - usually small eg 0.0001
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)

    def update_epsilon(self):
        self.num_epsilon_updates += 1
        self.epsilon = self.epsilon_schedule.update_epsilon(self.epsilon, self.num_epsilon_updates)

        # Log epsilon decay
        if self.tensor_log:
            self.writer.add_scalar("DQN-Epsilon", self.epsilon, self.num_epsilon_updates)

def train(agent, mdp, episodes, steps, init_episodes=10):
    per_episode_scores = []
    last_10_scores = deque(maxlen=100)
    iteration_counter = 0
    state_ri_buffer = []

    # Observation and reward normalization
    reward_rms = RunningMeanStd()
    obs_rms = RunningMeanStd(shape=(1, 84, 84))

    # Initialize the RMS normalizers
    if agent.exploration_method == "rnd":
        for episode in range(init_episodes):
            observation_buffer = []
            mdp.reset()
            init_observation = np.array(mdp.init_state.features())[-1, :, :]
            assert init_observation.shape == (84, 84), init_observation.shape
            observation_buffer.append(init_observation)
            while True:
                action = np.random.randint(0, overall_mdp.env.action_space.n)
                r, state = mdp.execute_agent_action(action)
                observation = np.array(state.features())[-1, :, :]
                observation_buffer.append(observation)
                if state.is_terminal():
                    break
            observation_batch = np.stack(observation_buffer)
            obs_rms.update(observation_batch)

    for episode in range(episodes):
        mdp.reset()
        state = deepcopy(mdp.init_state)

        observation_buffer = []
        intrinsic_reward_buffer = []

        init_observation = np.array(mdp.init_state.features())[-1, :, :]
        assert init_observation.shape == (84, 84), init_observation.shape
        observation_buffer.append(init_observation)

        score = 0.
        while True:
            iteration_counter += 1
            action = agent.act(state.features(), train_mode=True)
            reward, next_state = mdp.execute_agent_action(action)

            if agent.exploration_method == "rnd":
                observation = np.array(state.features())[-1, :, :]
                normalized_observation = ((observation - obs_rms.mean) / np.sqrt(obs_rms.var)).clip(-5, 5)
                intrinsic_reward = agent.rnd.get_single_reward(normalized_observation)

                observation_buffer.append(observation)
                intrinsic_reward_buffer.append(intrinsic_reward)
                normalized_intrinsic_reward = (intrinsic_reward - reward_rms.mean) / np.sqrt(reward_rms.var)

                # Logging
                player_position = mdp.get_player_position()
                state_ri_buffer.append((player_position, normalized_intrinsic_reward))
                reward += normalized_intrinsic_reward

                if agent.tensor_log:
                    agent.writer.add_scalar("Normalized Ri", normalized_intrinsic_reward, iteration_counter)

            agent.step(state.features(), action, reward, next_state.features(), next_state.is_terminal(), num_steps=1)
            agent.update_epsilon()
            state = next_state
            score += reward
            if agent.tensor_log:
                agent.writer.add_scalar("Score", score, iteration_counter)

            game_over = mdp.game_over if hasattr(mdp, 'game_over') else False
            if state.is_terminal() or game_over:
                break

        if agent.exploration_method == "rnd":
            reward_rms.update(np.stack(intrinsic_reward_buffer))
            obs_rms.update(np.stack(observation_buffer))

        last_10_scores.append(score)
        per_episode_scores.append(score)

        print('\rEpisode {}\tAverage Score: {:.2f}\tEpsilon: {:.2f}'.format(episode, np.mean(last_10_scores), agent.epsilon), end="")
        if episode % 100 == 0:
            print('\rEpisode {}\tAverage Score: {:.2f}\tEpsilon: {:.2f}'.format(episode, np.mean(last_10_scores), agent.epsilon))
    return per_episode_scores, state_ri_buffer

def test_forward_pass(dqn_agent, mdp):
    # load the weights from file
    mdp.reset()
    state = deepcopy(mdp.init_state)
    overall_reward = 0.
    mdp.render = True

    while not state.is_terminal():
        action = dqn_agent.act(state.features(), train_mode=False)
        reward, next_state = mdp.execute_agent_action(action)
        overall_reward += reward
        state = next_state

    mdp.render = False
    return overall_reward

def save_all_scores(experiment_name, log_dir, seed, scores):
    print("\rSaving training and validation scores..")
    training_scores_file_name = "{}_{}_training_scores.pkl".format(experiment_name, seed)

    if log_dir:
        training_scores_file_name = os.path.join(log_dir, training_scores_file_name)

    with open(training_scores_file_name, "wb+") as _f:
        pickle.dump(scores, _f)

def create_log_dir(experiment_name):
    path = os.path.join(os.getcwd(), experiment_name)
    try:
        os.mkdir(path)
    except OSError:
        print("Creation of the directory %s failed" % path)
    else:
        print("Successfully created the directory %s " % path)
    return path

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment_name", type=str, help="Experiment Name")
    parser.add_argument("--seed", type=int, help="Random seed for this run (default=0)", default=0)
    parser.add_argument("--episodes", type=int, help="# episodes", default=NUM_EPISODES)
    parser.add_argument("--steps", type=int, help="# steps", default=NUM_STEPS)
    parser.add_argument("--render", type=bool, help="Render the mdp env", default=False)
    parser.add_argument("--pixel_observation", type=bool, help="Images / Dense input", default=False)
    parser.add_argument("--exploration_method", type=str, default="eps-greedy")
    parser.add_argument("--eval_eps", type=float, default=0.05)
    parser.add_argument("--tensor_log", type=bool, default=False)
    args = parser.parse_args()

    logdir = create_log_dir(args.experiment_name)
    learning_rate = 1e-3 # 0.00025 for pong

    overall_mdp = GymMDP(env_name="MontezumaRevengeNoFrameskip-v4", pixel_observation=args.pixel_observation, render=args.render,
                         clip_rewards=False, term_func=None, seed=args.seed)
    # overall_mdp = LunarLanderMDP(render=args.render, seed=args.seed)

    state_dim = overall_mdp.env.observation_space.shape if args.pixel_observation else overall_mdp.env.observation_space.shape[0]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ddqn_agent = DQNAgent(state_size=state_dim, action_size=len(overall_mdp.actions),
                          trained_options=[], seed=args.seed, device=device,
                          name="GlobalDDQN", lr=learning_rate, tensor_log=args.tensor_log, use_double_dqn=True,
                          exploration_method=args.exploration_method, pixel_observation=args.pixel_observation,
                          evaluation_epsilon=args.eval_eps)
    ddqn_episode_scores, s_ri_buffer = train(ddqn_agent, overall_mdp, args.episodes, args.steps)
    save_all_scores(args.experiment_name, logdir, args.seed, ddqn_episode_scores)
