import numpy as np
import random
from collections import namedtuple, deque
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
import ipdb
from copy import deepcopy
import shutil
import os
import time
import argparse
import pickle
from collections import defaultdict
from sklearn import svm

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
from simple_rl.tasks.gym.wrappers import LazyFrames
from simple_rl.agents.func_approx.exploration.optimism.discrete.DiscreteCountBasedExplorationClass import DiscreteCountBasedExploration
from simple_rl.agents.func_approx.exploration.optimism.latent.LatentCountExplorationBonus import LatentCountExplorationBonus


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
                 gradient_clip=None, evaluation_epsilon=0.05, exploration_method="eps-decay",
                 pixel_observation=False, writer=None, experiment_name="", bonus_scaling_term="sqrt",
                 lam_scaling_term="fit", novelty_during_regression=True, normalize_states=False, optimization_quantity="",
                 phi_type="function", counting_epsilon=0.1, bonus_from_position=False):

        self.state_size = state_size
        self.action_size = action_size
        self.trained_options = trained_options
        self.learning_rate = lr
        self.use_ddqn = use_double_dqn
        self.gamma = gamma
        self.loss_function = loss_function
        self.gradient_clip = gradient_clip
        self.evaluation_epsilon = evaluation_epsilon
        assert exploration_method in ("eps-decay", "eps-const", "count-phi", "count-gt", "oc-svm"), exploration_method
        self.exploration_method = exploration_method
        self.pixel_observation = pixel_observation
        self.seed = random.seed(seed)
        np.random.seed(seed)
        self.tensor_log = tensor_log
        self.device = device
        self.bonus_scaling_term = bonus_scaling_term
        self.novelty_during_regression = novelty_during_regression
        assert phi_type in ("function", "raw")
        self.phi_type = phi_type
        self.counting_epsilon = counting_epsilon
        self.actions = list(range(self.action_size))
        self.bonus_from_position = bonus_from_position

        # Q-Network
        if pixel_observation:
            self.policy_network = ConvQNetwork(in_channels=state_size[0], n_actions=action_size).to(self.device)
            self.target_network = ConvQNetwork(in_channels=state_size[0], n_actions=action_size).to(self.device)
        else:
            self.policy_network = DenseQNetwork(state_size, action_size, seed, fc1_units=32, fc2_units=16).to(self.device)
            self.target_network = DenseQNetwork(state_size, action_size, seed, fc1_units=32, fc2_units=16).to(self.device)

        self.optimizer = optim.Adam(self.policy_network.parameters(), lr=lr)

        # Replay memory
        self.replay_buffer = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, seed, self.device, pixel_observation)
        # Initialize time step (for updating every UPDATE_EVERY steps)
        self.t_step = 0

        self.writer = None

        if self.tensor_log:
            if writer:
                self.writer = writer
            else:
                self.writer = SummaryWriter("runs/{}".format(experiment_name))


        # Epsilon strategy
        if exploration_method == "eps-decay":
            self.epsilon_schedule = GlobalEpsilonSchedule(eps_start, evaluation_epsilon) if "global" in name.lower() else OptionEpsilonSchedule(eps_start)
            self.epsilon = eps_start
        elif exploration_method == "eps-const":
            self.epsilon_schedule = ConstantEpsilonSchedule(evaluation_epsilon)
            self.epsilon = evaluation_epsilon
        elif exploration_method == "count-gt":
            self.epsilon_schedule = ConstantEpsilonSchedule(0)
            self.epsilon = 0
            self.novelty_tracker = DiscreteCountBasedExploration(action_size)
        elif exploration_method == "count-phi":
            self.epsilon_schedule = ConstantEpsilonSchedule(0)
            self.epsilon = 0.

            novelty_tracker_state_dim = 2 if bonus_from_position else state_size
            novelty_use_pixels = False if bonus_from_position else self.pixel_observation

            self.novelty_tracker = LatentCountExplorationBonus(state_dim=novelty_tracker_state_dim,
                                                               action_dim=action_size,
                                                               experiment_name=experiment_name,
                                                               pixel_observation=novelty_use_pixels,
                                                               normalize_states=normalize_states, writer=self.writer,
                                                               bonus_scaling_term=bonus_scaling_term,
                                                               lam_scaling_term=lam_scaling_term,
                                                               optimization_quantity=optimization_quantity,
                                                               num_frames=1,
                                                               device=device,
                                                               phi_type=phi_type,
                                                               epsilon=counting_epsilon)

        elif exploration_method == "oc-svm":
            self.visited_state_action_pairs = defaultdict(lambda : [])
            self.one_class_classifiers = [None] * len(self.actions)
            self.epsilon_schedule = ConstantEpsilonSchedule(0)
            self.epsilon = 0.
        else:
            raise NotImplementedError("{} not implemented", exploration_method)

        self.num_executions = 0 # Number of times act() is called (used for eps-decay)

        # Debugging attributes
        self.num_updates = 0
        self.num_epsilon_updates = 0


        print("\nCreating {} with lr={} and ddqn={} and buffer_sz={}\n".format(name, self.learning_rate,
                                                                               self.use_ddqn, BUFFER_SIZE))

        Agent.__init__(self, name, range(action_size), self.gamma)

    def act(self, state, position, train_mode=True, use_novelty=False):
        """
        Interface to the DQN agent: state can be output of env.step() and returned action can be input into next step().
        Args:
            state (np.array): numpy array state from Gym env
            position (np.array)
            train_mode (bool): if training, use the internal epsilon. If evaluating, set epsilon to min epsilon
            use_novelty (bool): Whether to use exploration bonuses during action-selection

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

        action_values = action_values.cpu().data.numpy()
        # Epsilon-greedy action selection
        if random.random() > epsilon:
            if use_novelty and self.exploration_method == "count-phi":
                bonus_input = position if self.bonus_from_position else state.cpu().numpy().squeeze(0)
                bonus = self.novelty_tracker.get_exploration_bonus(bonus_input)
                assert bonus.shape == action_values.shape
                return np.argmax(action_values + bonus)
            elif use_novelty and self.exploration_method == "count-gt":
                bonus = self.novelty_tracker.get_exploration_bonus(position)
                assert bonus.shape == action_values.shape
                return np.argmax(action_values + bonus)
            elif use_novelty and self.exploration_method == "oc-svm":
                Q_MAX = 100.
                novelty = np.array([clf.predict(position.reshape(1, -1))[0] if clf is not None else -1 for clf in self.one_class_classifiers])
                action_values = np.where(novelty == 1, action_values, Q_MAX)
            return np.argmax(action_values)

        return random.randint(0, self.action_size - 1)

    def get_best_actions_batched(self, states, next_positions):
        q_values = self.get_batched_qvalues(states, next_positions)
        return torch.argmax(q_values, dim=1)

    def get_value(self, state):
        action_values = self.get_qvalues(state)
        return np.max(action_values.cpu().data.numpy())

    def get_batched_values(self, states):

        states = torch.from_numpy(states).float().to(self.device)

        self.policy_network.eval()
        with torch.no_grad():
            action_values = self.policy_network(states)
        self.policy_network.train()

        action_values = action_values.cpu().numpy()
        values = np.max(action_values, axis=1)

        assert values.shape == (states.shape[0],), values.shape

        return values

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

    def get_batched_qvalues(self, states, positions):
        """
        Q-values corresponding to `states` for all ** permissible ** actions/options given `states`.
        Args:
            states (torch.tensor) of shape (64 x 4)
            positions (torch.tensor) of shape (64 x 2)

        Returns:
            qvalues (torch.tensor) of shape (64 x |A|)
        """
        self.policy_network.eval()
        with torch.no_grad():
            action_values = self.policy_network(states)
        self.policy_network.train()

        return action_values

    def _add_transition(self, state, position, action, reward, next_state, next_position, done, num_steps=1):
        """
        Does the adding to both the replay buffer and the count buffer.

        Args:
            state (np.array): state of the underlying gym env
            position (np.array)
            action (int)
            reward (float)
            next_state (np.array)
            next_position (np.array)
            done (bool): is_terminal
            num_steps (int): number of steps taken by the option to terminate

        """
        self.replay_buffer.add(state, position, action, reward, next_state, next_position, done, num_steps)
        if self.exploration_method == "count-gt":
            assert position.shape == (2,)
            assert isinstance(position, np.ndarray), position
            self.novelty_tracker.add_transition(tuple(position), action)
        if self.exploration_method == "count-phi":
            bonus_input = np.array(position) if self.bonus_from_position else np.array(state)
            next_bonus_input = np.array(next_position) if self.bonus_from_position else np.array(next_state)
            assert bonus_input.shape in ((2,), self.state_size)
            self.novelty_tracker.add_transition(bonus_input, action, next_bonus_input)
        if self.exploration_method == "oc-svm":
            assert position.shape == (2,)
            assert isinstance(position, np.ndarray), position
            self.visited_state_action_pairs[action].append(tuple(position))

    def _get_q_targets(self, next_states, next_positions):
        """
        Only ever used in _learn, in order to get targets that may or may not have an exploration bonus.
        Args:
            next_states (torch.Tensor): next states.
            next_positions (torch.Tensor): x, y positions corresponding to `next_states`

        """
        Q_targets_next = self.target_network(next_states).detach()
        if self.exploration_method == "count-gt" and self.novelty_during_regression:
            next_positions_array = next_positions.cpu().numpy()
            bonus_array = self.novelty_tracker.get_batched_exploration_bonus(next_positions_array)
            bonus_tensor = torch.from_numpy(bonus_array).float().to(self.device)
            assert Q_targets_next.shape == bonus_tensor.shape
            return Q_targets_next + bonus_tensor
        if self.exploration_method == "count-phi" and self.novelty_during_regression:
            next_bonus_input = next_positions if self.bonus_from_position else next_states
            next_bonus_input = next_bonus_input.cpu().numpy()
            bonus_array = self.novelty_tracker.get_batched_exploration_bonus(next_bonus_input)
            bonus_tensor = torch.from_numpy(bonus_array).float().to(self.device)
            assert Q_targets_next.shape == bonus_tensor.shape
            return Q_targets_next + bonus_tensor
        if self.exploration_method == "oc-svm" and self.novelty_during_regression:
            Q_MAX = 100.
            np_next_states = next_positions.cpu().numpy()
            np_Q_next = Q_targets_next.cpu().numpy()

            for action in self.actions:
                if self.one_class_classifiers[action] is not None:
                    X = np_next_states[:, :2]
                    Y = self.one_class_classifiers[action].predict(X)
                    np_Q_next[np.where(Y != 1)[0], action] = Q_MAX
                else:
                    np_Q_next[:, action] = Q_MAX

            return torch.from_numpy(np_Q_next).to(self.device)

        return Q_targets_next

    def step(self, state, position, action, reward, next_state, next_position, done, num_steps=1):
        """
        Interface method to perform 1 step of learning/optimization during training.
        Args:
            state (np.array): state of the underlying gym env
            position (np.array)
            action (int)
            reward (float)
            next_state (np.array)
            next_position (np.array)
            done (bool): is_terminal
            num_steps (int): number of steps taken by the option to terminate
        """
        # Save experience in replay memory
        self._add_transition(state, position, action, reward, next_state, next_position, done, num_steps)

        # Learn every UPDATE_EVERY time steps.
        self.t_step = (self.t_step + 1) % UPDATE_EVERY
        if self.t_step == 0:
            # If enough samples are available in memory, get random subset and learn
            if len(self.replay_buffer) > BATCH_SIZE:
                experiences = self.replay_buffer.sample(batch_size=BATCH_SIZE)
                self._learn(experiences, self.gamma)
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
        # print(f"gamma: {gamma}")
        states, positions, actions, rewards, next_states, next_positions, dones, steps = experiences

        # Get max predicted Q values (for next states) from target model
        if self.use_ddqn:
            raise NotImplementedError("Not implemented _get_q_targets for DDQN yet.")
            self.policy_network.eval()
            with torch.no_grad():
                selected_actions = self.policy_network(next_states).argmax(dim=1).unsqueeze(1)
            self.policy_network.train()

            Q_targets_next = self.target_network(next_states).detach().gather(1, selected_actions)
        else:
            Q_targets_next = self._get_q_targets(next_states, next_positions).max(1)[0].unsqueeze(1)


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

            if self.exploration_method == "count-phi":
                bonus_inputs = positions if self.bonus_from_position else states
                self.writer.add_scalar(
                    "DQN-AverageBonus",
                    # self.novelty_tracker.get_batched_exploration_bonus(states.cpu().numpy()).mean().item(),
                    self.novelty_tracker.get_batched_exploration_bonus(bonus_inputs.cpu().numpy()).mean().item(),
                    self.num_updates)

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

    def train_novelty_detector(self, mode="entire"):
        if self.exploration_method == "oc-svm":
            for action in self.actions:
                visited_states = np.array(list(self.visited_state_action_pairs[action]))
                if len(visited_states) > 0:
                    self.one_class_classifiers[action] = svm.OneClassSVM(kernel="rbf", gamma=50., nu=0.01)
                    X = visited_states[:, :2]
                    self.one_class_classifiers[action].fit(X)
        else:
            epochs = -1 if self.novelty_tracker.counting_space.optimization_quantity in ("chunked-bonus", "chunked-log") else 50
            self.novelty_tracker.train(epochs=epochs, mode=mode)

def train(agent, mdp, episodes, steps):
    per_episode_scores = []
    last_10_scores = deque(maxlen=10)
    iteration_counter = 0

    for episode in range(episodes):
        mdp.reset()
        state = deepcopy(mdp.init_state)

        score = 0.
        for _ in range(steps):
            iteration_counter += 1

            position = mdp.get_position()
            action = agent.act(state.features(), train_mode=True,
                               position=position, use_novelty=args.use_bonus_during_action_selection)
            reward, next_state = mdp.execute_agent_action(action)
            next_position = mdp.get_position()

            agent.step(state.features(), position, action, reward, next_state.features(), next_position, next_state.is_terminal(), num_steps=1)
            agent.update_epsilon()
            state = next_state
            score += reward
            if agent.tensor_log:
                agent.writer.add_scalar("Score", score, iteration_counter)

            game_over = mdp.game_over if hasattr(mdp, 'game_over') else False
            if state.is_terminal() or game_over:
                break

        agent.train_novelty_detector()

        last_10_scores.append(score)
        per_episode_scores.append(score)

        print('\rEpisode {}\tAverage Score: {:.2f}\tEpsilon: {:.2f}'.format(episode, np.mean(last_10_scores), agent.epsilon))
    return per_episode_scores

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
    print("\r\nSaving training and validation scores..")
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
    parser.add_argument("--exploration_method", type=str, default="eps-decay")
    parser.add_argument("--use_bonus_during_action_selection", type=bool, default=False)
    parser.add_argument("--eval_eps", type=float, default=0.05)
    parser.add_argument("--tensor_log", type=bool, default=False)
    args = parser.parse_args()

    logdir = create_log_dir(args.experiment_name)
    learning_rate = 1e-3 # 0.00025 for pong

    from simple_rl.tasks.gridworld.VisualGridWorldMDPClass import VisualGridWorldMDP
    # overall_mdp = GymMDP(env_name="MontezumaRevengeNoFrameskip-v4", pixel_observation=args.pixel_observation, render=args.render,
    #                      clip_rewards=False, term_func=None, seed=args.seed)
    # overall_mdp = LunarLanderMDP(render=args.render, seed=args.seed)
    overall_mdp = VisualGridWorldMDP(args.pixel_observation, False, 0.1, args.seed)

    # state_dim = overall_mdp.env.observation_space.shape if args.pixel_observation else overall_mdp.env.observation_space.shape[0]
    state_dim = overall_mdp.state_dim
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ddqn_agent = DQNAgent(state_size=state_dim, action_size=len(overall_mdp.actions),
                          trained_options=[], seed=args.seed, device=device,
                          name="GlobalDDQN", lr=learning_rate, tensor_log=args.tensor_log, use_double_dqn=False,
                          exploration_method=args.exploration_method, pixel_observation=args.pixel_observation,
                          evaluation_epsilon=args.eval_eps)
    ddqn_episode_scores = train(ddqn_agent, overall_mdp, args.episodes, args.steps)
    save_all_scores(args.experiment_name, logdir, args.seed, ddqn_episode_scores)
