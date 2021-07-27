import os
import ipdb
import pickle

import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from tqdm import tqdm
from copy import deepcopy
from torch.optim import Adam
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from simple_rl.agents.func_approx.dsc.dynamics.dynamics_model import DynamicsModel
from simple_rl.agents.func_approx.dsc.dynamics.replay_buffer import ReplayBuffer
from simple_rl.mdp.GoalDirectedMDPClass import GoalDirectedMDP
from simple_rl.agents.func_approx.rnd.RNDRewardLearner import RND


class MPCRND:
    def __init__(self, mdp, state_size, action_size, device, multithread=False):
        assert isinstance(mdp, GoalDirectedMDP)

        self.mdp = mdp
        self.device = device
        self.state_size = state_size
        self.action_size = action_size

        self.is_trained = False
        self.trained_options = []
        self.gamma = 0.95

        self.model = DynamicsModel(self.state_size, self.action_size, self.device)
        self.model.to(self.device)
        
        self.replay_buffer = ReplayBuffer(obs_dim=state_size, act_dim=action_size, size=int(3e5))

        if multithread:
            self.workers = os.cpu_count() - 2
        else:
            self.workers = 0

        # RND Parameters
        self.rnd = RND(state_size,
                       lr=1e-3,
                       n_epochs=1,
                       batch_size=1000,
                       device=device)

    def load_data(self):
        self.dataset = self._preprocess_data()
        self.model.set_standardization_vars(*self._get_standardization_vars())

    def train(self, epochs=100, batch_size=512):
        self.is_trained = True

        training_gen = DataLoader(self.dataset, batch_size=batch_size, num_workers=self.workers, shuffle=True,  pin_memory=True)
        loss_function = nn.MSELoss().to(self.device)
        optimizer = Adam(self.model.parameters(), lr=1e-3)
        
        for epoch in tqdm(range(epochs), desc=f'Training MPC model on {self.replay_buffer.size} points'):
            for states, actions, states_p in training_gen:
                states = states.to(self.device).float()
                actions = actions.to(self.device).float()
                states_p = states_p.to(self.device).float()
            
                optimizer.zero_grad()
                p = self.model.forward(states, actions)
                loss = loss_function(p, states_p)
                loss.backward()
                optimizer.step()

    def rollout(self, mdp, num_rollouts, num_steps, max_steps):
        s = deepcopy(mdp.cur_state)

        for steps_taken in range(num_steps):
            action = self.act(s, num_rollouts=num_rollouts, num_steps=num_steps)
        
            # execute action in mdp
            mdp.execute_agent_action(action)

            # retrieve current state
            s = deepcopy(mdp.cur_state)

        return deepcopy(mdp.cur_state), steps_taken

    def simulate(self, s, num_rollouts=14000, num_steps=7):
        """ Perform N simulations of length H. """

        torch_actions = 2 * torch.rand((num_rollouts, num_steps, self.action_size), device=self.device) - 1
        features = s if isinstance(s, np.ndarray) else s.features()
        torch_states = torch.tensor(features, device=self.device).repeat(num_rollouts, 1)
        costs = torch.zeros((num_rollouts, num_steps))

        for j in range(num_steps):
            actions = torch_actions[:, j, :]
            torch_states = self.model.predict_next_state(torch_states.float(), actions.float())
            costs[:, j] = self._get_costs(torch_states)

        return torch_actions.cpu().numpy(), costs.cpu().numpy()

    def act(self, s, num_rollouts=14000, num_steps=7):
        # sample actions for all steps
        actions, costs = self.simulate(s, num_rollouts, num_steps)

        # choose next action to execute
        gammas = np.power(self.gamma * np.ones(num_steps), np.arange(0, num_steps))
        cumulative_costs = np.sum(costs * gammas, axis=1)

        index = np.argmin(cumulative_costs) # retrieve action with least trajectory distance to goal
        action = actions[index, 0, :] # grab action corresponding to least distance
        return action

    def step(self, state, action, reward, next_state, done):
        self.replay_buffer.store(state, action, reward, next_state, done)
        self.rnd.update(next_state)

    def value_function(self, state, num_rollouts=14000, num_steps=7):
        """ Compute the value of `state` via simulation rollouts. """

        _, costs = self.simulate(state, num_rollouts, num_steps)
        rewards = -1. * costs
        gammas = np.power(self.gamma * np.ones(num_steps), np.arange(0, num_steps))
        cumulative_rewards = np.sum(rewards * gammas, axis=1)
        return cumulative_rewards.mean()

    def _preprocess_data(self):
        states = self.replay_buffer.obs_buf[:self.replay_buffer.size, :]
        actions = self.replay_buffer.act_buf[:self.replay_buffer.size, :]
        states_p = self.replay_buffer.obs2_buf[:self.replay_buffer.size, :]

        assert states.shape[1] == states_p.shape[1] == self.state_size, f"{states.shape, states_p.shape}"
        assert actions.shape[1] == self.action_size, f"{actions.shape}"

        states_delta = np.array(states_p) - np.array(states)
        
        self.mean_x = np.mean(states, axis=0)
        self.mean_y = np.mean(actions, axis=0)
        self.mean_z = np.mean(states_delta, axis=0)
        self.std_x = np.std(states - self.mean_x, axis=0)
        self.std_y = np.std(actions - self.mean_y, axis=0)
        self.std_z = np.std(states_delta - self.mean_z, axis=0)

        self._roundup()

        norm_states_delta = (states_delta - self.mean_z) / self.std_z

        dataset = RolloutDataset(states, actions, norm_states_delta)
        return dataset

    def _roundup(self, c=1e-5):
        """
        If any standarization variable is 0, add some constant to prevent NaN
        """
        self.std_x[self.std_x == 0] = c
        self.std_y[self.std_y == 0] = c
        self.std_z[self.std_z == 0] = c

    def _get_standardization_vars(self):
        return self.mean_x, self.mean_y, self.mean_z, self.std_x, self.std_y, self.std_z

    def save_model(self, path):
        if self.model is not None:
            state_dictionary = self.model.__getstate__()
            with open(path, 'wb') as f:
                pickle.dump(state_dictionary, f)
        else:
            print("no model has been trained yet!")

    def load_model(self, path):
        with open(path, 'rb') as f:
            state_dictionary = pickle.load(f)
        self.model = DynamicsModel(self.state_size, self.action_size, self.device)
        self.model.__setstate__(state_dictionary)

    # ------------------------------------------------------------
    # RND Methods
    # ------------------------------------------------------------

    def _get_costs(self, states):
        rewards = self.rnd.get_reward(states)
        costs = -1. * rewards

        assert states.shape[1] == self.mdp.state_space_size(), f"{states.shape}"
        assert rewards.shape == costs.shape, f"{rewards.shape, costs.shape}"
        assert rewards.shape[0] == states.shape[0], f"{rewards.shape, states.shape}"

        return costs

    def get_intrinsic_reward(self, next_state):
        """ Intrinsic reward computation for a single input state. """

        normalized_next_state = next_state[None, ...]
        intrinsic_reward = self.rnd.get_reward(normalized_next_state).item()
        return intrinsic_reward

    def batched_get_intrinsic_reward(self, states):
        assert isinstance(states, (np.ndarray, torch.tensor)), type(states)
        intrinsic_reward = self.rnd.get_reward(states).mean().item()
        return intrinsic_reward


class RolloutDataset(Dataset):
    def __init__(self, states, actions, states_p):
        self.states = states
        self.actions = actions
        self.states_p = states_p
    
    def __len__(self):
        return len(self.states)
    
    def __getitem__(self, idx):
        return self.states[idx], self.actions[idx], self.states_p[idx]
