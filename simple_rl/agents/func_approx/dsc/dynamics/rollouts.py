import random
import numpy as np

from copy import deepcopy
from sklearn import preprocessing
from torch.utils.data import Dataset
from simple_rl.tasks.d4rl_point_maze.D4RLPointMazeMDPClass import D4RLPointMazeMDP

class RolloutDataset(Dataset):
    def __init__(self, states, actions, states_p):
        self.states = states
        self.actions = actions
        self.states_p = states_p
    
    def __len__(self):
        return len(self.states)
    
    def __getitem__(self, idx):
        return self.states[idx], self.actions[idx], self.states_p[idx]
    
class RandomRolloutCollector:
    def __init__(self, epochs):
         self.mdp = D4RLPointMazeMDP(difficulty="medium")
         self._construct_train_set(epochs)
    
    def get_train_data(self):
        return self.train
    
    def construct_data(self, epochs, random_position):
        if random_position == True:
            x_low, y_low = self.mdp.get_x_y_low_lims()
            x_high, y_high = self.mdp.get_x_y_high_lims()
            x_pos = random.uniform(x_low, x_high)
            y_pos = random.uniform(y_low, y_high)
            self.mdp.set_xy((x_pos, y_pos))
        
        buffer = []
        for episode in range(epochs):
          self.mdp.reset()
          if random_position == True:
              self.mdp.set_xy((x_pos, y_pos))
    
          for step in range(1000):
            s = deepcopy(self.mdp.cur_state)
            a = self.mdp.sample_random_action()
            r, sp = self.mdp.execute_agent_action(a)
            
            x, y, theta, xdot, ydot, thetadot = s
            x2, y2, theta2, xdot2, ydot2, thetadot2 = sp
            
            np_s = np.array([x, y, theta, xdot, ydot, thetadot])
            np_sp = np.array([x2, y2, theta2, xdot2, ydot2, thetadot2])
            
            data = (np_s, a, np_sp - np_s)
        
            buffer.append(data)
            if sp.is_terminal():
                break
        
        # normalize data
        states = []
        actions = []
        states_p = []
        for state, action, state_p in buffer:
            states.append(state)
            actions.append(action)
            states_p.append(state_p)
        
        dataset = RolloutDataset(states, actions, (states_p - self.mean_z) / self.std_z)
        return dataset
    
    def get_standardization_vars(self):
        return self.mean_x, self.mean_y, self.mean_z, self.std_x, self.std_y, self.std_z
            
    def _construct_train_set(self, epochs):
        buffer = []
        for episode in range(epochs):
          self.mdp.reset()
          for step in range(1000):
            s = deepcopy(self.mdp.cur_state)
            a = self.mdp.sample_random_action()
            r, sp = self.mdp.execute_agent_action(a)
            
            x, y, theta, xdot, ydot, thetadot = s
            x2, y2, theta2, xdot2, ydot2, thetadot2 = sp
            
            np_s = np.array([x, y, theta, xdot, ydot, thetadot])
            np_sp = np.array([x2, y2, theta2, xdot2, ydot2, thetadot2])
            
            data = (np_s, a, np_sp - np_s)
        
            buffer.append(data)
            if sp.is_terminal():
                break
        
        # normalize data
        states = []
        actions = []
        states_p = []
        for state, action, state_p in buffer:
            states.append(state)
            actions.append(action)
            states_p.append(state_p)
            
        self.mean_x = np.mean(states, axis=0)
        self.mean_y = np.mean(actions, axis=0)
        self.mean_z = np.mean(states_p, axis=0)
        self.std_x = np.std(states - self.mean_x, axis=0)
        self.std_y = np.std(actions - self.mean_y, axis=0)
        self.std_z = np.std(states_p - self.mean_z, axis=0)
        
        dataset = RolloutDataset(states, actions, (states_p - self.mean_z) / self.std_z)
        self.train = dataset
        

def collect_random_rollout(epochs=10, random_position=False):
    """
    TODO: add noise to inputs
    """
    mdp = D4RLPointMazeMDP(seed=0, render=False, difficulty="medium")
    if random_position == True:
        x_low, y_low = mdp.get_x_y_low_lims()
        x_high, y_high = mdp.get_x_y_high_lims()
        x_pos = random.uniform(x_low, x_high)
        y_pos = random.uniform(y_low, y_high)
        mdp.set_xy((x_pos, y_pos))
    
    # collect data
    buffer = []
    
    for episode in range(10):
      mdp.reset()
#      if random_position == True:
#          mdp.set_xy((x_pos, y_pos))

      for step in range(1000):
        s = deepcopy(mdp.cur_state)
        a = mdp.sample_random_action()
        r, sp = mdp.execute_agent_action(a)
        
        x, y, theta, xdot, ydot, thetadot = s
        x2, y2, theta2, xdot2, ydot2, thetadot2 = sp
        
        np_s = np.array([x, y, theta, xdot, ydot, thetadot])
        np_sp = np.array([x2, y2, theta2, xdot2, ydot2, thetadot2])
        
        data = (np_s, a, np_sp - np_s)
    
        buffer.append(data)
        if sp.is_terminal():
            break
        
    # normalize data
    states = []
    actions = []
    states_p = []
    for state, action, state_p in buffer:
        states.append(state)
        actions.append(action)
        states_p.append(state_p)
        
    states_mean = np.mean(states, axis=0)
    states_std = np.std(states, axis=0)
    states = states - states_mean
    states = states / states_std
    
    np.mean(states, axis=0)
    np.std(states, axis=0)
    
    states_p_mean = np.mean(states_p, axis=0)
    states_p = states_p - states_p_mean
    states_p_std = np.std(states_p, axis=0)
    states_p = states_p / states_p_std
            
    state_scaler = preprocessing.StandardScaler().fit(states)
    action_scaler = preprocessing.StandardScaler().fit(actions)
    states_p_scaler = preprocessing.StandardScaler().fit(states_p)
    
    states_p_scaler.mean_
    states_p_scaler.var_
    
    states = state_scaler.transform(states)
    actions = action_scaler.transform(actions)
    states_p = states_p_scaler.transform(states_p)    
    dataset = RolloutDataset(states, actions, states_p)
        
    return dataset

if __name__== "__main__":
    mdp = D4RLPointMazeMDP(seed=0, render=False, difficulty="medium")
    mdp.reset()
    type(mdp.cur_state.data)
