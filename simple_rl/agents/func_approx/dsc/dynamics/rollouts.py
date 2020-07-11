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

def collect_random_rollout(epochs=10):
    """
    TODO: add noise to inputs
    """
    mdp = D4RLPointMazeMDP(seed=0, render=False)
    # test set change start point
    
    # collect data
    buffer = []
    
    for episode in range(epochs):
      mdp.reset()
      for step in range(1000):
        s = deepcopy(mdp.cur_state)
        a = mdp.sample_random_action()
        r, sp = mdp.execute_agent_action(a)
        
        x, y, has_key, theta, xdot, ydot, thetadot = s
        x2, y2, has_key2, theta2, xdot2, ydot2, thetadot2 = sp
        
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
    
    state_scaler = preprocessing.StandardScaler().fit(states)
    action_scaler = preprocessing.StandardScaler().fit(actions)
    states_p_scaler = preprocessing.StandardScaler().fit(states_p)
    
    states = state_scaler.transform(states)
    actions = action_scaler.transform(actions)
    states_p = states_p_scaler.transform(states_p)
    
    dataset = RolloutDataset(states, actions, states_p)
    
    return dataset
