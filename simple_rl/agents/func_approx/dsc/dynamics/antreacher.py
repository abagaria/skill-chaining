import random
import numpy as np
import matplotlib.pyplot as plt

from copy import deepcopy
from torch.utils.data import Dataset
from simple_rl.tasks.ant_reacher.AntReacherMDPClass import AntReacherMDP

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
         self.mdp = AntReacherMDP()
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
            
            data = (s.position, a, sp.position - s.position)
        
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

            data = (s.position, a, sp.position - s.position)
        
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
        
        # uncomment if want noise to be added
        # states = self._add_noise(np.array(states))
        # states_p = self._add_noise(np.array(states_p))
        
        dataset = RolloutDataset(states, actions, (states_p - self.mean_z) / self.std_z)
        self.train = dataset
        
    def _add_noise(self, data_inp, noiseToSignal=0.01):
        data = deepcopy(data_inp)
        mean_data = np.mean(data, axis = 0)
        std_of_noise = mean_data*noiseToSignal
        for j in range(mean_data.shape[0]):
            if(std_of_noise[j]>0):
                data[:,j] = np.copy(data[:,j]+np.random.normal(0, np.absolute(std_of_noise[j]), (data.shape[0],)))
        return data
        
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from simple_rl.agents.func_approx.dsc.dynamics.dynamics_model import DynamicsModel

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
collector = RandomRolloutCollector(10)
train = collector.get_train_data()
training_gen = DataLoader(train, batch_size=512, shuffle=True)

model = DynamicsModel(2, 8, *collector.get_standardization_vars())
model.to(device)
loss_function = nn.MSELoss().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

train_list = []

for epoch in range(1, 100):
    total_loss = 0
    for states, actions, states_p in training_gen:
        states = states.to(device).float()
        actions = actions.to(device).float()
        states_p = states_p.to(device).float()
        
        optimizer.zero_grad()
        p, q = model.compare_state(states, actions, states_p)
        loss = loss_function(p, q)
        loss.backward()
        optimizer.step()
        
        total_loss += loss_function(*model.compare_state(states, actions, states_p))
    
    train_list.append(total_loss.cpu().detach().item())
    
    print("Epoch {} total loss {}".format(epoch, total_loss))
    
######################## MPC stuff here #######################################
    
def enable_dropout(model):
  for m in model.modules():
    if m.__class__.__name__.startswith('Dropout'):
      m.train()
model.eval()
enable_dropout(model)

# TODO plot model horizon errors

mdp = AntReacherMDP()
mdp.reset()

thres = 0.6
goal = mdp.sample_random_state()
goal_x = goal[0]
goal_y = goal[1]

# setup starting variables
trajectory = []
model_trajectory = []
s = mdp.cur_state
trajectory.append(s.position)
model_trajectory.append(s.position)

s = mdp.cur_state
s = mdp.cur_state
sx, sy = s.position

num_rollouts = 7000
num_steps = 15

results = np.zeros((num_rollouts, num_steps))

# dropout uncertainty plotting
with torch.no_grad():
    mdp.reset()
    s = mdp.cur_state
    a = mdp.sample_random_action()
    state = torch.from_numpy(np.array([s.position[:2]])).cuda()
    action = torch.from_numpy(np.array([a])).cuda()
    model_traj = []
    for _ in range(100):
        next_state = model.predict_next_state(state.float(), action.float())
        model_traj.append(next_state.cpu().squeeze().numpy())
    traj = np.array(model_traj)
    plt.figure()
    plt.scatter(traj[:,0], traj[:,1])
    mdp.execute_agent_action(a)
    x, y = mdp.cur_state.position
    plt.scatter(x, y, color='red')
    x, y = np.median(traj, axis=0)
    plt.scatter(x, y, color='orange')

# TODO plot what it looks like for the planner 
with torch.no_grad():
    mdp.reset()
    s = mdp.cur_state
    model_traj = [s.position]
    act_traj = [s.position]
    action_history = []
    state = torch.from_numpy(np.array([s.position[:2]])).cuda()
    for i in range(num_steps):
        a = mdp.sample_random_action() 
        action_history.append(a)
        action = torch.from_numpy(np.array([a])).cuda()
        next_state = model.predict_next_state(state.float(), action.float())
        state = next_state
        model_traj.append(state.cpu().squeeze().numpy())
    
    for action in action_history:
        mdp.execute_agent_action(action)
        s = deepcopy(mdp.cur_state)
        act_traj.append(s.position)
    
    traj = np.array(act_traj)
    traj2 = np.array(model_traj)

    plt.figure()
    plt.plot(traj[:,0], traj[:,1])
    plt.plot(traj2[:,0], traj2[:,1], color='green')
    plt.show()
    
with torch.no_grad():
    mdp.reset()
    s = mdp.cur_state
    model_traj = [s.position]
    act_traj = [s.position]
    state = torch.from_numpy(np.array([s.position[:2]])).cuda()
    for i in range(num_steps):
        a = mdp.sample_random_action() 
        action = torch.from_numpy(np.array([a])).cuda()
        next_state = model.predict_next_state(state.float(), action.float())
        model_traj.append(next_state.cpu().squeeze().numpy())

        mdp.execute_agent_action(a)
        s = mdp.cur_state
        state = torch.from_numpy(np.array([s.position[:2]])).cuda()
        act_traj.append(s.position)
    
    traj = np.array(act_traj)
    traj2 = np.array(model_traj)

    plt.figure()
    plt.plot(traj[:,0], traj[:,1])
    plt.plot(traj2[:,0], traj2[:,1], color='green')
    plt.show()

with torch.no_grad():
    while abs(sx - goal_x) >= thres or abs(sy - goal_y) >= thres:
        # sample actions for all steps
        np_actions = np.zeros((num_rollouts, num_steps, 8))
        np_states = np.repeat(s.position, num_rollouts).reshape(-1, 2)
        for i in range(num_rollouts):
            for j in range(num_steps):
                np_actions[i,j,:] = mdp.sample_random_action()
        
        # compute next states for each step
        for j in range(num_steps):
            actions = np_actions[:,j,:]
            states_t = torch.from_numpy(np_states)
            actions_t= torch.from_numpy(actions)
            # transfer to gpu
            states_t = states_t.cuda()
            actions_t = actions_t.cuda()
            # run inference
            pred = model.predict_next_state(states_t.float(), actions_t.float())
            # convert results to numpy for next iteration
            np_states = pred.cpu().numpy()
            # update results with whatever distance metric
            results[:,j] = (sx - np_states[:,0]) ** 2 + (sy - np_states[:,1]) ** 2
        
        # choose next action to execute
        gammas = np.power(0.9 * np.ones(num_steps), np.arange(0, num_steps))
#        summed_results = np.median(results, axis=1)
        summed_results = np.sum(results * gammas, axis=1)
        index = np.argmin(summed_results) # retreive action with least trajectory distance to goal
        action = np_actions[index,0,:] # grab action corresponding to least distance
        
        # save model predictions
        _s = torch.from_numpy(np.array([s.position[0:2]])).cuda()
        _a = torch.from_numpy(np.array([action])).cuda()
        pred = model.predict_next_state(_s.float(), _a.float())
        model_trajectory.append(pred.cpu().squeeze().numpy())
        
        # execute action in mdp
        mdp.execute_agent_action(action)
        # update s for new state
        s = mdp.cur_state
        sx, sy = s.position
        
        trajectory.append(s.position)
        
        print("action taken!")
    print("done!")

# TODO baseline random walk
# TODO plot various errors mean, median, gamma=0.95 etc to pick decision function
# TODO compare timesteps between here and nagabandi paper

# plot results
traj = np.array(trajectory)
traj2 = np.array(model_trajectory)

plt.figure()
plt.plot(traj[:,0], traj[:,1])
plt.plot(traj2[:,0], traj2[:,1], color='green')
plt.show()
