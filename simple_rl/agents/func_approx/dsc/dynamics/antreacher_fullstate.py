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
            
            s_pos = np.array(s)
            sp_pos = np.array(sp)
            data = (s_pos, a, sp_pos - s_pos)
        
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
            
            s_pos = np.array(s)
            sp_pos = np.array(sp)
            data = (s_pos, a, sp_pos - s_pos)
        
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
        self._roundup()
        
        states_p = np.nan_to_num((states_p - self.mean_z) / self.std_z)
        
        dataset = RolloutDataset(states, actions, states_p)
        self.train = dataset
        
    def _roundup(self, c=1e-5):
        """
        If any standarization variable is 0, add some constant to prevent NaN
        """
        self.mean_x[self.mean_x == 0] = c
        self.mean_y[self.mean_y == 0] = c
        self.mean_z[self.mean_z == 0] = c
        self.std_x[self.std_x == 0] = c
        self.std_y[self.std_y == 0] = c
        self.std_z[self.std_z == 0] = c
        
    def _add_noise(self, data_inp, noiseToSignal=0.01):
        """
        Directly copied from Nagabandi code base
        """
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

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
collector = RandomRolloutCollector(10)
train = collector.get_train_data()
test = collector.construct_data(epochs=10, random_position=False)
training_gen = DataLoader(train, batch_size=512, shuffle=True)
testing_gen = DataLoader(test, batch_size=512, shuffle=False)

model = DynamicsModel(31, 8, *collector.get_standardization_vars())
model.to(device)
loss_function = nn.MSELoss().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

train_list = []

for epoch in range(1, 100):
    total_loss = 0
    test_loss = 0
    for states, actions, states_p in training_gen:
        states = states.to(device).float()
        actions = actions.to(device).float()
        states_p = states_p.to(device).float()
        
        optimizer.zero_grad()
        p = model.forward(states, actions)
        loss = loss_function(p, states_p)
        loss.backward()
        optimizer.step()
        
        total_loss += loss_function(*model.compare_state(states, actions, states_p))
        
    for states, actions, states_p in testing_gen:
        states = states.to(device).float()
        actions = actions.to(device).float()
        states_p = states_p.to(device).float()
        
        test_loss += loss_function(*model.compare_state(states, actions, states_p))
    
    train_list.append(total_loss.cpu().detach().item())
    
    
    print("Epoch {} total loss {}, test_loss {}".format(epoch, total_loss, test_loss))
    
######################## MPC stuff here #######################################
# TODO plot model horizon errors

mdp = AntReacherMDP()
mdp.reset()

thres = 0.6
goal = mdp.sample_random_state()
# goal_x = goal[0]
# goal_y = goal[1]

goals = []
step_200 = []

for _ in range(20):
    mdp.reset()
    goal = mdp.sample_random_state()
    goals.append(goal)
    goal_x = goal[0]
    goal_y = goal[1]

    # setup starting variables
    trajectory = []
    s = mdp.cur_state
    trajectory.append(s.position)

    s = mdp.cur_state
    s = mdp.cur_state
    sx, sy = s.position

    num_rollouts = 14000
    num_steps = 7

    results = np.zeros((num_rollouts, num_steps))

    # mpc rollout
    with torch.no_grad():
        while (abs(sx - goal_x) >= thres or abs(sy - goal_y) >= thres) and len(trajectory) < 201:
            # sample actions for all steps
            np_actions = np.zeros((num_rollouts, num_steps, 8))
            np_states = np.repeat(np.array([s]), num_rollouts, axis=0)
            for i in range(num_rollouts):
                for j in range(num_steps):
                    np_actions[i,j,:] = mdp.sample_random_action()
            
            # compute next states for each step
            for j in range(num_steps):
                actions = np_actions[:,j,:]
                states_t = torch.from_numpy(np_states)
                actions_t= torch.from_numpy(actions)
                # transfer to gpu
                states_t = states_t.to(device)
                actions_t = actions_t.to(device)

                pred = model.predict_next_state(states_t.float(), actions_t.float())
                np_states = pred.cpu().numpy()
                # update results with whatever distance metric
                results[:,j] = (goal_x - np_states[:,0]) ** 2 + (goal_y - np_states[:,1]) ** 2
            
            # choose next action to execute
            gammas = np.power(0.95 * np.ones(num_steps), np.arange(0, num_steps))
            summed_results = np.sum(results * gammas, axis=1)
            index = np.argmin(summed_results) # retrieve action with least trajectory distance to goal
            action = np_actions[index,0,:] # grab action corresponding to least distance
            
            # execute action in mdp
            mdp.execute_agent_action(action)
            
            # update s for new state
            s = mdp.cur_state
            sx, sy = s.position
            
            trajectory.append(s.position)
            
            print("action taken!, {}".format(len(trajectory)))
        print("done!")
    
    step_200.append(trajectory)

for i in range(len(goals)):
    goal_x, goal_y = goals[i]
    traj = step_200[i]
    t = np.array(traj)
    plt.figure()
    plt.scatter(goal_x, goal_y, color='green')
    plt.plot(t[:,0], t[:,1])
    plt.savefig('plots/{}'.format(i))

# setup starting variables
trajectory = []
model_trajectory = []
s = mdp.cur_state
trajectory.append(s.position)
model_trajectory.append(s.position)

s = mdp.cur_state
s = mdp.cur_state
sx, sy = s.position

num_rollouts = 14000
num_steps = 7

results = np.zeros((num_rollouts, num_steps))

# mpc rollout
with torch.no_grad():
    while abs(sx - goal_x) >= thres or abs(sy - goal_y) >= thres:
        # sample actions for all steps
        np_actions = np.zeros((num_rollouts, num_steps, 8))
        np_states = np.repeat(np.array([s]), num_rollouts, axis=0)
        for i in range(num_rollouts):
            for j in range(num_steps):
                np_actions[i,j,:] = mdp.sample_random_action()
        
        # compute next states for each step
        for j in range(num_steps):
            actions = np_actions[:,j,:]
            states_t = torch.from_numpy(np_states)
            actions_t= torch.from_numpy(actions)
            # transfer to gpu
            states_t = states_t.to(device)
            actions_t = actions_t.to(device)

            pred = model.predict_next_state(states_t.float(), actions_t.float())
            np_states = pred.cpu().numpy()
            # update results with whatever distance metric
            results[:,j] = (goal_x - np_states[:,0]) ** 2 + (goal_y - np_states[:,1]) ** 2
        
        # choose next action to execute
        gammas = np.power(0.95 * np.ones(num_steps), np.arange(0, num_steps))
        summed_results = np.sum(results * gammas, axis=1)
        index = np.argmin(summed_results) # retrieve action with least trajectory distance to goal
        action = np_actions[index,0,:] # grab action corresponding to least distance
        
        # execute action in mdp
        mdp.execute_agent_action(action)
        
        # update s for new state
        s = mdp.cur_state
        sx, sy = s.position
        
        trajectory.append(s.position)
        
        print("action taken!, {}".format(len(trajectory)))
    print("done!")
    
# plot results
traj = np.array(trajectory)
traj2 = np.array(model_trajectory)

plt.figure()
plt.plot(traj[:,0], traj[:,1])
plt.plot(traj2[:,0], traj2[:,1], color='green')
plt.show()
    
# TODO plot metrics
    # plot original distance to goal
        # based on loc of goal, how many steps it take MPC to reach goal (bar plot, mean, std) (5x5)
import pickle
from simple_rl.agents.func_approx.dsc.dynamics.plot_utils import num_steps_per_goal
goals = []
for _ in range(30):
    goals.append(mdp.sample_random_state())
res = num_steps_per_goal(model, mdp, goals, thres, 14000, 7, runs_per=5)
pickle.dump(res, open("res_goal.pkl", "wb"))

# model prediction error on trajectory (x = horizon, y = error of x,y pos)
from simple_rl.agents.func_approx.dsc.dynamics.plot_utils import model_trajectory_prediction_error

model_traj, actual_traj = model_trajectory_prediction_error(model, mdp, 100, 100)
diff_traj = model_traj - actual_traj
def loss(a):
    return np.sqrt(a[0] ** 2 + a[1] ** 2)
test = np.apply_along_axis(loss, 2, diff_traj)
test = np.mean(test, axis=0)
%matplotlib qt
plt.figure()
plt.xlabel("horizon step")
plt.ylabel("MSE loss")
plt.xticks(np.arange(0, 100, 1))
plt.plot(test)

# plot number of steps to reach goal (x = length of mpc horizon, or number of action seq, y = number of steps it takes)
import pickle
from simple_rl.agents.func_approx.dsc.dynamics.plot_utils import num_steps_per_horizon
res = num_steps_per_horizon(model, mdp, goal_x, goal_y, thres, 14000, 3, 15, runs_per=10)
pickle.dump(res, open("res.pkl", "wb"))

# dropout uncertainty plotting
# 0.00022379143089372574 for no monte carlo
# 0.00010050453958282986 for with monte carlo
#with torch.no_grad():
#    mdp.reset()
#    s = mdp.cur_state
#    a = mdp.sample_random_action()
#    state = torch.from_numpy(np.array([s])).cuda()
#    action = torch.from_numpy(np.array([a])).cuda()
#    # model_traj = []
#    next_state = model.predict_next_state(state.float(), action.float())
#    n_s = next_state.cpu().squeeze().numpy()[:2]
#    # for _ in range(100):
#    #     next_state = model.predict_next_state(state.float(), action.float())
#    #     model_traj.append(next_state.cpu().squeeze().numpy())
#    # traj = np.array(model_traj)
#    plt.figure()
#    # plt.scatter(traj[:,0], traj[:,1])
#    plt.scatter(n_s[0], n_s[1])
#    mdp.execute_agent_action(a)
#    x, y = mdp.cur_state.position
#    plt.scatter(x, y, color='red')
#    # _t = np.median(traj, axis=0)
#    # plt.scatter(_t[0], _t[1], color='orange')
#    
#    other_error += (x - n_s[0]) ** 2 + (y - n_s[1]) ** 2
#    other_count += 1
#
#(other_error - error) / error
#other_error / error
#

