# -*- coding: utf-8 -*-
import torch
import numpy as np
import matplotlib.pyplot as plt

from copy import deepcopy
from simple_rl.tasks.d4rl_point_maze.D4RLPointMazeMDPClass import D4RLPointMazeMDP
from simple_rl.agents.func_approx.dsc.dynamics.dynamics_model import DynamicsModel

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = torch.load("./model.pth")
model.to(device)

mdp = D4RLPointMazeMDP(difficulty="medium", render=False)
mdp.reset()

thres = 0.6
goal = mdp.sample_random_state()
goal_x = goal[0]
goal_y = goal[1]

# setup starting variables
# TODO add model predicting trajectory
# TODO try on ant reacher
trajectory = []
model_trajectory = []
s = mdp.cur_state
x, y, theta, xdot, ydot, thetadot = s
np_s = np.array([[x, y, theta, xdot, ydot, thetadot]])
trajectory.append(np_s[0, 0:2])
model_trajectory.append(np_s[0, 0:2])
s = torch.from_numpy(np_s)
s = s.cuda()

num_rollouts = 1000
num_steps = 10

with torch.no_grad():
    while abs(s.cpu().data[0][0] - goal_x) >= thres or abs(s.cpu().data[0][1] - goal_y) >= thres:
        reward = {}
        first_action = {}
        for i in range(num_rollouts):
            start_a = mdp.sample_random_action()
            reward[i] = float("inf")
            first_action[i] = start_a
            a = torch.from_numpy(np.array([start_a]))
            a = a.cuda()
            sim_s = deepcopy(s)
            for _ in range(num_steps):
                next_s = model.predict_next_state(sim_s.float(), a.float())
                s_cpu = next_s.cpu()
                reward[i] = min((s_cpu.data[0][0] - goal_x) ** 2 + (s_cpu.data[0][1] - goal_y) ** 2, reward[i])
                if abs(s_cpu.data[0][0] - goal_x) <= thres and abs(s_cpu.data[0][1] - goal_y) <= thres:
                    break
                sim_s = next_s
                a = torch.from_numpy(np.array([mdp.sample_random_action()]))
                a = a.cuda()

        i = min(reward, key=reward.get)
        action = first_action[i]
        action = torch.from_numpy(np.array([action]))
        action = action.cuda()
        pred_s = model.predict_next_state(s.float(), action.float())
        model_trajectory.append(pred_s.cpu().data[0][0:2].numpy())
        
        mdp.execute_agent_action(first_action[i])
    
        s = mdp.cur_state
        x, y, theta, xdot, ydot, thetadot = s
        np_s = np.array([[x, y, theta, xdot, ydot, thetadot]])
        trajectory.append(np_s[0, 0:2])
        s = torch.from_numpy(np_s)
        s = s.cuda()
        print("action executed!")
    print("done!")

# plot results
traj = np.array(trajectory)
traj2 = np.array(model_trajectory)

plt.figure()
plt.plot(traj[:,0], traj[:,1])     
plt.plot(traj2[:,0], traj2[:,1], color='green')
plt.show()
        