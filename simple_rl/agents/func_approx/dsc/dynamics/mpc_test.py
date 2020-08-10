import torch
import numpy as np
import matplotlib.pyplot as plt

from copy import deepcopy

from simple_rl.agents.func_approx.dsc.dynamics.mpc import MPC
from simple_rl.tasks.ant_reacher.AntReacherMDPClass import AntReacherMDP

def generate_data(mdp, epochs):
    buffer = []
    for episode in range(epochs):
        mdp.reset()
        for step in range(1000):
            s = deepcopy(mdp.cur_state)
            a = mdp.sample_random_action()
            r, sp = mdp.execute_agent_action(a)
            
            s_pos = np.array(s)
            sp_pos = np.array(sp)
            data = [s_pos, a, sp_pos]
        
            buffer.append(data)
            if sp.is_terminal():
                break
    return buffer

mdp = AntReacherMDP()
dataset = generate_data(mdp, 10)
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
mpc_controller = MPC(31, 8, dataset, device)
mpc_controller.train()

# run mpc
mdp = AntReacherMDP()
mdp.reset()
thres = 0.6
_ = mdp.sample_random_state()
goal = mdp.sample_random_state()
goal_x = goal[0]
goal_y = goal[1]

# setup starting variables
trajectory = []
s = mdp.cur_state
trajectory.append(s.position)
sx, sy = s.position

while abs(sx - goal_x) >= thres or abs(sy - goal_y) >= thres:
    action = mpc_controller.mpc_rollout(mdp, 14000, 7, goal)
    
    # execute action in mdp
    mdp.execute_agent_action(action)
    print("action executed!, {}".format(len(trajectory)))
    
    # update s for new state
    s = mdp.cur_state
    sx, sy = s.position

    trajectory.append(s.position)
