import torch
import numpy as np
import matplotlib.pyplot as plt

from copy import deepcopy

from simple_rl.agents.func_approx.dsc.dynamics.mpc import MPC
from simple_rl.tasks.ant_reacher.AntReacherMDPClass import AntReacherMDP

def generate_data(mdp, epochs):
    states = []
    actions = []
    states_p = []
    for episode in range(epochs):
        mdp.reset()
        for step in range(1000):
            s = deepcopy(mdp.cur_state)
            a = mdp.sample_random_action()
            r, sp = mdp.execute_agent_action(a)
            
            states.append(np.array(s))
            actions.append(a)
            states_p.append(np.array(sp))
        
            if sp.is_terminal():
                break
    return states, actions, states_p

mdp = AntReacherMDP()
states, actions, states_p = generate_data(mdp, 10)
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
mpc_controller = MPC(31, 8, device)
mpc_controller.load_data(states, actions, states_p)
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
    action = mpc_controller.mpc_act(mdp, 14000, 7, goal)
    
    # execute action in mdp
    mdp.execute_agent_action(action)
    print("action executed!, {}".format(len(trajectory)))
    
    # update s for new state
    s = mdp.cur_state
    sx, sy = s.position

    trajectory.append(s.position)
