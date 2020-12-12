import torch
import pickle
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy
from simple_rl.tasks.ant_reacher.AntReacherMDPClass import AntReacherMDP
from simple_rl.agents.func_approx.dsc.dynamics.mpc import MPC

def generate_data(mdp, epochs, positions):
    global mpc

    states = []
    actions = []
    states_p = []

    for position in positions:
        for episode in range(epochs):
            mdp.set_xy(position)
            for step in range(1000):
                s = deepcopy(mdp.cur_state)
                a = mdp.sample_random_action()
                r, sp = mdp.execute_agent_action(a)

                mpc.step(s, a, r, sp, False)
                
                states.append(np.array(s, dtype='float64'))
                actions.append(a)
                states_p.append(np.array(sp, dtype='float64'))
            
                if sp.is_terminal():
                    break
    return states, actions, states_p

def test(goal, num_steps=1000):
    mdp.goal_state = goal
    step = 0
    trajectory = [deepcopy(mdp.cur_state)]
    while step < num_steps:
        action = mpc.act(mdp.cur_state, goal)
        mdp.execute_agent_action(action)
        trajectory.append(deepcopy(mdp.cur_state))
        step += 1
    return trajectory

mdp = AntReacherMDP(render=False)
state_dim = mdp.state_space_size()
action_dim = mdp.action_space_size()

mpc = MPC(mdp, state_dim, action_dim, dense_reward=True, device=torch.device('cuda:0'))

a, b, c = generate_data(mdp, 1, positions = [np.array((i,j)) for i in range(-10,10) for j in range(-10,10)])

plt.figure()
plt.scatter([q[0] for q in a], [q[1] for q in a])
plt.savefig("datadist.png")

mpc.load_data()
mpc.train(100, 1024)

mdp.reset()
traj = test(np.array((10,10)))

plt.figure()
x_low_lim, y_low_lim = mdp.get_x_y_low_lims()
x_high_lim, y_high_lim = mdp.get_x_y_high_lims()
plt.xlim((x_low_lim, x_high_lim))
plt.ylim((y_low_lim, y_high_lim))
for l in trajectory_1:
    x, y = l[0], l[1]
    plt.plot(x, y, "-o")
plt.savefig("traj.png")