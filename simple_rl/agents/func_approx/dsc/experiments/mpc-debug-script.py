import ipdb
import torch
import pickle
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy
from simple_rl.tasks.d4rl_ant_maze.D4RLAntMazeMDPClass import D4RLAntMazeMDP
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
    step = 0
    trajectory = [deepcopy(mdp.cur_state)]
    while step < num_steps and not mdp.sparse_gc_reward_function(mdp.cur_state, goal, {})[1]:
        action = mpc.act(mdp.cur_state, goal)
        mdp.execute_agent_action(action)
        trajectory.append(deepcopy(mdp.cur_state))
        step += 1
    return trajectory

mdp = D4RLAntMazeMDP(maze_size="umaze", seed=1, render=False, goal_state=np.array((0, 8)))
state_dim = mdp.state_space_size()
action_dim = mdp.action_space_size()

mpc = MPC(mdp, state_dim, action_dim, dense_reward=True, device=torch.device('cuda:0'))

a, b, c = generate_data(mdp, 10, positions = [np.array([0,0]), np.array([4,0]), np.array([8,0]), np.array([8,2]), np.array([8,4]), np.array([8,6]), np.array([8,8]), np.array([6,8]), np.array([4,8]), np.array([2,8]), np.array([0,8])])

plt.figure()
plt.scatter([q[0] for q in a], [q[1] for q in a])
plt.savefig("datadist.png")

mpc.load_data()
mpc.train(100, 1024)

mdp.reset()
_, steps_taken1, trajectory_1 = mpc._rollout_debug(mdp, 14000, 7, np.array([8,0]), max_steps=300)

mdp.reset()
_, steps_taken2, trajectory_2 = mpc._rollout_debug(mdp, 14000, 7, np.array([8,4]), max_steps=1000)

mdp.reset()
_, steps_taken3, trajectory_3 = mpc._rollout_debug(mdp, 14000, 7, np.array([8,0]), max_steps=1000)

mdp.reset()
_, steps_taken4, trajectory_4 = mpc._rollout_debug(mdp, 14000, 7, np.array([4,4]), max_steps=300)

plt.figure()
x_low_lim, y_low_lim = mdp.get_x_y_low_lims()
x_high_lim, y_high_lim = mdp.get_x_y_high_lims()
plt.xlim((x_low_lim, x_high_lim))
plt.ylim((y_low_lim, y_high_lim))
for l in trajectory_1:
    x, y = l[0], l[1]
    plt.plot(x, y, "-o")
plt.savefig("traj.png")


def test(mpc, goal, num_steps=1000):
    step = 0
    trajectory = [deepcopy(exp.mdp.cur_state)]
    while step < num_steps and not mdp.sparse_gc_reward_function(exp.mdp.cur_state, goal, {})[1]:
        action = mpc.act(exp.mdp.cur_state, goal)
        exp.mdp.execute_agent_action(action)
        trajectory.append(deepcopy(exp.mdp.cur_state))
        step += 1
    return trajectory