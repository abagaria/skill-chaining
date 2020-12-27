import torch, ipdb
import pickle
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy
from scipy.ndimage import uniform_filter1d
from simple_rl.tasks.leap_wrapper.LeapWrapperMDPClass import LeapWrapperMDP
from simple_rl.agents.func_approx.dsc.dynamics.mpc import MPC

def generate_train_data(mdp, epochs, num_positions, num_steps=1000):
    global mpc

    states = []
    actions = []
    states_p = []

    for i in range(num_positions):
        position = mdp.sample_random_state()
        print(f"[{i+1}/{num_positions}] Generating train data for {position}")
        for episode in range(epochs):
            mdp.reset_to_start_state(position)
            for step in range(num_steps):
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

def generate_validation_data(mdp, epochs, num_positions, num_steps=1000):
    states = []
    actions = []
    states_p = []
    for i in range(num_positions):
        position = mdp.sample_random_state()
        print(f"[{i + 1}/{num_positions}] Generating validation data for {position}")
        for episode in range(epochs):
            mdp.reset_to_start_state(position)
            traj_states = []
            traj_actions = []
            traj_states_p = []
            for step in range(num_steps):
                s = deepcopy(mdp.cur_state)
                a = mdp.sample_random_action()
                r, sp = mdp.execute_agent_action(a)

                traj_states.append(np.array(s, dtype='float64'))
                traj_actions.append(a)
                traj_states_p.append(np.array(sp, dtype='float64'))

                if sp.is_terminal():
                    break
            states.append(traj_states)
            actions.append(traj_actions)
            states_p.append(traj_states_p)
    return states, actions, states_p

def test(goal, num_steps=1000):
    rewards = []
    mdp.goal_state = goal
    step = 0
    trajectory = [deepcopy(mdp.cur_state)]
    while step < num_steps:
        action = mpc.act(mdp.cur_state, goal)
        r,_ = mdp.execute_agent_action(action)
        rewards.append(r)
        trajectory.append(deepcopy(mdp.cur_state))
        step += 1
    return trajectory, rewards

mdp = LeapWrapperMDP(task_agnostic=False, use_hard_coded_events=False, dense_reward=True)
mdp.env.seed(0)
state_dim = mdp.state_space_size()
action_dim = mdp.action_space_size()

mpc = MPC(mdp, state_dim, action_dim, dense_reward=True, device=torch.device('cuda:0'))

a, b, c = generate_train_data(mdp, 1, 400)
a = np.array(a)

# plot data distribution
low_hand_x, low_hand_y, _, low_puck_x, low_puck_y = mdp.get_low_lims()
high_hand_x, high_hand_y, _, high_puck_x, high_puck_y = mdp.get_high_lims()
fig, (ax1, ax2) = plt.subplots(1, 2)
ax1.set_xlim((low_hand_x, high_hand_x))
ax1.set_ylim((low_hand_y, high_hand_y))
ax2.set_xlim((low_puck_x, high_puck_x))
ax2.set_ylim((low_puck_y, high_puck_y))
ax1.scatter(a[:,0], a[:,1], alpha=0.2)
ax2.scatter(a[:,3], a[:,4], alpha=0.2)
fig.savefig("dist.png")

mpc.load_data()
mpc.train(500, 1024)

# plot validation error
val_a, val_b, val_c = generate_validation_data(mdp, 1, 100)
error = mpc.compute_validation_error(val_a, val_b, val_c)
plt.figure()
plt.plot(range(len(error)), error)
plt.savefig("validation_error.png")

mdp.reset()
goal = np.array([0.15, 0.6, 0.02, -0.15, 0.6])
traj, rewards = test(goal)

# ax1 = endeff; ax2 = puck
fig, (ax1, ax2) = plt.subplots(1, 2)
ax1.set_xlim((low_hand_x, high_hand_x))
ax1.set_ylim((low_hand_y, high_hand_y))
ax2.set_xlim((low_puck_x, high_puck_x))
ax2.set_ylim((low_puck_y, high_puck_y))
for l in traj:
    endeff_x, endeff_y, endeff_z, puck_x, puck_y = l
    ax1.plot(endeff_x, endeff_y, "-o", alpha=0.8)
    ax2.plot(puck_x, puck_y, "-o", alpha=0.8)
fig.savefig("trajectory.png")
plt.close(fig)

# plot test-time trajectory reward (should increase over time)
rewards = uniform_filter1d(rewards, 20, mode='nearest', output=np.float)
plt.figure()
plt.plot(range(len(rewards)), rewards)
plt.xlim((0, len(rewards)))
plt.savefig("mpc_learning_curve.png")
plt.close('all')