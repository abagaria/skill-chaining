import torch, ipdb, pdb
import pickle
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy
from scipy.ndimage import uniform_filter1d
from simple_rl.tasks.leap_wrapper.LeapWrapperMDPClass import LeapWrapperMDP
from simple_rl.agents.func_approx.dsc.dynamics.mpc import MPC

def generate_train_data(mdp, epochs, num_positions, num_steps=1000, puck_interactions=False):
    global mpc

    states = []
    actions = []
    states_p = []

    for i in range(num_positions):
        position = mdp.sample_random_state() if not puck_interactions else sample_arm_puck_touching_state(mdp)
        print(f"[{i+1}/{num_positions}] Generating train data for {position}")
        for episode in range(epochs):
            mdp.reset_to_start_state(position)
            for step in range(num_steps):
                s = deepcopy(mdp.cur_state)
                a = mdp.sample_random_action() if not puck_interactions else sample_puck_action(s)
                r, sp = mdp.execute_agent_action(a)
                mpc.step(s, a, r, sp, False)
                
                states.append(np.array(s, dtype='float64'))
                actions.append(a)
                states_p.append(np.array(sp, dtype='float64'))
            
                if sp.is_terminal() or (puck_interactions and not valid_puck_state(sp)):
                    break
    return states, actions, states_p

def generate_validation_data(mdp, epochs, num_positions, num_steps=1000, puck_interactions=False):
    states = []
    actions = []
    states_p = []
    for i in range(num_positions):
        position = mdp.sample_random_state() if not puck_interactions else sample_arm_puck_touching_state(mdp)
        print(f"[{i + 1}/{num_positions}] Generating validation data for {position}")
        for episode in range(epochs):
            mdp.reset_to_start_state(position)
            traj_states = []
            traj_actions = []
            traj_states_p = []
            for step in range(num_steps):
                s = deepcopy(mdp.cur_state)
                a = mdp.sample_random_action() if not puck_interactions else sample_puck_action(s)
                r, sp = mdp.execute_agent_action(a)

                traj_states.append(np.array(s, dtype='float64'))
                traj_actions.append(a)
                traj_states_p.append(np.array(sp, dtype='float64'))

                if sp.is_terminal() or (puck_interactions and not valid_puck_state(sp)):
                    break
            states.append(traj_states)
            actions.append(traj_actions)
            states_p.append(traj_states_p)
    return states, actions, states_p

def valid_puck_state(state):
    puck_pos = state[3:]
    puck_low = [-.25, .4]
    puck_high = [.25, .8]
    return np.less(puck_low, puck_pos).all() and np.less(puck_pos, puck_high).all()

def sample_arm_puck_touching_state(mdp):
    max_iter = 500
    for _ in range(max_iter):
        state = mdp.sample_random_state()
        if is_touching(state):
            break
    return state

def sample_puck_action(state):
    state = np.array(state)
    arm_pos = state[:2]
    puck_pos = state[3:]
    touch_diff = (puck_pos - arm_pos)
    touch_vector_factor = 0.5
    tangential_noise_factor = 5
    noise = np.random.uniform(-1, 1) * np.array([-touch_diff[1], touch_diff[0]])
    action = noise * tangential_noise_factor + touch_diff * touch_vector_factor
    clipped_action = np.clip(action, [-1,-1], [1,1])
    return clipped_action

def test(goals, num_steps=1000):
    rewards = []
    trajs = []
    for i, goal in enumerate(goals):
        print(f"{i}: MPC towards {goal}")
        mdp.reset()
        trajectory = [deepcopy(mdp.cur_state)]
        reward = []
        mdp.set_goal(goal)
        for step in range(num_steps):
            action = mpc.act(mdp.cur_state, goal, num_rollouts=4000, num_steps=5)
            r,_ = mdp.execute_agent_action(action)
            reward.append(r)
            trajectory.append(deepcopy(mdp.cur_state))
        trajs.append(trajectory)
        rewards.append(reward)
    return trajs, rewards

def num_puck_interactions(mpc):
    count = 0
    for state in mpc.train_data.states:
        count += is_touching(state, 0.1)
    return count / len(mpc.train_data.states) * 100

def is_touching(state, touch_indicator=0.06):
    state = np.array(state)
    hand_pos = state[:2]
    puck_pos = state[3:]
    touch_diff = hand_pos - puck_pos
    touch_distance = np.linalg.norm(touch_diff, ord=2)
    return touch_distance < touch_indicator

def plot_data_distribution(mdp, states):
    states = np.array(states)
    low_hand_x, low_hand_y, _, low_puck_x, low_puck_y = mdp.get_low_lims()
    high_hand_x, high_hand_y, _, high_puck_x, high_puck_y = mdp.get_high_lims()
    fig, (ax1, ax2) = plt.subplots(1, 2)
    ax1.set_xlim((low_hand_x, high_hand_x))
    ax1.set_ylim((low_hand_y, high_hand_y))
    ax2.set_xlim((low_puck_x, high_puck_x))
    ax2.set_ylim((low_puck_y, high_puck_y))
    ax1.scatter(states[:, 0], states[:, 1], alpha=0.2)
    ax2.scatter(states[:, 3], states[:, 4], alpha=0.2)
    fig.savefig("dist.png")

def plot_validation_error(error, filename):
    plt.figure()
    plt.plot(range(len(error)), error)
    plt.savefig(filename)
    plt.close()

def plot_traj_rewards(rewards):
    train_time = len(rewards[0])
    smooth_learning_curve = uniform_filter1d(rewards, 20, mode='nearest', output=np.float)
    mean = np.mean(smooth_learning_curve, axis=0)
    std_err = np.std(smooth_learning_curve, axis=0)
    y_low, y_high = np.amin(rewards), 0
    plt.figure()
    plt.plot(range(train_time), mean, '-')
    plt.fill_between(range(train_time), np.maximum(mean - std_err, y_low), np.minimum(mean + std_err, y_high), alpha=0.2)
    plt.xlim((0, train_time))
    plt.ylim((y_low, y_high))
    plt.savefig("mpc_learning_curve.png")
    plt.close()

if __name__ == "__main__":
    # run parameters
    GOAL_TYPE = 'hand'
    DEBUG = False
    NUM_TRAJS = 4
    puck_interactions = GOAL_TYPE == 'puck'

    # create objects
    mdp = LeapWrapperMDP(goal_type=GOAL_TYPE, dense_reward=True, task_agnostic=False)
    mdp.env.seed(0)
    mpc = MPC(mdp, mdp.state_space_size(), mdp.action_space_size(), dense_reward=True, device=torch.device('cuda:0'))


    if DEBUG:
        a, b, c = generate_train_data(mdp, 2, 2, num_steps=50, puck_interactions=puck_interactions)
        mpc.load_data()
        mpc.train(2, 2)
        val_random_a, val_random_b, val_random_c = generate_validation_data(mdp, 2, 2, num_steps=20, puck_interactions=False)
        val_puck_a, val_puck_b, val_puck_c = generate_validation_data(mdp, 2, 2, num_steps=5, puck_interactions=True)
        goals = [mdp.sample_random_state() for _ in range(2)]
        trajs, rewards = test(goals)
    else:
        # train MPC
        # a, b, c = generate_train_data(mdp, 10, 1500, num_steps=5000, puck_interactions=puck_interactions)
        a, b, c = generate_train_data(mdp, 1, 400, num_steps=1000, puck_interactions=puck_interactions)
        mpc.load_data()
        mpc.train(750, 1024)
        plot_data_distribution(mdp, a)
        if puck_interactions:
            print(num_puck_interactions(mpc))

        # validation error
        val_random_a, val_random_b, val_random_c = generate_validation_data(mdp, 1, 100, num_steps=1000, puck_interactions=False)
        random_error = mpc.compute_validation_error(val_random_a, val_random_b, val_random_c)
        plot_validation_error(random_error, 'validation_random_error.png')
        plot_validation_error(random_error[:10], 'validation_random_error_truncated.png')
        # val_puck_a, val_puck_b, val_puck_c = generate_validation_data(mdp, 5, 300, num_steps=5000, puck_interactions=True)
        # puck_error = mpc.compute_validation_error(val_puck_a, val_puck_b, val_puck_c)
        # plot_validation_error(puck_error, 'validation_puck_error.png')
        # plot_validation_error(puck_error[:10], 'validation_puck_error_truncated.png')

        # Test MPC trajectories
        goals = [mdp.sample_random_state() for _ in range(NUM_TRAJS)]
        trajs, rewards = test(goals, num_steps=50)
        plot_traj_rewards(rewards)


# NOTE: ax1 = endeff; ax2 = puck
# fig, (ax1, ax2) = plt.subplots(1, 2)
# ax1.set_xlim((low_hand_x, high_hand_x))
# ax1.set_ylim((low_hand_y, high_hand_y))
# ax2.set_xlim((low_puck_x, high_puck_x))
# ax2.set_ylim((low_puck_y, high_puck_y))

# for l in traj:
#     endeff_x, endeff_y, endeff_z, puck_x, puck_y = l
#     ax1.plot(endeff_x, endeff_y, "-o", alpha=0.8)
#     ax2.plot(puck_x, puck_y, "-o", alpha=0.8)
# arm_start = mdp.init_state[:2]
# puck_start = mdp.init_state[3:]
# puck_goal = goal[3:]
# ax1.scatter(arm_start[0], arm_start[1], color="k", label="endeff start", marker="x", s=180)
# ax2.scatter(puck_start[0], puck_start[1], color="k", label="puck start", marker="+", s=180)
# ax2.scatter(puck_goal[0], puck_goal[1], color="k", label="puck start", marker="*", s=400)
# fig.savefig("trajectory.png")
# plt.close(fig)

