import torch, ipdb, math, pickle
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy
from scipy.ndimage import uniform_filter1d
from simple_rl.tasks.leap_wrapper.LeapWrapperMDPClass import LeapWrapperMDP
from simple_rl.agents.func_approx.dsc.dynamics.mpc import MPC


def generate_data(mdp, epochs, num_positions, num_steps=1000, puck_interactions=False):
    states = []
    actions = []
    states_p = []
    rewards = []
    for i in range(num_positions):
        start_state = mdp.sample_state() if not puck_interactions else sample_arm_puck_touching_state(mdp)
        print(f"[{i + 1}/{num_positions}] Generating validation data for {start_state}")
        for episode in range(epochs):
            mdp.reset_to_start_state(start_state)
            traj_states = []
            traj_actions = []
            traj_states_p = []
            traj_rewards = []
            for step in range(num_steps):
                s = deepcopy(mdp.cur_state)
                a = mdp.sample_random_action() if not puck_interactions else sample_puck_action(s)
                r, sp = mdp.execute_agent_action(a)

                traj_states.append(np.array(s, dtype='float64'))
                traj_actions.append(a)
                traj_states_p.append(np.array(sp, dtype='float64'))
                traj_rewards.append(r)

                if sp.is_terminal() or (puck_interactions and not valid_puck_state(sp)):
                    break
            states.append(traj_states)
            actions.append(traj_actions)
            states_p.append(traj_states_p)
            rewards.append(traj_rewards)
    return states, actions, states_p, rewards


def valid_puck_state(state):
    puck_pos = state[3:]
    puck_low = [-.4, .3]
    puck_high = [.4, .9]
    return np.less(puck_low, puck_pos).all() and np.less(puck_pos, puck_high).all()


def sample_arm_puck_touching_state(mdp):
    while True:
        state = mdp.sample_state()
        if is_touching(state):
            return state


def sample_puck_action(state):
    state = np.array(state)
    arm_pos = state[:2]
    puck_pos = state[3:]

    touch_diff = (puck_pos - arm_pos)
    scaled_touch_diff = touch_diff / np.linalg.norm(touch_diff) * np.random.uniform()
    orth_vector = np.array([-scaled_touch_diff[1], scaled_touch_diff[0]])
    noise = orth_vector * np.random.uniform(-.1, .1)

    action = noise + scaled_touch_diff
    clipped_action = np.clip(action, [-1, -1], [1, 1])
    return clipped_action


def test(starts, goals, num_steps=1000):
    rewards = []
    trajs = []
    actions = []
    for i, (start, goal) in enumerate(zip(starts, goals)):
        print(f"{i}: MPC from {start} to {goal}")
        mdp.reset_to_start_state(start)
        trajectory = [deepcopy(mdp.cur_state)]
        reward = []
        action = []
        mdp.set_goal(goal)
        for step in range(num_steps):
            a = mpc.act(mdp.cur_state, goal, num_rollouts=20000, num_steps=10)
            r,_ = mdp.execute_agent_action(a)
            reward.append(r)
            action.append(a)
            trajectory.append(deepcopy(mdp.cur_state))
        trajs.append(trajectory)
        rewards.append(reward)
        actions.append(action)
    return trajs, rewards, actions


def num_puck_interactions(mpc):
    count = 0
    for state in mpc.train_data.states:
        count += is_touching(state)
    return count / len(mpc.train_data.states) * 100


def is_touching(state):
    state = np.array(state)
    hand_pos = state[:2]
    puck_pos = state[3:]
    touch_diff = hand_pos - puck_pos
    touch_distance = np.linalg.norm(touch_diff, ord=2)
    return touch_distance < 0.1


def plot_data_distribution(mdp):
    np_states = mpc.replay_buffer.obs_buf[:mpc.replay_buffer.size]
    low_hand_x, low_hand_y, _, low_puck_x, low_puck_y = mdp.get_low_lims()
    high_hand_x, high_hand_y, _, high_puck_x, high_puck_y = mdp.get_high_lims()
    fig, (ax1, ax2) = plt.subplots(1, 2)
    ax1.set_xlim((low_hand_x, high_hand_x))
    ax1.set_ylim((low_hand_y, high_hand_y))
    ax2.set_xlim((low_puck_x, high_puck_x))
    ax2.set_ylim((low_puck_y, high_puck_y))
    ax1.scatter(np_states[:, 0], np_states[:, 1], alpha=0.2)
    ax2.scatter(np_states[:, 3], np_states[:, 4], alpha=0.2)
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


def sample_complex_puck_start_goal_states(mdp):
    def scale(vector, length):
        """ Returns the unit vector of the vector.  """
        return vector / np.linalg.norm(vector) * length

    def valid_arm_start(hand_pos):
        low_hand_x, low_hand_y, _, _, _ = mdp.get_low_lims()
        high_hand_x, high_hand_y, _, _, _ = mdp.get_high_lims()
        return low_hand_x <= hand_pos[0] <= high_hand_x and low_hand_y <= hand_pos[1] <= high_hand_y

    while True:
        start, goal = mdp.sample_valid_goal(), mdp.sample_valid_goal()
        puck_start = start[3:]
        puck_goal = goal[3:]
        if np.linalg.norm(puck_start - puck_goal) < 0.4:
            continue
        puck_diff = scale(puck_start - puck_goal, .14)
        hand_start = puck_start + puck_diff
        if valid_arm_start(hand_start):
            start[:2] = hand_start
            return start, goal


def calculate_puck_rewards(trajs, goals):
    trajs = np.array(trajs)
    goals = np.array(goals)
    goal_pos = np.expand_dims(goals[:, 3:], 1)
    curr_pos = trajs[..., 3:]
    return -np.linalg.norm(goal_pos - curr_pos, axis=2)


def load_data(a, b, c, d):
    for states, actions, state_ps, rewards in zip(a, b, c, d):
        for state, action, r, state_p in zip(states, actions, rewards, state_ps):
            mpc.step(state, action, r, state_p, False)
    mpc.load_data()


if __name__ == "__main__":
    # run parameters
    GOAL_TYPE = 'hand_and_puck'
    DEBUG = False
    NUM_TRAJS = 15
    puck_interactions = 'puck' in GOAL_TYPE

    # create objects
    mdp = LeapWrapperMDP(goal_type=GOAL_TYPE, dense_reward=True, task_agnostic=False)
    mdp.env.seed(0)
    mpc = MPC(mdp, mdp.state_space_size(), mdp.action_space_size(), dense_reward=True, device=torch.device('cuda:0'))

    if DEBUG:
        a, b, c, d = generate_data(mdp, 2, 2, num_steps=50, puck_interactions=puck_interactions)
        load_data(a, b, c, d)
        mpc.train(2, 2)
        val_random_a, val_random_b, val_random_c, _ = generate_data(mdp, 2, 2, num_steps=20, puck_interactions=False)
        val_puck_a, val_puck_b, val_puck_c, _ = generate_data(mdp, 2, 2, num_steps=5, puck_interactions=True)

        if GOAL_TYPE == 'hand_and_puck':
            goals, starts = [], []
            for _ in range(NUM_TRAJS):
                start, goal = sample_complex_puck_start_goal_states(mdp)
                goals.append(goal)
                starts.append(start)
        else:
            goals = [mdp.sample_valid_goal() for _ in range(2)]
            starts = [deepcopy(mdp.init_state) for _ in range(2)]
        trajs, rewards, actions = test(starts, goals, num_steps=1)
    else:
        # train MPC
        a, b, c, d = generate_data(mdp, 1, 2000, num_steps=200, puck_interactions=True)
        e, f, g, h = generate_data(mdp, 1, 350, num_steps=200, puck_interactions=False)
        load_data(a, b, c, d)
        load_data(e, f, g, h)
        mpc.train(750, 1024)
        plot_data_distribution(mdp)
        if puck_interactions:
            print(num_puck_interactions(mpc))

        # validation error
        val_rand_a, val_rand_b, val_rand_c, _ = generate_data(mdp, 1, 200, num_steps=200, puck_interactions=False)
        random_error = mpc.compute_validation_error(val_rand_a, val_rand_b, val_rand_c)
        plot_validation_error(random_error, 'validation_random_error.png')
        plot_validation_error(random_error[:5], 'validation_random_error_truncated.png')
        val_puck_a, val_puck_b, val_puck_c, _ = generate_data(mdp, 1, 200, num_steps=200, puck_interactions=True)
        puck_error = mpc.compute_validation_error(val_puck_a, val_puck_b, val_puck_c)
        plot_validation_error(puck_error, 'validation_puck_error.png')
        plot_validation_error(puck_error[:5], 'validation_puck_error_truncated.png')

        # Test MPC trajectories
        if GOAL_TYPE == 'hand_and_puck':
            goals, starts = [], []
            for _ in range(NUM_TRAJS):
                start, goal = sample_complex_puck_start_goal_states(mdp)
                goals.append(goal)
                starts.append(start)
        else:
            goals = [mdp.sample_valid_goal() for _ in range(NUM_TRAJS)]
            starts = [mdp.sample_valid_goal() for _ in range(NUM_TRAJS)]

        trajs, rewards, actions = test(starts, goals, num_steps=200)
        puck_rewards = calculate_puck_rewards(trajs, goals)
        plot_traj_rewards(puck_rewards)

