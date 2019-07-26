import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("white")
import pdb

from simple_rl.tasks.point_maze.PortablePointMazeMDPClass import PortablePointMazeMDP

ANGLE_SWEEP = np.arange(-np.pi, np.pi+np.pi/8, np.pi/8)


mdp = PortablePointMazeMDP(0, train_mode=False, test_mode=True, render=False, dense_reward=False)

def step(a):
    r, s = mdp.execute_agent_action(a)
    ego_obs = s.aspace_features()
    return ego_obs, s

def print_ego(obs):
    door_obs = obs[:8]
    key_obs = obs[8:12]
    lock_obs = obs[12:16]
    has_key = obs[16]
    print("Door obs = ", door_obs)
    print("Key  obs = ", key_obs)
    print("Lock obs = ", lock_obs)
    print("Has key  = ", has_key)
    print()

def collect_data():
    traj = []
    ego_traj = []
    for _ in range(10000):
        a1 = np.random.uniform(-1., 1., 1)
        a2 = np.random.uniform(-0.25, 0.25, 1)
        a  = np.array([a1, a2])
        ego_obs, next_state = step(a)
        print_ego(ego_obs)
        traj.append(next_state)
        ego_traj.append(ego_obs)
    return traj, ego_traj

def get_key_observations(ego_traj):
    key_observations = [obs[8:12] for obs in ego_traj]
    return key_observations

def get_lock_observations(ego_traj):
    lock_observations = [obs[12:16] for obs in ego_traj]
    return lock_observations

def get_door_observations(ego_traj):
    door_observations = [obs[:8] for obs in ego_traj]
    return door_observations

def get_key_distances(ego_traj):
    key_observations = get_key_observations(ego_traj)
    return [obs[0] for obs in key_observations]

def get_lock_distances(ego_traj):
    lock_observations = get_lock_observations(ego_traj)
    return [obs[0] for obs in lock_observations]

def get_door_distances(ego_traj):
    door_observations = get_door_observations(ego_traj)
    door1_distances = [obs[0] for obs in door_observations]
    door2_distances = [obs[1] for obs in door_observations]
    return door1_distances, door2_distances

def place_agent_in_key_room():
    position = np.array([6, 8])
    mdp.env.wrapped_env.set_xy(position)

    noop_action = np.array([0., 0.])
    step(noop_action)

def place_agent_in_key_room_2():
    position = np.array([9, 9])
    mdp.env.wrapped_env.set_xy(position)

    noop_action = np.array([0., 0.])
    step(noop_action)

def place_agent_in_lock_room():
    if mdp.train_mode:
        position = np.array([0, 8])
    else:
        position = np.array([8, 0])
    mdp.env.wrapped_env.set_xy(position)

    noop_action = np.array([0., 0.])
    step(noop_action)

def place_agent_in_room(room_number):
    if room_number == 1:
        position = np.array([0., 0.])
    elif room_number == 2:
        position = np.array([8., 0.])
    else:
        raise ValueError("Use place_agent_in_lock/key_room() method for room {}".format(room_number))
    mdp.env.wrapped_env.set_xy(position)

    noop_action = np.array([0., 0.])
    step(noop_action)

def sweep_key_angle():
    noop_action = np.array([0., 0.])
    key_angles = []
    for theta in ANGLE_SWEEP:
        mdp.env.wrapped_env.set_ori(theta)
        ego_obs, next_state = step(noop_action)
        key_observation = get_key_observations([ego_obs])[0]
        key_angle = key_observation[1]
        key_angles.append(key_angle)
    return key_angles

def sweep_lock_angle():
    noop_action = np.array([0, 0])
    lock_angles = []
    for theta in ANGLE_SWEEP:
        mdp.env.wrapped_env.set_ori(theta)
        ego_obs, next_state = step(noop_action)
        lock_observation = get_lock_observations([ego_obs])[0]
        lock_angle = lock_observation[1]
        lock_angles.append(lock_angle)
    return lock_angles

def sweep_door_angle():
    noop_action = np.array([0., 0.])
    door1_angles, door2_angles = [], []
    for theta in ANGLE_SWEEP:
        mdp.env.wrapped_env.set_ori(theta)
        ego_obs, next_state = step(noop_action)
        door_observation = get_door_observations([ego_obs])[0]
        door1_angle = door_observation[2]
        door2_angle = door_observation[3]
        door1_angles.append(door1_angle)
        door2_angles.append(door2_angle)
    return door1_angles, door2_angles

def plot_key_distances(traj, ego_traj):
    # Global x, y positions
    x = [s.position[0] for s in traj]
    y = [s.position[1] for s in traj]

    # Distance to key for each of the x-y points above
    key_distances = get_key_distances(ego_traj)

    # Plotting
    # plt.scatter(x, y, c=key_distances)
    plt.scatter(x, y, c=list(map(lambda x: x / max(key_distances), key_distances)), norm=matplotlib.colors.LogNorm())
    plt.colorbar()
    plt.title("Agent-Space: Distance to Key")
    plt.savefig("aspace_distance_to_key.png")
    plt.show()

def plot_lock_distances(traj, ego_traj):
    x = [s.position[0] for s in traj]
    y = [s.position[1] for s in traj]
    lock_distances = get_lock_distances(ego_traj)
    plt.figure()
    # plt.scatter(x, y, c=lock_distances)
    plt.scatter(x, y, c=list(map(lambda x: x / max(lock_distances), lock_distances)), norm=matplotlib.colors.LogNorm())
    plt.colorbar()
    plt.title("Agent-Space: Distance to Lock")
    plt.savefig("aspace_distance_to_lock.png")
    plt.show()

def plot_door_distances(traj, ego_traj):
    x = [s.position[0] for s in traj]
    y = [s.position[1] for s in traj]
    door1_distances, door2_distances = get_door_distances(ego_traj)

    plt.figure()
    plt.subplot(1, 2, 1)
    plt.scatter(x, y, c=list(map(lambda x: x / max(door1_distances), door1_distances)), norm=matplotlib.colors.LogNorm())
    plt.colorbar()
    plt.title("Distance to 1st Door in Current Room")

    plt.subplot(1, 2, 2)
    plt.scatter(x, y, c=list(map(lambda x: x / max(door2_distances), door2_distances)), norm=matplotlib.colors.LogNorm())
    plt.colorbar()
    plt.title("Distance to 2nd Door in Current Room")
    plt.suptitle("Agent Space")
    plt.savefig("aspace_distance_to_doors.png")
    plt.show()

def plot_collected_door_angles(traj, ego_traj):
    x = [s.position[0] for s in traj]
    y = [s.position[1] for s in traj]
    key_distances = get_key_distances(ego_traj)
    door_observations = get_door_observations(ego_traj)
    door1_angle = [obs[2] for obs in door_observations]
    measured_door_angles = []
    measured_x, measured_y = [], []
    for x0, y0, key_distance, door_angle in zip(x, y, key_distances, door1_angle):
        if key_distance < 100:
            measured_door_angles.append(door_angle)
            measured_x.append(x0)
            measured_y.append(y0)

    plt.figure()
    plt.scatter(measured_x, measured_y, c=measured_door_angles)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.colorbar()
    plt.title("Angle to Lock-Room Door")
    plt.show()
    plt.close()

def plot_key_angles():
    place_agent_in_key_room()
    key_angles = sweep_key_angle()
    state_angles = ANGLE_SWEEP

    plt.figure()
    plt.plot(np.rad2deg(state_angles), np.rad2deg(key_angles), "ro--")
    plt.xlabel("Global angle (degrees)")
    plt.ylabel("Angle to key (degrees)")
    plt.title("Agent Space: Angle to Key")
    plt.show()
    plt.close()

def plot_key_angles_2():
    place_agent_in_key_room_2()
    key_angles = sweep_key_angle()
    state_angles = ANGLE_SWEEP

    plt.figure()
    plt.plot(np.rad2deg(state_angles), np.rad2deg(key_angles), "ro--")
    plt.xlabel("Global angle (degrees)")
    plt.ylabel("Angle to key (degrees)")
    plt.title("Agent Space: Angle to Key")
    plt.show()
    plt.close()

def plot_lock_angles():
    place_agent_in_lock_room()
    lock_angles = sweep_lock_angle()
    state_angles = ANGLE_SWEEP

    plt.figure()
    plt.plot(np.rad2deg(state_angles), np.rad2deg(lock_angles), "ro--")
    plt.xlabel("Global angle (degrees)")
    plt.ylabel("Angle to key (degrees)")
    plt.title("Agent Space: Angle to Lock")
    plt.show()
    plt.close()

def plot_door_angles(room_number):
    if room_number == 1 or room_number == 2:
        place_agent_in_room(room_number)
    elif room_number == 3:
        place_agent_in_key_room()
    else:
        place_agent_in_lock_room()

    door1_angles, door2_angles = sweep_door_angle()
    state_angles = ANGLE_SWEEP

    plt.figure()
    plt.subplot(1, 2, 1)
    plt.plot(np.rad2deg(state_angles), np.rad2deg(door1_angles), "ro--")
    plt.xlabel("Global angle (degrees)")
    plt.ylabel("Angle to door 1 (degrees)")

    plt.subplot(1, 2, 2)
    plt.plot(np.rad2deg(state_angles), np.rad2deg(door2_angles), "ro--")
    plt.xlabel("Global angle (degrees)")
    plt.ylabel("Angle to door 2 (degrees)")

    plt.suptitle("Agent Space: Angle to Doors in Room {}".format(room_number))
    plt.show()
    plt.close()

if __name__ == "__main__":
    trajectory, egocentric_trajectory = collect_data()
    plot_key_distances(trajectory, egocentric_trajectory)
    plot_lock_distances(trajectory, egocentric_trajectory)
    plot_door_distances(trajectory, egocentric_trajectory)
    plot_key_angles()
    plot_key_angles_2()
    plot_lock_angles()
    plot_door_angles(1)
    plot_door_angles(2)
    plot_door_angles(3)
    plot_door_angles(4)
