import pickle
import numpy as np
from tqdm import tqdm
from copy import deepcopy
from simple_rl.agents.func_approx.dqn.DQNAgentClass import ReplayBuffer
from simple_rl.tasks.point_maze.PointMazeMDPClass import PointMazeMDP
from simple_rl.tasks.ant_reacher.AntReacherMDPClass import AntReacherMDP


def get_saved_transition_data(env_name, device):
    if env_name == "ant-maze":
        with open("d4rl_transitions.pkl", "rb") as f:
            transitions = pickle.load(f)
    elif env_name == "ant-reacher":
        with open("ant_reacher_transitions.pkl", "rb") as f:
            transitions = pickle.load(f)

    r_buffer = ReplayBuffer(10, int(1e6), 64, 0, device)

    for transition in transitions:
        r_buffer.add(*tuple(transition))

    return r_buffer


def get_random_data(env_name, device, n_samples=int(1e6), periodic_resets=True):
    if env_name == "point-maze":
        mdp = PointMazeMDP(0)
        action_dim = 2
    elif env_name == "ant-reacher":
        mdp = AntReacherMDP()
        action_dim = 8

    r_buffer = ReplayBuffer(10, n_samples, 128, 0, device)

    for step in tqdm(range(n_samples)):
        state = deepcopy(mdp.cur_state)
        action = np.random.uniform(-1, +1, size=(action_dim,))
        reward, next_state = mdp.execute_agent_action(action)

        r_buffer.add(state.features(), action, reward, next_state.features(), next_state.is_terminal(), 1)

        if next_state.is_terminal() or (periodic_resets and (step % 2000 == 0)):
            mdp.reset()

    return r_buffer

