import numpy as np
import torch
import matplotlib.pyplot as plt

def visualize_ddpg_replay_buffer(solver, solver_number, directory):
    print('*' * 80)
    print("Plotting value function...")
    print('*' * 80)
    states = np.array([exp[0] for exp in solver.replay_buffer.memory])
    actions = np.array([exp[1] for exp in solver.replay_buffer.memory])
    states_tensor = torch.from_numpy(states).float().to(solver.device)
    actions_tensor = torch.from_numpy(actions).float().to(solver.device)
    qvalues = solver.get_qvalues(states_tensor, actions_tensor).cpu().numpy().squeeze(1)
    plt.scatter(states[:, 0], states[:, 1], c=qvalues)

    plt.xlim(-10, 10)
    plt.ylim(-10, 10)

    plt.xlabel("x position")
    plt.ylabel("y position")

    cbar = plt.colorbar()
    cbar.set_label("predicted value")

    plt.title(f"DDPG {solver_number}, unconditioned")

    plt.savefig(directory / f'ddpg_value_function_{solver_number}.png')
    plt.close()

def visualize_gc_ddpg_replay_buffer(solver, goal_state, solver_number, directory):
    print('*' * 80)
    print("Plotting value function...")
    print('*' * 80)
    def condition_on_goal(state):
        gc_state = state
        gc_state[-2] = goal_state[0]
        gc_state[-1] = goal_state[1]

        return gc_state

    states = np.array([condition_on_goal(exp[0]) for exp in solver.replay_buffer.memory])
    actions = np.array([exp[1] for exp in solver.replay_buffer.memory])

    states_tensor = torch.from_numpy(states).float().to(solver.device)
    actions_tensor = torch.from_numpy(actions).float().to(solver.device)
    qvalues = solver.get_qvalues(states_tensor, actions_tensor).cpu().numpy().squeeze(1)
    plt.scatter(states[:, 0], states[:, 1], c=qvalues)

    plt.xlim(-10, 10)
    plt.ylim(-10, 10)

    plt.xlabel("x position")
    plt.ylabel("y position")

    cbar = plt.colorbar()
    cbar.set_label("predicted value")

    plt.title(f"DDPG {solver_number}, conditioned on ({goal_state[0]},{goal_state[1]})")

    plt.savefig(directory / f'ddpg_gc_value_function_{solver_number}.png')
    plt.close()
