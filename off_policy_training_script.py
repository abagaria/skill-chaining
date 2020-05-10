import torch
import pickle
import argparse
import numpy as np
from tqdm import tqdm
from simple_rl.tasks.point_reacher.PointReacherMDPClass import PointReacherMDP
from simple_rl.tasks.d4rl_ant_maze.D4RLAntMazeMDPClass import D4RLAntMazeMDP
from simple_rl.agents.func_approx.dsc.utils import *
from simple_rl.agents.func_approx.ddpg.DDPGAgentClass import DDPGAgent

parser = argparse.ArgumentParser()
parser.add_argument("--lr_a", type=float, help="DDPG Actor learning rate", default=1e-4)
parser.add_argument("--lr_c", type=float, help="DDPG Critic learning rate", default=1e-3)   
parser.add_argument("--support", type=str, default="local")
parser.add_argument("--reward", type=str, default="sparse")
args = parser.parse_args()

assert args.support in ("local", "global")
assert args.reward in ("sparse", "dense")

start_state = np.array((4, 0))
goal_state = np.array((7, 0))

init_predicate = lambda s: 4 <= s[0] <= 8 and s[1] < 2
goal_predicate = lambda s: np.linalg.norm(s[:2] - goal_state) <= 0.5


def get_sparse_subgoal_reward(s_prime):
  if goal_predicate(s_prime):
    return 10.
  return -1.


def get_dense_subgoal_reward(s_prime):
  if goal_predicate(s_prime):
    return 10.
  return -np.linalg.norm(s_prime[:2] - goal_state) / np.linalg.norm(goal_state-start_state)


def initialize_with_global_ddpg(solver, memory):
    """" Initialize option policy - unrestricted support because not checking if I_o(s) is true. """
    for state, action, reward, next_state, done in tqdm(memory, desc=f"Initializing {solver.name} policy"):
      if goal_predicate(state):
        continue

      if goal_predicate(next_state):
        solver.step(state, action, 10., next_state, True)
      else:
        option_reward = get_sparse_subgoal_reward(next_state) if args.reward == "sparse" else get_dense_subgoal_reward(next_state)
        solver.step(state, action, option_reward, next_state, done)
    solver.epsilon = 0.1


def initialize_with_local_support(solver, memory, init_predicate):
    for state, action, reward, next_state, done in tqdm(memory, desc=f"Initializing {solver.name} policy"):
      if goal_predicate(state):
        continue
      if goal_predicate(next_state):
        solver.step(state, action, 10., next_state, True)
      elif init_predicate(state):
        option_reward = get_sparse_subgoal_reward(next_state) if args.reward == "sparse" else get_dense_subgoal_reward(next_state)
        solver.step(state, action, option_reward, next_state, done)
    solver.epsilon = 0.1  

def reset(mdp, start_state):
    """ Reset the MDP to either the default start state or to the specified one. """
    mdp.reset()

    start_position = start_state[:2]
    mdp.set_xy(start_position)

def trained_execution(mdp, solver):
  state = mdp.cur_state
  score = 0.
  num_steps = 0

  while not goal_predicate(state.position) and num_steps < 200:
    action = solver.act(state.features(), evaluation_mode=True)
    reward, state = mdp.execute_agent_action(action)

    score += reward
    num_steps += 1

  if goal_predicate(state.position):
    return True

  return False

def measure_success(mdp, solver):
  successes = []
  for _ in tqdm(range(100)):
    reset(mdp, start_state)
    successes.append(trained_execution(mdp, solver))
  return sum(successes)

overall_mdp = D4RLAntMazeMDP(maze_size="medium", seed=0, render=False)
reset(overall_mdp, start_state)
state_dim = overall_mdp.state_space_size()
action_dim = overall_mdp.action_space_size()  

lr_a = args.lr_a
lr_c = args.lr_c

device = torch.device("cuda")
batch_size = 64
name = f"option_dppg_lra_{lr_a}_lrc_{lr_c}_support_{args.support}_{args.reward}_reward"
experiment_name = 'd4rl-ant-umaze-3events-toy-2'
solver = DDPGAgent(state_dim, action_dim, 0, device, lr_a, lr_c, batch_size, tensor_log=True, writer=None, name=name, exploration="")

with open("ant_og_replay_buffer.pkl", "rb") as f:
  memory = pickle.load(f)

if args.support == "global":
  initialize_with_global_ddpg(solver, memory)
else:
  initialize_with_local_support(solver, memory, init_predicate)

make_chunked_value_function_plot(solver, 0, 0, experiment_name)
success_rate = measure_success(overall_mdp, solver)
print(f"{name} success rate = {success_rate}")
