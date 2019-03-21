# Python imports.
import random
import numpy as np
import scipy.interpolate
import matplotlib.pyplot as plt
import time
import torch
import seaborn as sns
sns.set()

class Experience(object):
	def __init__(self, s, a, r, s_prime):
		self.state = s
		self.action = a
		self.reward = r
		self.next_state = s_prime

	def serialize(self):
		return self.state, self.action, self.reward, self.next_state

	def __str__(self):
		return "s: {}, a: {}, r: {}, s': {}".format(self.state, self.action, self.reward, self.next_state)

	def __repr__(self):
		return str(self)

	def __eq__(self, other):
		return isinstance(other, Experience) and self.__dict__ == other.__dict__

	def __ne__(self, other):
		return not self == other

# ---------------
# Plotting utils
# ---------------

def plot_trajectory(trajectory, color='k', marker="o"):
	for i, state in enumerate(trajectory):
		x = state.position[0]
		y = state.position[1]
		plt.scatter(x, y, c=color, alpha=float(i) / len(trajectory), marker=marker)

def plot_all_trajectories_in_initiation_data(initiation_data, marker="o"):
	possible_colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple',
					   'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan']
	for i, trajectory in enumerate(initiation_data):
		plot_trajectory(trajectory, color=possible_colors[i], marker=marker)

def sample_states_and_create_meshgrid(solver, state_form, device):
	# Array of experience tuples
	replay_buffer = solver.replay_buffer.memory

	# Extract all visited states in the replay buffer
	if len(replay_buffer) > 1000:
		states = random.sample([e.state for e in replay_buffer], 1000)
	else:
		states = [e.state for e in replay_buffer]

	states_array = np.vstack(states)

	x = states_array[:, 0]
	y = states_array[:, 1]

	xi, yi = np.linspace(x.min(), x.max(), 500), np.linspace(y.min(), y.max(), 500)
	xx, yy = np.meshgrid(xi, yi)

	if state_form == "tensor":
		states_array = torch.from_numpy(states_array).float().to(device)

	return x, y, xx, yy, states_array

def render_sampled_value_function(solver, device, episode=None):
	x, y, xx, yy, states_tensor = sample_states_and_create_meshgrid(solver, "tensor", device)

	# Forward pass through our Q-function
	with torch.no_grad():
		q_values = solver.get_batched_qvalues(states_tensor)
		values = torch.max(q_values, dim=-1)[0].cpu().data.numpy()

	try:
		rbf = scipy.interpolate.Rbf(x, y, values, function="linear")
	except:
		print("Could not render value function because of interpolation error")
		plt.close()
		return

	zz = rbf(xx, yy)
	plt.contourf(xx, yy, zz)
	plt.colorbar()
	plt.title("Value Function at episode {}".format(episode))
	name = solver.name if episode is None else solver.name + "_" + str(episode)
	plt.savefig("{}_value_function.png".format(name))
	plt.close()


def render_sampled_initiation_classifier(option, global_solver):
	x, y, xx, yy, states_array = sample_states_and_create_meshgrid(global_solver, "array", None)
	inits = option.batched_is_init_true(states_array)

	try:
		rbf = scipy.interpolate.Rbf(x, y, inits, function="linear")
	except:
		print("Could not render initiation classifier because of interpolation error")
		plt.close()
		return

	zz = rbf(xx, yy)

	plt.contourf(xx, yy, zz, cmap=plt.cm.coolwarm, alpha=0.8)
	plt.colorbar()
	plot_all_trajectories_in_initiation_data(option.positive_examples)
	plt.xlabel("x")
	plt.ylabel("y")
	plt.savefig("{}_svm_{}.png".format(option.name, time.time()))
	plt.close()

def visualize_dqn_replay_buffer(solver, experiment_name=""):
	goal_transitions = list(filter(lambda e: e.reward >= 0 and e.done == 1, solver.replay_buffer.memory))
	cliff_transitions = list(filter(lambda e: e.reward < 0 and e.done == 1, solver.replay_buffer.memory))
	non_terminals = list(filter(lambda e: e.done == 0, solver.replay_buffer.memory))

	goal_x = [e.next_state.position[0] for e in goal_transitions]
	goal_y = [e.next_state.position[1] for e in goal_transitions]
	cliff_x = [e.next_state.position[0] for e in cliff_transitions]
	cliff_y = [e.next_state.position[1] for e in cliff_transitions]
	non_term_x = [e.next_state.position[0] for e in non_terminals]
	non_term_y = [e.next_state.position[1] for e in non_terminals]

	# background_image = imread("pinball_domain.png")

	plt.figure()
	plt.scatter(cliff_x, cliff_y, alpha=0.67, label="cliff")
	plt.scatter(non_term_x, non_term_y, alpha=0.2, label="non_terminal")
	plt.scatter(goal_x, goal_y, alpha=0.67, label="goal")
	# plt.imshow(background_image, zorder=0, alpha=0.5, extent=[0., 1., 1., 0.])

	plt.legend()
	plt.title("# transitions = {}".format(len(solver.replay_buffer)))
	plt.savefig("{}_replay_buffer_analysis_{}.png".format(solver.name, experiment_name))
	plt.close()

def visualize_smdp_updates(global_solver, mdp, experiment_name=""):
	smdp_transitions = list(filter(lambda e: not mdp.is_primitive_action(e.action), global_solver.replay_buffer.memory))
	negative_transitions = list(filter(lambda e: e.done == 0, smdp_transitions))
	terminal_transitions = list(filter(lambda e: e.done == 1, smdp_transitions))
	positive_start_x = [e.state.position[0] for e in terminal_transitions]
	positive_start_y = [e.state.position[1] for e in terminal_transitions]
	positive_end_x = [e.next_state.position[0] for e in terminal_transitions]
	positive_end_y = [e.next_state.position[1] for e in terminal_transitions]
	negative_start_x = [e.state.position[0] for e in negative_transitions]
	negative_start_y = [e.state.position[1] for e in negative_transitions]
	negative_end_x = [e.next_state.position[0] for e in negative_transitions]
	negative_end_y = [e.next_state.position[1] for e in negative_transitions]

	plt.figure(figsize=(8, 5))
	plt.scatter(positive_start_x, positive_start_y, alpha=0.6, label="+s")
	plt.scatter(negative_start_x, negative_start_y, alpha=0.6, label="-s")
	plt.scatter(positive_end_x, positive_end_y, alpha=0.6, label="+s'")
	plt.scatter(negative_end_x, negative_end_y, alpha=0.6, label="-s'")
	plt.legend()
	plt.title("# updates = {}".format(len(smdp_transitions)))
	plt.savefig("DQN_SMDP_Updates_{}.png".format(experiment_name))
	plt.close()