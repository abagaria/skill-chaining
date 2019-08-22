# Python imports.
import pdb
import numpy as np
import scipy.interpolate
import imageio
import matplotlib.pyplot as plt
import itertools
import seaborn as sns
sns.set()

# Other imports.
from simple_rl.tasks.point_maze.PointMazeStateClass import PointMazeState
from simple_rl.tasks.point_maze.PortablePointMazeStateClass import PortablePointMazeState

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
		if isinstance(state, PointMazeState):
			x = state.position[0]
			y = state.position[1]
		else:
			x = state[0]
			y = state[1]

		if marker == "x":
			plt.scatter(x, y, c=color, s=100, marker=marker)
		else:
			plt.scatter(x, y, c=color, alpha=float(i) / len(trajectory), marker=marker)

def plot_all_trajectories_in_initiation_data(initiation_data, marker="o"):
	possible_colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple',
					   'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan']
	for i, trajectory in enumerate(initiation_data):
		color_idx = i % len(possible_colors)
		plot_trajectory(trajectory, color=possible_colors[color_idx], marker=marker)

def get_grid_states():
	ss = []
	for x in np.arange(-2., 11., 1.):
		for y in np.arange(-2., 11., 1.):
			s = PointMazeState(position=np.array([x, y]), velocity=np.array([0., 0.]),
							   theta=0., theta_dot=0., has_key=False, done=False)
			ss.append(s)
	return ss

def get_values(solver, init_values=False):
	values = []
	for x in np.arange(-2., 11., 1.):
		for y in np.arange(-2., 11., 1.):
			v = []
			for has_key in [False, True]:
				s = PointMazeState(position=np.array([x, y]), velocity=np.array([0, 0]),
								   theta=0, theta_dot=0, has_key=has_key, done=False)

				if not init_values:
					v.append(solver.get_value(s.features()))
				else:
					v.append(solver.is_init_true(s))

			if not init_values:
				values.append(np.mean(v))
			else:
				values.append(np.sum(v))

	return values

def get_initiation_set_values(option, has_key):
	values = []
	for x in np.arange(-2., 11., 1.):
		for y in np.arange(-2., 11., 1.):
			s = PointMazeState(position=np.array([x, y]), velocity=np.array([0, 0]),
							   theta=0, theta_dot=0, has_key=has_key, done=False)
			values.append(option.is_init_true(s))

	return values

def render_sampled_value_function(solver, episode=None, experiment_name=""):
	states = get_grid_states()
	values = get_values(solver)

	x = np.array([state.position[0] for state in states])
	y = np.array([state.position[1] for state in states])
	xi, yi = np.linspace(x.min(), x.max(), 1000), np.linspace(y.min(), y.max(), 1000)
	xx, yy = np.meshgrid(xi, yi)
	rbf = scipy.interpolate.Rbf(x, y, values, function="linear")
	zz = rbf(xx, yy)
	plt.imshow(zz, vmin=min(values), vmax=max(values), extent=[x.min(), x.max(), y.min(), y.max()], origin="lower")
	plt.colorbar()
	name = solver.name if episode is None else solver.name + "_{}_{}".format(experiment_name, episode)
	plt.savefig("value_function_plots/{}/{}_value_function.png".format(experiment_name, name))
	plt.close()

def render_agent_space_initiation_classifier(option, episode, experiment_name):
	def generate_plot(has_key):
		experiences = option.solver.replay_buffer.memory
		filtered_states = [exp[3] for exp in experiences if exp[3][-1] == has_key]
		key_distances = [state[8] for state in filtered_states]
		lock_distances = [state[12] for state in filtered_states]
		init_values = option.batched_is_init_true(np.array(filtered_states))
		plt.scatter(key_distances, lock_distances, c=init_values, cmap=plt.cm.coolwarm)
	
		plt.xlim((-95, 105))
		plt.ylim((-5, 20))

		plt.title("{} Initiation Set".format(option.name))
		plt.xlabel("Key Distance")
		plt.ylabel("Lock Distance")
		plt.savefig("initiation_set_plots/{}/init_clf_episode_{}_{}_has_key_{}.png".format(experiment_name, episode, option.name, has_key))

	generate_plot(has_key=True)
	generate_plot(has_key=False)

def render_sampled_initiation_classifier(option, episode, experiment_name):
	def generate_plot(has_key):
		states = get_grid_states()
		values = get_initiation_set_values(option, has_key=has_key)

		x = np.array([state.position[0] for state in states])
		y = np.array([state.position[1] for state in states])
		xi, yi = np.linspace(x.min(), x.max(), 1000), np.linspace(y.min(), y.max(), 1000)
		xx, yy = np.meshgrid(xi, yi)
		rbf = scipy.interpolate.Rbf(x, y, values, function="linear")
		zz = rbf(xx, yy)
		plt.imshow(zz, vmin=min(values), vmax=max(values), extent=[x.min(), x.max(), y.min(), y.max()], origin="lower", alpha=0.6, cmap=plt.cm.bwr)
		plt.colorbar()

		# Plot trajectories
		positive_examples = option.construct_feature_matrix(option.positive_examples)
		negative_examples = option.construct_feature_matrix(option.negative_examples)
		plt.scatter(positive_examples[:, 0], positive_examples[:, 1], label="positive", cmap=plt.cm.coolwarm, alpha=0.3)

		if negative_examples.shape[0] > 0:
			plt.scatter(negative_examples[:, 0], negative_examples[:, 1], label="negative", cmap=plt.cm.coolwarm, alpha=0.3)

		background_image = imageio.imread("four_room_domain.png")
		plt.imshow(background_image, zorder=0, alpha=0.5, extent=[-2.5, 10., -2.5, 10.])

		name = option.name if episode is None else option.name + "_{}_{}".format(experiment_name, episode)
		plt.title("{} Initiation Set".format(option.name))
		plt.savefig("initiation_set_plots/{}/{}_has_key_{}_initiation_classifier_{}.png".format(experiment_name, name, has_key, option.seed))
		plt.close()

	generate_plot(has_key=True)
	generate_plot(has_key=False)

def make_meshgrid(x, y, h=.02):
	x_min, x_max = x.min() - 1, x.max() + 1
	y_min, y_max = y.min() - 1, y.max() + 1
	xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
						 np.arange(y_min, y_max, h))
	return xx, yy

def plot_one_class_initiation_classifier(option, episode=None, experiment_name=""):
	plt.figure(figsize=(8.0, 5.0))
	flattened_examples = list(itertools.chain.from_iterable(option.positive_examples))
	X = option.construct_aspace_matrix(option.positive_examples)
	Y = option.batched_is_init_true(X)

	pos_predicted_idx = np.where(Y == 1)[0]
	neg_predicted_idx = np.where(Y == -1)[0]
	pos_predicted_examples = [flattened_examples[i] for i in pos_predicted_idx]
	neg_predicted_examples = [flattened_examples[i] for i in neg_predicted_idx]

	def generate_subplot(has_key):
		positive_positions = np.array([example.position for example in pos_predicted_examples if example.has_key == has_key])
		negative_positions = np.array([example.position for example in neg_predicted_examples if example.has_key == has_key])

		plt.subplot(1, 2, int(has_key) + 1)
		if positive_positions.shape[0] > 0:
			plt.scatter(positive_positions[:, 0], positive_positions[:, 1], label="positive")
		if negative_positions.shape[0] > 0:
			plt.scatter(negative_positions[:, 0], negative_positions[:, 1], label="negative")

		plt.xlim((-2, 10))
		plt.ylim((-2, 10))

		background_image = imageio.imread("four_room_domain.png")
		plt.imshow(background_image, zorder=0, alpha=0.5, extent=[-2, 10., -2, 10.])

		plt.xlabel("x")
		plt.ylabel("y")
		plt.legend()
		plt.title("{} has_key = {}".format(option.name, has_key))

	generate_subplot(False)
	generate_subplot(True)

	name = option.name if episode is None else option.name + "_{}_{}".format(experiment_name, episode)
	plt.savefig("initiation_set_plots/{}/{}_{}_one_class_svm.png".format(experiment_name, name, option.seed))
	plt.close()

def plot_two_class_initiation_classifier(option, episode=None, experiment_name=""):
	plt.figure(figsize=(8.0, 5.0))

	positive_feature_matrix = option.construct_aspace_matrix(option.positive_examples)
	negative_feature_matrix = option.construct_aspace_matrix(option.negative_examples)
	flattened_examples = list(itertools.chain.from_iterable(option.positive_examples)) +\
						 list(itertools.chain.from_iterable(option.negative_examples))

	try:
		X = np.concatenate((positive_feature_matrix, negative_feature_matrix))
	except:
		X = positive_feature_matrix
	predictions = option.batched_is_init_true(X)

	pos_predicted_idx = np.where(predictions == 1)[0]
	neg_predicted_idx = np.where(predictions != 1)[0]

	pos_predicted_examples = [flattened_examples[i] for i in pos_predicted_idx]
	neg_predicted_examples = [flattened_examples[i] for i in neg_predicted_idx]

	def generate_subplot(has_key):
		positive_positions = np.array([example.position for example in pos_predicted_examples if example.has_key == has_key])
		negative_positions = np.array([example.position for example in neg_predicted_examples if example.has_key == has_key])

		plt.subplot(1, 2, int(has_key) + 1)

		if positive_positions.shape[0] > 0:
			plt.scatter(positive_positions[:, 0], positive_positions[:, 1], label="positive", c="green")
		if negative_positions.shape[0] > 0:
			plt.scatter(negative_positions[:, 0], negative_positions[:, 1], label="negative", c="black")

		plt.xlim((-2, 10))
		plt.ylim((-2, 10))

		background_image = imageio.imread("four_room_domain.png")
		plt.imshow(background_image, zorder=0, alpha=0.5, extent=[-2, 10., -2, 10.])

		plt.xlabel("x")
		plt.ylabel("y")
		plt.legend()
		plt.title("{} has_key = {}".format(option.name, has_key))

	generate_subplot(False)
	generate_subplot(True)
	name = option.name if episode is None else option.name + "_{}_{}".format(experiment_name, episode)
	plt.savefig("initiation_set_plots/{}/{}_{}_two_class_svm.png".format(experiment_name, name, option.seed))
	plt.close()

def plot_initiation_training_examples(option, episode=None, experiment_name=""):
	getx = lambda l: [x[0] for x in l]
	gety = lambda l: [x[1] for x in l]
	pos = list(itertools.chain.from_iterable(option.positive_examples))
	neg = list(itertools.chain.from_iterable(option.negative_examples))
	position_pos_has_key = [state.position for state in pos if state.has_key == 1]
	position_pos_no_key  = [state.position for state in pos if state.has_key == 0]
	position_neg_has_key = [state.position for state in neg if state.has_key == 1]
	position_neg_no_key  = [state.position for state in neg if state.has_key == 0]
	plt.figure()
	plt.scatter(getx(position_pos_has_key), gety(position_pos_has_key), label="+ has key")
	plt.scatter(getx(position_pos_no_key), gety(position_pos_no_key), label="+ no key")
	plt.scatter(getx(position_neg_has_key), gety(position_neg_has_key), label="- has key")
	plt.scatter(getx(position_neg_no_key), gety(position_neg_no_key), label="- no key")
	plt.legend()
	name = option.name if episode is None else option.name + "_{}_{}".format(experiment_name, episode)
	plt.savefig("initiation_set_plots/{}/{}_{}_svm_training_egs.png".format(experiment_name, name, option.seed))
	plt.close()

def visualize_dqn_replay_buffer(solver, experiment_name=""):
	goal_transitions = list(filter(lambda e: e[2] >= 0 and e[4] == 1, solver.replay_buffer.memory))
	cliff_transitions = list(filter(lambda e: e[2] < 0 and e[4] == 1, solver.replay_buffer.memory))
	non_terminals = list(filter(lambda e: e[4] == 0, solver.replay_buffer.memory))

	goal_x = [e[3][0] for e in goal_transitions]
	goal_y = [e[3][1] for e in goal_transitions]
	cliff_x = [e[3][0] for e in cliff_transitions]
	cliff_y = [e[3][1] for e in cliff_transitions]
	non_term_x = [e[3][0] for e in non_terminals]
	non_term_y = [e[3][1] for e in non_terminals]

	# background_image = imread("pinball_domain.png")

	plt.figure()
	plt.scatter(cliff_x, cliff_y, alpha=0.67, label="cliff")
	plt.scatter(non_term_x, non_term_y, alpha=0.2, label="non_terminal")
	plt.scatter(goal_x, goal_y, alpha=0.67, label="goal")
	# plt.imshow(background_image, zorder=0, alpha=0.5, extent=[0., 1., 1., 0.])

	plt.legend()
	plt.title("# transitions = {}".format(len(solver.replay_buffer)))
	plt.savefig("value_function_plots/{}/{}_replay_buffer_analysis.png".format(experiment_name, solver.name))
	plt.close()

def visualize_smdp_updates(global_solver, experiment_name=""):
	smdp_transitions = list(filter(lambda e: isinstance(e.action, int) and e.action > 0, global_solver.replay_buffer.memory))
	negative_transitions = list(filter(lambda e: e.done == 0, smdp_transitions))
	terminal_transitions = list(filter(lambda e: e.done == 1, smdp_transitions))
	positive_start_x = [e.state[0] for e in terminal_transitions]
	positive_start_y = [e.state[1] for e in terminal_transitions]
	positive_end_x = [e.next_state[0] for e in terminal_transitions]
	positive_end_y = [e.next_state[1] for e in terminal_transitions]
	negative_start_x = [e.state[0] for e in negative_transitions]
	negative_start_y = [e.state[1] for e in negative_transitions]
	negative_end_x = [e.next_state[0] for e in negative_transitions]
	negative_end_y = [e.next_state[1] for e in negative_transitions]

	plt.figure(figsize=(8, 5))
	plt.scatter(positive_start_x, positive_start_y, alpha=0.6, label="+s")
	plt.scatter(negative_start_x, negative_start_y, alpha=0.6, label="-s")
	plt.scatter(positive_end_x, positive_end_y, alpha=0.6, label="+s'")
	plt.scatter(negative_end_x, negative_end_y, alpha=0.6, label="-s'")
	plt.legend()
	plt.title("# updates = {}".format(len(smdp_transitions)))
	plt.savefig("value_function_plots/{}/DQN_SMDP_Updates.png".format(experiment_name))
	plt.close()

def visualize_option_end_states_in_pspace(global_solver, experiment_name=""):
	plt.figure(figsize=(8, 5))
	for option_idx in range(1, len(global_solver.trained_options)):
		# Get all the transitions from the replay buffer of the policy over options that concern `option_idx`
		smdp_transitions = list(filter(lambda e: isinstance(e.action, int) and e.action == option_idx, global_solver.replay_buffer.memory))

		# Transitions that satisfy the goal of the MDP
		terminal_transitions = list(filter(lambda e: e.done == 1, smdp_transitions))

		# All other s' in (s, o, r, s') tuples where o is `option_idx`
		negative_transitions = list(filter(lambda e: e.done == 0, smdp_transitions))

		positive_end_x = [e.next_state[0] for e in terminal_transitions]
		positive_end_y = [e.next_state[1] for e in terminal_transitions]
		negative_end_x = [e.next_state[0] for e in negative_transitions]
		negative_end_y = [e.next_state[1] for e in negative_transitions]

		plt.scatter(positive_end_x + negative_end_x, positive_end_y + negative_end_y, alpha=0.1, label="Option {}".format(option_idx))

	plt.legend()
	plt.title("Samples of Option Termination Sets")
	plt.savefig("value_function_plots/{}/DQN_SMDP_Updates.png".format(experiment_name))
	plt.close()

def visualize_option_term_states_in_aspace(option, experiment_name=""):
	plt.figure(figsize=(16, 10))
	terminal_transitions = [transition for transition in option.solver.replay_buffer.memory if transition[-1] == 1]
	terminal_next_states = [transition[3] for transition in terminal_transitions]
	terminal_next_state_matrix = np.array(terminal_next_states)
	feature_idx = PortablePointMazeState.initiation_classifier_feature_indices()
	feature_matrix = terminal_next_state_matrix[:, feature_idx]
	door1_distances = feature_matrix[:, 0]
	door2_distances = feature_matrix[:, 1]
	key_distances   = feature_matrix[:, 2]
	lock_distances  = feature_matrix[:, 3]
	has_keys		= feature_matrix[:, 4]

	plt.subplot(2, 2, 1)
	plt.hist2d(door1_distances, door2_distances, bins=100)
	plt.xlabel("Door 1"); plt.ylabel("Door 2")
	plt.subplot(2, 2, 2)
	plt.hist(key_distances, bins=100)
	plt.xlabel("Key Distance")
	plt.subplot(2, 2, 3)
	plt.hist(lock_distances, bins=100)
	plt.xlabel("Lock Distance")
	plt.subplot(2, 2, 4)
	plt.hist(has_keys, bins=2)
	plt.xlabel("Has Key")
	plt.suptitle("Option {} Sampled Termination Set".format(option.option_idx))
	plt.savefig("value_function_plots/{}/Option{}_SampledTermSet.svg".format(experiment_name, option.option_idx))
	plt.close()


def visualize_next_state_reward_heat_map(solver, episode=None, experiment_name=""):
	next_states = [experience[3] for experience in solver.replay_buffer.memory]
	rewards = [experience[2] for experience in solver.replay_buffer.memory]
	x = np.array([state[0] for state in next_states])
	y = np.array([state[1] for state in next_states])

	plt.scatter(x, y, None, c=rewards, cmap=plt.cm.coolwarm)
	plt.colorbar()
	plt.xlabel("x")
	plt.ylabel("y")
	plt.title("Replay Buffer Reward Heat Map")
	plt.xlim((-2, 10))
	plt.ylim((-2, 10))

	name = solver.name if episode is None else solver.name + "_{}_{}".format(experiment_name, episode)
	plt.savefig("value_function_plots/{}/{}_replay_buffer_reward_map.png".format(experiment_name, name))
	plt.close()
