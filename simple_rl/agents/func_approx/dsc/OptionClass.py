# Python imports.
from __future__ import print_function
import random
import numpy as np
import pdb
from copy import deepcopy
import torch

from sklearn import svm
from sklearn.covariance import EllipticEnvelope
import itertools
from scipy.spatial import distance

# TODO: additional imports 
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegressionCV as LRCV
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
sns.set(color_codes=True)

# Other imports.
from simple_rl.mdp.StateClass import State
from simple_rl.agents.func_approx.ddpg.DDPGAgentClass import DDPGAgent
from simple_rl.agents.func_approx.dsc.utils import Experience

class Option(object):

	def __init__(self, overall_mdp, name, global_solver, lr_actor, lr_critic, ddpg_batch_size, classifier_type="ocsvm",
				 subgoal_reward=0., max_steps=20000, seed=0, parent=None, num_subgoal_hits_required=3, buffer_length=20,
				 dense_reward=False, enable_timeout=True, timeout=100, initiation_period=2,
				 generate_plots=False, device=torch.device("cpu"), writer=None):
		'''
		Args:
			overall_mdp (MDP)
			name (str)
			global_solver (DDPGAgent)
			lr_actor (float)
			lr_critic (float)
			ddpg_batch_size (int)
			classifier_type (str)
			subgoal_reward (float)
			max_steps (int)
			seed (int)
			parent (Option)
			dense_reward (bool)
			enable_timeout (bool)
			timeout (int)
			initiation_period (int)
			generate_plots (bool)
			device (torch.device)
			writer (SummaryWriter)
		'''
		self.name = name
		self.subgoal_reward = subgoal_reward
		self.max_steps = max_steps
		self.seed = seed
		self.parent = parent
		self.dense_reward = dense_reward
		self.initiation_period = initiation_period
		self.enable_timeout = enable_timeout
		self.classifier_type = classifier_type
		self.generate_plots = generate_plots
		self.writer = writer

		self.timeout = np.inf

		# Global option operates on a time-scale of 1 while child (learned) options are temporally extended
		if enable_timeout:
			self.timeout = 1 if name == "global_option" else timeout

		if self.name == "global_option":
			self.option_idx = 0
		elif self.name == "overall_goal_policy":
			self.option_idx = 1
		else:
			self.option_idx = self.parent.option_idx + 1

		print("Creating {} with enable_timeout={}".format(name, enable_timeout))

		random.seed(seed)
		np.random.seed(seed)

		state_size = overall_mdp.state_space_size()
		action_size = overall_mdp.action_space_size()

		solver_name = "{}_ddpg_agent".format(self.name)
		self.global_solver = DDPGAgent(state_size, action_size, seed, device, lr_actor, lr_critic, ddpg_batch_size, name=solver_name) if name == "global_option" else global_solver
		self.solver = DDPGAgent(state_size, action_size, seed, device, lr_actor, lr_critic, ddpg_batch_size, tensor_log=(writer is not None), writer=writer, name=solver_name)

		# Attributes related to initiation set classifiers
		self.num_goal_hits = 0
		self.positive_examples = []
		self.negative_examples = []
		self.experience_buffer = []
		self.initiation_classifier = None

		# TODO: initiation set classifier variables
		self.X = None		
		self.y = None		
		self.tcsvm = None
		self.ocsvm = None
		self.tcsvm_prob = []
		self.ocsvm_prob = []
		self.diff_prob = []

		self.num_subgoal_hits_required = num_subgoal_hits_required
		self.buffer_length = buffer_length

		self.overall_mdp = overall_mdp
		self.final_transitions = []

		# Debug member variables
		self.num_executions = 0
		self.taken_or_not = []
		self.n_taken_or_not = 0

	def __str__(self):
		return self.name

	def __repr__(self):
		return str(self)

	def __hash__(self):
		return hash(self.name)

	def __eq__(self, other):
		if not isinstance(other, Option):
			return False
		return str(self) == str(other)

	def __ne__(self, other):
		return not self == other

	def get_training_phase(self):
		if self.num_goal_hits < self.num_subgoal_hits_required:
			return "gestation"
		if self.num_goal_hits < (self.num_subgoal_hits_required + self.initiation_period):
			return "initiation"
		if self.num_goal_hits == (self.num_subgoal_hits_required + self.initiation_period):
			return "initiation_done"
		return "trained"

	def initialize_with_global_ddpg(self):
		for my_param, global_param in zip(self.solver.actor.parameters(), self.global_solver.actor.parameters()):
			my_param.data.copy_(global_param.data)
		for my_param, global_param in zip(self.solver.critic.parameters(), self.global_solver.critic.parameters()):
			my_param.data.copy_(global_param.data)
		for my_param, global_param in zip(self.solver.target_actor.parameters(), self.global_solver.target_actor.parameters()):
			my_param.data.copy_(global_param.data)
		for my_param, global_param in zip(self.solver.target_critic.parameters(), self.global_solver.target_critic.parameters()):
			my_param.data.copy_(global_param.data)

		# Not using off_policy_update() because we have numpy arrays not state objects here
		for state, action, reward, next_state, done in self.global_solver.replay_buffer.memory:
			if self.is_init_true(state):
				if self.is_term_true(next_state):
					self.solver.step(state, action, self.subgoal_reward, next_state, True)
				else:
					subgoal_reward = self.get_subgoal_reward(next_state)
					self.solver.step(state, action, subgoal_reward, next_state, done)

	def batched_is_init_true(self, state_matrix):
		if self.name == "global_option":
			return np.ones((state_matrix.shape[0]))
		position_matrix = state_matrix[:, :2]
		return self.initiation_classifier.predict(position_matrix) == 1

	def is_init_true(self, ground_state):
		if self.name == "global_option":
			return True
		features = ground_state.features()[:2] if isinstance(ground_state, State) else ground_state[:2]
		return self.initiation_classifier.predict([features])[0] == 1

	def is_term_true(self, ground_state):
		if self.parent is not None:
			return self.parent.is_init_true(ground_state)

		# If option does not have a parent, it must be the goal option or the global option
		assert self.name == "overall_goal_policy" or self.name == "global_option", "{}".format(self.name)
		return self.overall_mdp.is_goal_state(ground_state)

	def add_initiation_experience(self, states):
		assert type(states) == list, "Expected initiation experience sample to be a queue"
		segmented_states = deepcopy(states)
		if len(states) >= self.buffer_length:
			segmented_states = segmented_states[-self.buffer_length:]
		segmented_positions = [segmented_state.position for segmented_state in segmented_states]
		self.positive_examples.append(segmented_positions)

	def add_experience_buffer(self, experience_queue):
		assert type(experience_queue) == list, "Expected initiation experience sample to be a list"
		segmented_experiences = deepcopy(experience_queue)
		if len(segmented_experiences) >= self.buffer_length:
			segmented_experiences = segmented_experiences[-self.buffer_length:]
		experiences = [Experience(*exp) for exp in segmented_experiences]
		self.experience_buffer.append(experiences)

	@staticmethod
	def construct_feature_matrix(examples):
		states = list(itertools.chain.from_iterable(examples))
		return np.array(states)

	def get_distances_to_goal(self, position_matrix):
		if self.parent is None:
			goal_position = self.overall_mdp.goal_position
			return distance.cdist(goal_position[None, ...], position_matrix, "euclidean")

		distances = -self.parent.initiation_classifier.decision_function(position_matrix)
		distances[distances <= 0.] = 0.
		return distances

	@staticmethod
	def distance_to_weights(distances):
		weights = np.copy(distances)
		for row in range(weights.shape[0]):
			if weights[row] > 0.:
				weights[row] = np.exp(-1. * weights[row])
			else:
				weights[row] = 1.
		return weights

	def train_one_class_svm(self):
		assert len(self.positive_examples) == self.num_subgoal_hits_required, "Expected init data to be a list of lists"
		positive_feature_matrix = self.construct_feature_matrix(self.positive_examples)

		# Smaller gamma -> influence of example reaches farther. Using scale leads to smaller gamma than auto.
		self.initiation_classifier = svm.OneClassSVM(kernel="rbf", nu=0.1, gamma="scale")
		self.initiation_classifier.fit(positive_feature_matrix)

	def train_elliptic_envelope_classifier(self):
		assert len(self.positive_examples) == self.num_subgoal_hits_required, "Expected init data to be a list of lists"
		positive_feature_matrix = self.construct_feature_matrix(self.positive_examples)

		self.initiation_classifier = EllipticEnvelope(contamination=0.2)
		self.initiation_classifier.fit(positive_feature_matrix)

	def train_two_class_classifier(self):
		positive_feature_matrix = self.construct_feature_matrix(self.positive_examples)
		negative_feature_matrix = self.construct_feature_matrix(self.negative_examples)
		positive_labels = [1] * positive_feature_matrix.shape[0]
		negative_labels = [0] * negative_feature_matrix.shape[0]

		X = np.concatenate((positive_feature_matrix, negative_feature_matrix))
		Y = np.concatenate((positive_labels, negative_labels))

		# if len(self.negative_examples) >= 10:
		kwargs = {"kernel": "rbf", "gamma": "scale", "class_weight": "balanced"}
		# else:
		# 	kwargs = {"kernel": "linear", "gamma": "scale"}

		# We use a 2-class balanced SVM which sets class weights based on their ratios in the training data
		initiation_classifier = svm.SVC(**kwargs)
		initiation_classifier.fit(X, Y)

		training_predictions = initiation_classifier.predict(X)
		positive_training_examples = X[training_predictions == 1]

		self.initiation_classifier = svm.OneClassSVM(kernel="rbf", nu=0.1, gamma="scale")
		self.initiation_classifier.fit(positive_training_examples)

		self.classifier_type = "tcsvm"

	def train_initiation_classifier(self):
		if self.classifier_type == "ocsvm":
			self.train_one_class_svm()
		elif self.classifier_type == "elliptic":
			self.train_elliptic_envelope_classifier()
		else:
			raise NotImplementedError("{} not supported".format(self.classifier_type))

	def initialize_option_policy(self):
		# Initialize the local DDPG solver with the weights of the global option's DDPG solver
		self.initialize_with_global_ddpg()

		self.solver.epsilon = self.global_solver.epsilon

		# Fitted Q-iteration on the experiences that led to triggering the current option's termination condition
		experience_buffer = list(itertools.chain.from_iterable(self.experience_buffer))
		for experience in experience_buffer:
			state, action, reward, next_state = experience.serialize()
			self.update_option_solver(state, action, reward, next_state)

	def train(self, experience_buffer, state_buffer):
		"""
		Called every time the agent hits the current option's termination set.
		Args:
			experience_buffer (list)
			state_buffer (list)
		Returns:
			trained (bool): whether or not we actually trained this option
		"""
		# TODO: remove
		print("|-> (Option::train): call (train untrained option)")
		print("  |-> num_goal_hits: {}, num_subgoal_hits_required: {} ".format(self.num_goal_hits, self.num_subgoal_hits_required))
		
		self.add_initiation_experience(state_buffer)
		self.add_experience_buffer(experience_buffer)
		self.num_goal_hits += 1

		if self.num_goal_hits >= self.num_subgoal_hits_required:
			# self.train_initiation_classifier()
			# TODO: call new initiation set classifier
			self.train_initiation_set_classifier()
			self.initialize_option_policy()
			return True

	def get_subgoal_reward(self, state):

		if self.is_term_true(state):
			print("~~~~~ Warning: subgoal query at goal ~~~~~")
			return 0.

		# Return step penalty in sparse reward domain
		if not self.dense_reward:
			return -1.

		# Rewards based on position only
		position_vector = state.features()[:2] if isinstance(state, State) else state[:2]

		# For global and parent option, we use the negative distance to the goal state
		if self.parent is None:
			return -0.1 * self.overall_mdp.distance_to_goal(position_vector)

		# For every other option, we use the negative distance to the parent's initiation set classifier
		dist = self.parent.initiation_classifier.decision_function(position_vector.reshape(1, -1))[0]

		# Decision_function returns a negative distance for points not inside the classifier
		subgoal_reward = 0. if dist >= 0 else dist
		return subgoal_reward

	def off_policy_update(self, state, action, reward, next_state):
		""" Make off-policy updates to the current option's low level DDPG solver. """
		assert self.overall_mdp.is_primitive_action(action), "option should be markov: {}".format(action)
		assert not state.is_terminal(), "Terminal state did not terminate at some point"

		# Don't make updates while walking around the termination set of an option
		if self.is_term_true(state):
			print("[off_policy_update] Warning: called updater on {} term states: {}".format(self.name, state))
			return

		# Off-policy updates for states outside tne initiation set were discarded
		if self.is_init_true(state) and self.is_term_true(next_state):
			self.solver.step(state.features(), action, self.subgoal_reward, next_state.features(), True)
		elif self.is_init_true(state):
			subgoal_reward = self.get_subgoal_reward(next_state)
			self.solver.step(state.features(), action, subgoal_reward, next_state.features(), next_state.is_terminal())

	def update_option_solver(self, s, a, r, s_prime):
		""" Make on-policy updates to the current option's low-level DDPG solver. """
		assert self.overall_mdp.is_primitive_action(a), "Option solver should be over primitive actions: {}".format(a)
		assert not s.is_terminal(), "Terminal state did not terminate at some point"

		if self.is_term_true(s):
			print("[update_option_solver] Warning: called updater on {} term states: {}".format(self.name, s))
			return

		if self.is_term_true(s_prime):
			print("{} execution successful".format(self.name))
			self.solver.step(s.features(), a, self.subgoal_reward, s_prime.features(), True)
		elif s_prime.is_terminal():
			print("[{}]: {} is_terminal() but not term_true()".format(self.name, s))
			self.solver.step(s.features(), a, self.subgoal_reward, s_prime.features(), True)
		else:
			subgoal_reward = self.get_subgoal_reward(s_prime)
			self.solver.step(s.features(), a, subgoal_reward, s_prime.features(), False)

	def execute_option_in_mdp(self, mdp, step_number, episode=None):
		"""
		Option main control loop.

		Args:
			mdp (MDP): environment where actions are being taken
			step_number (int): how many steps have already elapsed in the outer control loop.

		Returns:
			option_transitions (list): list of (s, a, r, s') tuples
			discounted_reward (float): cumulative discounted reward obtained by executing the option
		"""
		start_state = deepcopy(mdp.cur_state)
		state = mdp.cur_state

		if self.is_init_true(state):
			option_transitions = []
			total_reward = 0.
			self.num_executions += 1
			num_steps = 0
			visited_states = []

			while not self.is_term_true(state) and not state.is_terminal() and \
					step_number < self.max_steps and num_steps < self.timeout:

				action = self.solver.act(state.features(), evaluation_mode=False)
				reward, next_state = mdp.execute_agent_action(action, option_idx=self.option_idx)

				self.update_option_solver(state, action, reward, next_state)

				# Note: We are not using the option augmented subgoal reward while making off-policy updates to global DQN
				assert mdp.is_primitive_action(action), "Option solver should be over primitive actions: {}".format(action)

				if self.name != "global_option":
					self.global_solver.step(state.features(), action, reward, next_state.features(), next_state.is_terminal())
					self.global_solver.update_epsilon()

				self.solver.update_epsilon()
				option_transitions.append((state, action, reward, next_state))

				total_reward += reward
				state = next_state

				# step_number is to check if we exhaust the episodic step budget
				# num_steps is to appropriately discount the rewards during option execution (and check for timeouts)
				step_number += 1
				num_steps += 1

			# Don't forget to add the final state to the followed trajectory
			visited_states.append(state)

			if self.is_term_true(state):
				self.num_goal_hits += 1

			if self.get_training_phase() == "initiation" and self.name != "global_option":
				self.refine_initiation_set_classifier(visited_states, start_state, state, num_steps, step_number)

			if self.writer is not None:
				self.writer.add_scalar("{}_ExecutionLength".format(self.name), len(option_transitions), self.num_executions)

			return option_transitions, total_reward

		raise Warning("Wanted to execute {}, but initiation condition not met".format(self))

	def refine_initiation_set_classifier(self, visited_states, start_state, final_state, num_steps,
                                      outer_step_number, episode=None):
		if self.is_term_true(final_state):  # success
			positive_states = [start_state] + visited_states[-self.buffer_length:]
			positive_examples = [state.position for state in positive_states]
			self.positive_examples.append(positive_examples)

		elif num_steps == self.timeout:
			negative_examples = [start_state.position]
			self.negative_examples.append(negative_examples)
		else:
			assert final_state.is_terminal() or outer_step_number == self.max_steps, \
				"Hit else case, but {} was not terminal".format(final_state)

		# Refine the initiation set classifier
		if len(self.negative_examples) > 0:
			# self.train_two_class_classifier()
			# TODO: call new initiation set classifier
			self.train_initiation_set_classifier()


	# TODO: utilities
	def make_meshgrid(self, x, y, h=.02):
		"""Create a mesh of points to plot in

		Args:
			x: data to base x-axis meshgrid on
			y: data to base y-axis meshgrid on
			h: stepsize for meshgrid, optional

		Returns:
			X and y mesh grid
		"""
		x_min, x_max = x.min() - 1, x.max() + 1
		y_min, y_max = y.min() - 1, y.max() + 1
		xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                       np.arange(y_min, y_max, h))
		return xx, yy

	# TODO: utilities
	def plot_contours(self, ax, clf, xx, yy, **params):
		"""Plot the decision boundaries for a classifier.

		Args:
			ax: matplotlib axes object
			clf: a classifier
			xx: meshgrid ndarray
			yy: meshgrid ndarray
			params: dictionary of params to pass to contourf, optional
		
		Returns:
			Contour of decision boundary
		"""
		Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
		Z = Z.reshape(xx.shape)
		out = ax.contourf(xx, yy, Z, **params)
		return out

	# TODO: utilities
	def create_plots(self, models, titles, X, y, grid, episode=None):
		"""Creates plots of models.
		
			Args:
				models: tuple of models
				titles: tuple of title names
				X: input
				grid: tuple of grid size 
		"""

		# Set-up grids for plotting.
		fig, sub = plt.subplots(grid)
		plt.subplots_adjust(wspace=0.4, hspace=0.4)

		X0, X1 = X[:, 0], X[:, 1]
		xx, yy = self.make_meshgrid(X0, X1)

		for clf, title, ax in zip(models, titles, sub.flatten()):
			self.plot_contours(ax, clf, xx, yy,
                            cmap=plt.cm.coolwarm, alpha=0.8)
			ax.scatter(X0, X1, c=y, cmap=plt.cm.coolwarm, s=20, edgecolors='k')
			ax.set_xlim(xx.min(), xx.max())
			ax.set_ylim(yy.min(), yy.max())
			ax.set_xlabel('x-coord')
			ax.set_ylabel('y-coord')
			ax.set_xticks(())
			ax.set_yticks(())
			ax.set_title(title)

		# plt.show()
		plt.savefig("notes/classifier_plots/ep-{}-classifier.png".format(episode))
		plt.close()

		# TODO: remove
		print("|-> notes/classifier_plots/ep-{}-classifier.png saved!".format(episode))

	# TODO: Optimistic classifier
	def opt_clf(self, X_, y_):
		"""
		Optimistic classifier that is two-class SVM 
		
		Args:
			X_: Input data
			y_: Input labels
		
		Returns:
			Fitted two-class linear SVM
		"""
		tcsvm = svm.LinearSVC(max_iter=1000)
		
		return tcsvm.fit(X_, y_)
	
	# TODO: Pessimistic classifier
	def pes_clf(self, tcsvm_, X_, y_):
		"""
		Pessimistic classifier that is a one-class SVM
		
		Args:
			tcsmv_: fitted two-class SVM
			X_: Input data
			y_: Input labels

		Returns:
			Fitted one-class SVM on positive classification of two-class SVM
		"""
		# Get subset of inputs that are on the (+) side of the optimistic classifier
		y_pred = tcsvm_.predict(X_)
		X_pos, y_pos, X_neg, y_neg = [], [], [], []
		for i, label in enumerate(y_pred):
			if label == 1:
				X_pos.append(X_[i])
				y_pos.append(y_[i])
			else:
				X_neg.append(X_[i])
				y_neg.append(y_[i])
		X_pos, y_pos, X_neg, y_neg = np.array(X_pos), np.array(y_pos), np.array(X_neg), np.array(y_neg)

		# Fit one-class SVM (non-linear) from (+) subset of inputs from two-class SVM
		ocsvm = svm.OneClassSVM(kernel="rbf", nu=0.1, gamma="scale")
		
		return ocsvm.fit(X_pos, y_pos)

	# TODO: Platt Scalling module
	def platt_scale(self, ocsvm_, X_, train_size, cv_size, outlier=False):
		"""
		Uses Platt Scalling to get probability values from one-class SVM
		
		Args:
			ocsvm_: One-class SVM on positive classification of two-class SVM
			X_: Input data
			train_size: Proportion of data that is included in train split
			cv_size: Number of folds used in cross-validation generator
			outlier: Toggle to set predictions using outlier data
		
		Returns:
			Prediction probabilities on test data
		"""
		
		# Get SVM predictions
		y_SVM_pred = ocsvm_.predict(X_)
		if outlier:
			y_SVM_pred = y_SVM_pred * -1
		
		# Split the data and SVM labels
		X_train, X_test, y_train, y_test = train_test_split(X_, y_SVM_pred, train_size=train_size)

		# Train using logistic regression layer with cross validation 
		lr = LRCV(cv=cv_size)
		lr.fit(X_train, y_train)

		return lr.predict_proba(X_test)

	# TODO: utilities
	def get_examples(self, k):
		"""Gets first k (x,y) states from the global replay buffer"""
		
		# Replay buffer entry: (state, action, reward, next_state, terminal)
		exp_states = [row[0] for row in list(self.global_solver.replay_buffer.memory)][:k]
		
		# Return only (x,y) coordinates
		return [row[:2] for row in exp_states]


	# TODO: main initiation set classifier call
	def train_initiation_set_classifier(self):
		"""Initiation Set Classifier"""
		
		# TODO: remove
		print("|-> (OptionClass::train_initiation_set_classifier): call")

		# If no negative examples, pick first 20 states to be negative
		if not self.negative_examples:
			self.negative_examples.append(self.get_examples(20))

		# Create input and labels
		positive_feature_matrix = self.construct_feature_matrix(
			self.positive_examples)
		negative_feature_matrix = self.construct_feature_matrix(
			self.negative_examples)
		positive_labels = [1] * positive_feature_matrix.shape[0]
		negative_labels = [0] * negative_feature_matrix.shape[0]

		# Save input and labels
		self.X = np.concatenate((positive_feature_matrix[:,:2], negative_feature_matrix[:,:2]))
		self.y = np.concatenate((positive_labels, negative_labels))

		# Fit classifiers
		self.tcsvm = self.opt_clf(self.X, self.y)
		self.ocsvm = self.pes_clf(self.tcsvm, self.X, self.y)

		# Estimations for pessimistic classifier
		prob_inlier = self.platt_scale(self.ocsvm, self.X, 0.90, 5)
		# prob_outlier = platt_scale(one_clf, X, 0.90, 5, outlier=True)

		# TODO: remove
		print("  |-> option: {}".format(self.name))

		# Save probabilities
		self.tcsvm_prob.append(self.tcsvm.score(self.X, self.y))
		self.ocsvm_prob.append(np.average(prob_inlier[:, 1]))
		self.diff_prob.append(abs(self.tcsvm_prob[-1] - self.ocsvm_prob[-1]))

		# Set initiation set classifier
		self.initiation_classifier = self.ocsvm

	def trained_option_execution(self, mdp, outer_step_counter):
		state = mdp.cur_state
		score, step_number = 0., deepcopy(outer_step_counter)
		num_steps = 0
		state_option_trajectory = []

		while not self.is_term_true(state) and not state.is_terminal()\
				and step_number < self.max_steps and num_steps < self.timeout:
			state_option_trajectory.append((self.option_idx, deepcopy(state)))
			action = self.solver.act(state.features(), evaluation_mode=True)
			reward, state = mdp.execute_agent_action(action, option_idx=self.option_idx)
			score += reward
			step_number += 1
			num_steps += 1
		return score, state, step_number, state_option_trajectory
