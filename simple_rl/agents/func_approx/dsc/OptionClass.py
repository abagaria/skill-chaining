# Python imports.
from __future__ import print_function
import random
import numpy as np
import pdb
from copy import deepcopy
import torch
from pathlib import Path

from sklearn import svm
from sklearn.covariance import EllipticEnvelope
import itertools
from scipy.spatial import distance

# TODO: additional imports 
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegressionCV as LRCV
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import matplotlib.patches as mpatches
import matplotlib.cm as cm
import seaborn as sns
import pandas as pd

# sns.set(color_codes=True)
sns.set_style("white")

# Debugging imports
import inspect

# Other imports.
from simple_rl.mdp.StateClass import State
from simple_rl.agents.func_approx.ddpg.DDPGAgentClass import DDPGAgent
from simple_rl.agents.func_approx.dqn.DQNAgentClass import DQNAgent
from simple_rl.agents.func_approx.dsc.utils import Experience

class Option(object):

	def __init__(self, overall_mdp, name, global_solver, lr_actor, lr_critic, lr_dqn, ddpg_batch_size, classifier_type="ocsvm",
				 subgoal_reward=0., max_steps=20000, seed=0, parent=None, num_subgoal_hits_required=3, buffer_length=20,
				 dense_reward=False, enable_timeout=True, timeout=100, initiation_period=2,
				 generate_plots=False, device=torch.device("cpu"), writer=None, nu=0.5, experiment_name=None, discrete_actions=False,
				 tensor_log=False, use_old=False, episode=0):
		'''
		Args:
			overall_mdp (MDP)
			name (str)
			global_solver (DDPGAgent/DQNAgent)
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
			nu (float)
			experiment_name (str)
			discrete_actions (bool)
			tensor_log (bool)
			lr_dqn (float)
			use_old (bool)
			episode (int)
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
		self.device = device
		self.writer = writer
		self.timeout = np.inf
		self.nu = nu
		self.experiment_name = experiment_name
		self.discrete_actions = discrete_actions
		self.tensor_log = tensor_log
		self.lr_dqn = lr_dqn
		self.use_old = use_old

		# TODO: step seed from main dsc loop
		# self.step_seed = None

		# Global option operates on a time-scale of 1 while child (learned) options are temporally extended
		if enable_timeout:
			self.timeout = 1 if name == "global_option" else timeout

		if self.name == "global_option":
			self.option_idx = 0
		elif self.name == "overall_goal_policy_option":
			self.option_idx = 1
		else:
			self.option_idx = self.parent.option_idx + 1

		print("Creating {} with enable_timeout={}".format(name, enable_timeout))

		random.seed(seed)
		np.random.seed(seed)

		state_size = overall_mdp.state_space_size()
		action_size = overall_mdp.action_space_size()

		# TODO: use DQN solver if discrete actions
		if self.discrete_actions:
			if name == "global_option":
				print("|-> ", end="")
				self.global_solver = DQNAgent(state_size=state_size, action_size=action_size,
											trained_options=[], seed=self.seed,
											lr=self.lr_dqn, name="(global_solver) DQN-Agent-{}".format(self.name),
											eps_start=1.0, tensor_log=self.tensor_log,
											use_double_dqn=True, writer=self.writer,
											device=self.device, for_option=True,
											batch_size=ddpg_batch_size)
			else:
				self.global_solver = global_solver
			print("|-> ", end="")
			self.solver = DQNAgent(state_size=state_size, action_size=action_size,
										  trained_options=[], seed=self.seed,
										  lr=self.lr_dqn, name="(solver) DQN-Agent-{}".format(self.name),
										  eps_start=1.0, tensor_log=self.tensor_log,
										  use_double_dqn=True, writer=self.writer,
										  device=self.device, for_option=True,
										  batch_size=ddpg_batch_size)
		else:
			if name == "global_option":
				print("|-> ", end="")
				self.global_solver = DDPGAgent(state_size, action_size, seed, self.device, lr_actor, lr_critic, ddpg_batch_size, name="(global_solver) DDPG-Agent-{}".format(self.name))
			else:
				self.global_solver = global_solver
			print("|-> ", end="")
			self.solver = DDPGAgent(state_size, action_size, seed, self.device, lr_actor, lr_critic, ddpg_batch_size, tensor_log=(writer is not None), writer=writer, name="(solver) DDPG-Agent-{}".format(self.name))

		# Attributes related to initiation set classifiers
		self.num_goal_hits = 0
		self.positive_examples = []
		self.negative_examples = []
		self.experience_buffer = []
		self.initiation_classifier = None

		# TODO: opt/pes classifier variables
		self.X = None
		self.y = None
		self.optimistic_classifier = None
		self.pessimistic_classifier = None
		self.approx_pessimistic_classifier = None
		# self.psmod = None
		# self.train_counter = 0
		self.optimistic_classifier_probs = []
		self.pessimistic_classifier_probs = []
		self.X_global = None
		self.chain_break = []
		self.episode = episode

		self.num_subgoal_hits_required = num_subgoal_hits_required
		self.buffer_length = buffer_length

		self.overall_mdp = overall_mdp
		self.final_transitions = []

		# Debug member variables
		self.num_executions = 0
		self.taken_or_not = []
		self.n_taken_or_not = 0

		print()

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

	def update_episode(self, episode):
		self.episode = episode

	# TODO: set random seed
	# def set_step_seed(self, seed):
	# 	self.step_seed = seed

	# TODO: get seeded value
	def get_seeded_random_value(self, seed):
		np.random.seed(seed)
		return np.random.random()

	# TODO: get seeded array
	def get_seeded_random_array(self, seed, len_array):
		np.random.seed(seed)
		return np.random.rand(len_array)

	def get_training_phase(self):
		
		# TODO: use old DSC
		if self.use_old:
			if self.num_goal_hits < self.num_subgoal_hits_required:
				return "gestation"
			if self.num_goal_hits < (self.num_subgoal_hits_required + self.initiation_period):
				return "initiation"
			if self.num_goal_hits == (self.num_subgoal_hits_required + self.initiation_period):
				return "initiation_done"
			return "trained"
		else:
			# TODO: never done training
			if self.num_goal_hits < self.num_subgoal_hits_required:
				return "gestation"
			if self.num_goal_hits < (self.num_subgoal_hits_required + self.initiation_period):
				return "initiation"

	def initialize_with_global_dqn(self):
		for my_param, global_param in zip(self.solver.policy_network.parameters(), self.global_solver.policy_network.parameters()):
			my_param.data.copy_(global_param.data)
		for my_param, global_param in zip(self.solver.target_network.parameters(), self.global_solver.target_network.parameters()):
			my_param.data.copy_(global_param.data)

		# Not using off_policy_update() because we have numpy arrays not state objects here
		for state, action, reward, next_state, done, _ in self.global_solver.replay_buffer.memory:
			if self.is_init_true(state):
				if self.is_term_true(next_state):
					self.solver.step(state, action, self.subgoal_reward, next_state, True, -1)
				else:
					subgoal_reward = self.get_subgoal_reward(next_state)
					self.solver.step(state, action, subgoal_reward, next_state, done, -1)

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

	# TODO: utilities
	def get_average_predict_proba(self, clf, X):
		y_pred = (clf.predict(X) > 0).astype(int)
		
		# Only care about positive examples
		probs = np.multiply(clf.predict_proba(X)[:,1], y_pred)
		return np.average(probs[probs > 0])

	def batched_is_init_true(self, state_matrix):
		if self.name == "global_option":
			return np.ones((state_matrix.shape[0]))

		# TODO: hack for treasure game domain
		if "treasure" in self.overall_mdp.env_name:
			states = state_matrix
		else:
			states = state_matrix[:, :2]
		
		# TODO: use old DSC
		if self.use_old:
			return self.initiation_classifier.predict(states) == 1
		else:
			if self.parent is None:
				return self.optimistic_classifier.predict(states) == 1
			else:
				linked = (self.optimistic_classifier.predict(states) == 1) == (self.parent.batched_is_term_true(states) == 1)
				if not linked.all():
					self.chain_break.append(self.episode) 
				return linked
		
		# TODO: use predict pessimistic clf
		# return self.pessimistic_classifier.predict(position_matrix) == 1
		
		# TODO: use proba predict pessimistic clf
		# return self.get_seeded_random_array(self.step_seed, len(position_matrix)) < self.approx_pessimistic_classifier.predict_proba(position_matrix)[:,-1]

		# TODO: probabilistic batch initation with optimistic classifier
		# return self.get_seeded_random_array(self.step_seed, len(position_matrix)) < self.optimistic_classifier.predict_proba(position_matrix)[:,-1]

	# TODO: added seed
	def is_init_true(self, ground_state):
		if self.name == "global_option":
			return True
		
		# TODO: hack for treasure game domain
		if "treasure" in self.overall_mdp.env_name:
			state = ground_state.features() if isinstance(ground_state, State) else ground_state
			state = np.array(state)
		else:
			state = ground_state.features()[:2] if isinstance(ground_state, State) else ground_state[:2]
			state = np.array(state)

		if self.use_old:
			return self.initiation_classifier.predict([state])[0] == 1
		else:
			if self.parent is None: # goal option
				return self.optimistic_classifier.predict([state])[0] == 1
			else:	# chain must be linked
				linked = self.optimistic_classifier.predict([state])[0] == 1 and self.parent.is_term_true([state]) == True
				if not linked:
					self.chain_break.append(self.episode)
				return linked

			# TODO: use optimistic clf
			# return self.optimistic_classifier.predict([state])[0] == 1

		# TODO use pessimistic clf
		# return self.pessimistic_classifier.predict([state])[0] == 1

		# TODO: use proba predict pessimistic clf
		# return self.get_seeded_random_value(self.step_seed) < self.approx_pessimistic_classifier.predict_proba(state.reshape(1,-1)).flatten()[-1]

		# TODO: probabilistic initiation with optimistic classifier
		# return self.get_seeded_random_value(self.step_seed) < self.optimistic_classifier.predict_proba(state.reshape(1,-1)).flatten()[-1]

	def is_term_true(self, ground_state):
		# TODO: use old DSC
		if self.use_old:
			if self.parent is not None:
				return self.parent.is_init_true(ground_state)
		else:
			# TODO: hack for treasure game domain
			if "treasure" in self.overall_mdp.env_name:
				state = ground_state.features() if isinstance(ground_state, State) else ground_state
				state = np.array(state)
			else:
				state = ground_state.features()[:2] if isinstance(ground_state, State) else ground_state[:2]
				state = np.array(state)

			# TODO: termination with pessimistic classifier
			if (self.parent is not None) and (self.pessimistic_classifier is not None):
				return self.pessimistic_classifier.predict(state.reshape(1,-1))[-1] == 1
			elif self.parent is not None:
				# NOTE: untrained option will not have any trained classifiers so use parent init until trained
				return self.parent.is_init_true(ground_state)
		
		# If option does not have a parent, it must be the goal option or the global option
		# assert self.name == "overall_goal_policy" or self.name == "global_option", "{}".format(self.name)
		return self.overall_mdp.is_goal_state(ground_state)

	# TODO: needed for new term method
	def batched_is_term_true(self, state_matrix):
		# TODO: use old DSC
		if self.use_old:
			if self.parent is not None:
				return self.parent.batched_is_init_true(state_matrix)
		else:
			# TODO: hack for treasure game domain
			if "treasure" in self.overall_mdp.env_name:
				state = state_matrix
			else:
				state = state_matrix[:, :2]

			# TODO: termination with pessimistic classifier
			if (self.parent is not None) and (self.pessimistic_classifier is not None):
				return self.pessimistic_classifier.predict(state) == 1
			elif self.parent is not None:
				return self.parent.batched_is_init_true(state)

		return self.overall_mdp.is_goal_state(state_matrix)

	def add_initiation_experience(self, states):
		assert type(states) == list, "Expected initiation experience sample to be a queue"
		segmented_states = deepcopy(states)
		if len(states) >= self.buffer_length:
			segmented_states = segmented_states[-self.buffer_length:]

		# TODO: hack for treasure domain
		if "treasure" in self.overall_mdp.env_name:
			assert np.array(segmented_states).shape[1] == len(self.overall_mdp.init_state.features()), "OptionClass::add_initiation_experience: Wrong size of state"
			self.positive_examples.append(segmented_states)
		else:
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

	# TODO: old
	def train_one_class_svm(self):
		assert len(self.positive_examples) == self.num_subgoal_hits_required, "Expected init data to be a list of lists"
		positive_feature_matrix = self.construct_feature_matrix(self.positive_examples)

		# Smaller gamma -> influence of example reaches farther. Using scale leads to smaller gamma than auto.
		self.initiation_classifier = svm.OneClassSVM(kernel="rbf", nu=0.1, gamma="scale")
		self.initiation_classifier.fit(positive_feature_matrix)

	# TODO: old
	def train_elliptic_envelope_classifier(self):
		assert len(self.positive_examples) == self.num_subgoal_hits_required, "Expected init data to be a list of lists"
		positive_feature_matrix = self.construct_feature_matrix(self.positive_examples)

		self.initiation_classifier = EllipticEnvelope(contamination=0.2)
		self.initiation_classifier.fit(positive_feature_matrix)

	# TODO: old
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

	# TODO: old
	def old_train_initiation_classifier(self):
		if self.classifier_type == "ocsvm":
			self.train_one_class_svm()
		elif self.classifier_type == "elliptic":
			self.train_elliptic_envelope_classifier()
		else:
			raise NotImplementedError("{} not supported".format(self.classifier_type))

	def initialize_option_policy(self):
		
		# Initialize the local solver with the weights of the global option's solver
		if self.discrete_actions:
			self.initialize_with_global_dqn()
		else:
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
		print("    |-> Option::train: call (train {})".format(self.name))
		
		self.add_initiation_experience(state_buffer)
		self.add_experience_buffer(experience_buffer)
		self.num_goal_hits += 1

		# TODO: remove
		print("      |-> num_goal_hits: {}, num_subgoal_hits_required: {} ".format(
			self.num_goal_hits, self.num_subgoal_hits_required))

		if self.num_goal_hits >= self.num_subgoal_hits_required:
			# TODO: use old DSC
			if self.use_old:
				self.old_train_initiation_classifier()
			else:
				# TODO: call new initiation set classifier
				self.train_initiation_classifier()
			
			self.initialize_option_policy()
			return True
		
		# TODO: use old DSC
		if self.use_old:
			return False

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
			# print("[update_option_solver] Warning: called updater on {} term states: {}".format(self.name, s))
			return

		# TODO: use discrete solver (eg DQN)
		if self.discrete_actions:
			if self.is_term_true(s_prime):
				print("{} execution successful".format(self.name))
				self.solver.step(s.features(), a, self.subgoal_reward, s_prime.features(), True, -1)
			elif s_prime.is_terminal():
				print("[{}]: {} is_terminal() but not term_true()".format(self.name, s))
				self.solver.step(s.features(), a, self.subgoal_reward, s_prime.features(), True, -1)
			else:
				subgoal_reward = self.get_subgoal_reward(s_prime)
				self.solver.step(s.features(), a, subgoal_reward, s_prime.features(), False, -1)
		else:
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

				# TODO: use DQN act
				if self.discrete_actions:
					action = self.solver.act(state.features(), train_mode=True)
				else:
					action = self.solver.act(state.features(), evaluation_mode=False)

				reward, next_state = mdp.execute_agent_action(action, option_idx=self.option_idx)

				self.update_option_solver(state, action, reward, next_state)

				# Note: We are not using the option augmented subgoal reward while making off-policy updates to global DQN
				assert mdp.is_primitive_action(action), "Option solver should be over primitive actions: {}".format(action)

				# TODO: use DQN act
				if self.name != "global_option" and self.discrete_actions:
					self.global_solver.step(state.features(), action, reward, next_state.features(), next_state.is_terminal(), -1)
					self.global_solver.update_epsilon()
				elif self.name != "global_option":
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

			# print("    |-> option_transitions = {}".format(option_transitions))

			# Don't forget to add the final state to the followed trajectory
			visited_states.append(state)

			if self.is_term_true(state):
				self.num_goal_hits += 1

			# TODO: use old DSC
			if self.use_old:
				if self.get_training_phase() == "initiation" and self.name != "global_option":
					self.refine_option_classifiers(visited_states, start_state, state, num_steps, step_number)
			else:
				# TODO: refine after execution (no longer initiation period)
				if self.name != "global_option" and self.get_training_phase() != "gestation":
					self.refine_option_classifiers(
						visited_states, start_state, state, num_steps, step_number)

			if self.writer is not None:
				self.writer.add_scalar("{}_ExecutionLength".format(self.name), len(option_transitions), self.num_executions)

			assert option_transitions != [], "OptionClass::execute_option_in_mdp: If option_transitions is empty then option shouldn't have been picked by solver."
			
			return option_transitions, total_reward

		raise Warning("Wanted to execute {}, but initiation condition not met".format(self.name))

	def refine_option_classifiers(self, visited_states, start_state, final_state, num_steps,
                                      outer_step_number, episode=None):
		if self.is_term_true(final_state):  # success
			positive_states = [start_state] + visited_states[-self.buffer_length:]
			# TODO: hack for treasure game domain
			if "treasure" in self.overall_mdp.env_name:
				assert np.array(positive_states).shape[1] == len(self.overall_mdp.init_state.features()), "OptionClass::refine_option_classifiers: Wrong size of state (positive)"
				positive_examples = positive_states
			else:
				positive_examples = [state.position for state in positive_states]
			self.positive_examples.append(positive_examples)
		elif num_steps == self.timeout:
			# TODO: hack for treasure game domain
			if "treasure" in self.overall_mdp.env_name:
				negative_examples = [start_state]
				assert np.array(negative_examples).shape[1] == len(self.overall_mdp.init_state.features()), "OptionClass::refine_option_classifiers: Wrong size of state (negative)"

			else:
				negative_examples = [start_state.position]
			self.negative_examples.append(negative_examples)
		else:
			assert final_state.is_terminal() or outer_step_number == self.max_steps, \
				"Hit else case, but {} was not terminal".format(final_state)

		# Refine option classifiers
		if len(self.negative_examples) > 0:
			# TODO: use old DSC
			if self.use_old:
				self.train_two_class_classifier()
			else:
				# TODO: call new initiation set classifier
				self.train_initiation_classifier()


	# # TODO: utilities
	# def make_meshgrid(self, x, y, h=.01):
	# 	"""Create a mesh of points to plot in

	# 	Args:
	# 		x: data to base x-axis meshgrid on
	# 		y: data to base y-axis meshgrid on
	# 		h: stepsize for meshgrid, optional

	# 	Returns:
	# 		X and y mesh grid
	# 	"""
	# 	x_min, x_max = x.min() - 1, x.max() + 1
	# 	y_min, y_max = y.min() - 1, y.max() + 1
	# 	xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
    #                    np.arange(y_min, y_max, h))
	# 	return xx, yy

	# # TODO: utilities
	# def plot_contours(self, ax, clf, xx, yy, **params):
	# 	"""Plot the decision boundaries for a classifier.

	# 	Args:
	# 		ax: matplotlib axes object
	# 		clf: a classifier
	# 		xx: meshgrid ndarray
	# 		yy: meshgrid ndarray
	# 		params: dictionary of params to pass to contourf, optional
		
	# 	Returns:
	# 		Contour of decision boundary
	# 	"""
	# 	Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
	# 	Z = Z.reshape(xx.shape)
	# 	out = ax.contourf(xx, yy, Z, **params)
	# 	return out

	# # TODO: utilities
	# def plot_boundary(self, X_global, X_buff, y_labels, clfs, colors, option_id, cnt, experiment_name, alpha):
	# 	# Create plotting dir (if not created)
	# 	path = 'plots/{}/clf_plots'.format(experiment_name)
	# 	Path(path).mkdir(exist_ok=True)
		
	# 	x_coord, y_coord = X_global[:,0], X_global[:,1]
	# 	x_mesh, y_mesh = self.make_meshgrid(x_coord, y_coord, h=0.008)

	# 	patches = []

	# 	# Plot classifier boundaries
	# 	for (clf_name, clf), color in zip(clfs.items(), colors):
	# 		z = clf.predict(np.c_[x_mesh.ravel(), y_mesh.ravel()])
	# 		z = (z.reshape(x_mesh.shape) > 0).astype(int)
	# 		z = np.ma.masked_where(z == 0, z)

	# 		cf = plt.contourf(x_mesh, y_mesh, z, colors=color, alpha=alpha)
	# 		patches.append(mpatches.Patch(color=color, label=clf_name, alpha=alpha))

	# 	cb = plt.colorbar()
	# 	cb.remove()

	# 	# Plot positive examples
	# 	X_pos = X_buff[y_labels == 1]
	# 	x_pos_coord, y_pos_coord = X_pos[:,0], X_pos[:,1]
	# 	pos_color = 'black'
	# 	p = plt.scatter(x_pos_coord, y_pos_coord, marker='+', c=pos_color, label='positive samples', alpha=alpha)
	# 	patches.append(p)

	# 	plt.xticks(())
	# 	plt.yticks(())
	# 	plt.title("Option {} - Classifier boundaries (t = {})".format(option_id, cnt))
	# 	plt.legend(handles=patches, 
	# 			   bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
		
	# 	# Save plots
	# 	plt.savefig("{}/option_{}_{}.png".format(path, option_id, cnt),
    #                 bbox_inches='tight', edgecolor='black')
	# 	plt.close()

	# 	# TODO: remove
	# 	print("      |-> {}/option_{}_{}.png saved!".format(path, option_id, cnt))

	# # TODO: utilities
	# def plot_state_probs(self, X, clfs, option_id, cmaps, cnt, experiment_name):
	# 	# Create plotting dir (if not created)
	# 	path = 'plots/{}/state_prob_plots'.format(experiment_name)
	# 	Path(path).mkdir(exist_ok=True)
		
	# 	# Generate mesh grid
	# 	x_coord, y_coord = X[:, 0], X[:, 1]
	# 	x_mesh, y_mesh = self.make_meshgrid(x_coord, y_coord)

	# 	# Plot state probability estimates
	# 	for (clf_name, clf), cmap in zip(clfs.items(), cmaps):

	# 		fig, ax = plt.subplots()
			
	# 		# Only plot positive predictions
	# 		X_mesh = np.vstack([x_mesh.flatten(), y_mesh.flatten()]).T
	# 		probs = clf.predict_proba(X_mesh)[:, 1]
	# 		preds = (clf.predict(X_mesh) > 0).astype(int)
	# 		probs = np.multiply(probs, preds)

	# 		ax.set_title(
	# 			'Option {} - Probability estimates of {} (t = {})'.format(option_id, clf_name, cnt))
			
	# 		states = ax.pcolormesh(x_mesh, y_mesh, probs.reshape(x_mesh.shape), shading='gouraud', cmap=cmap, vmin=0.0, vmax=1.0)
	# 		cbar = fig.colorbar(states)

	# 		ax.set_xticks(())
	# 		ax.set_yticks(())
			
	# 		# Save plots
	# 		plt.savefig("{}/option_{}_{}_{}.png".format(path, option_id,
	# 					clf_name, cnt), bbox_inches='tight', edgecolor='black')
	# 		plt.close()

	# 		# TODO: remove
	# 		print("      |-> {}/option_{}_{}_{}.png saved!".format(path, option_id, clf_name, cnt))

	# TODO: Optimistic classifier
	def train_optimistic_classifier(self, X, y):
		"""
		Optimistic classifier that is a two-class SVM 
		
		Args:
			X: Input data
			y: Input labels
		
		Returns:
			Fitted two-class linear SVM
		"""
		# return svm.SVC(gamma='scale', probability=True, class_weight='balanced').fit(X, y)
		
		# TODO: exclude probability
		return svm.SVC(gamma='scale', class_weight='balanced').fit(X, y)
	
	# TODO: Pessimistic classifier
	def train_pessimistic_classifier(self, tc_svm, X, y, nu):
		"""
		Pessimistic classifier that is a one-class SVM
		
		Args:
			tc_smv: fitted two-class SVM
			X: Input data
			y: Input labels

		Returns:
			Fitted one-class SVM on positive classification of two-class SVM
		"""
		# Get subset of inputs that are on the (+) side of the optimistic classifier
		y_pred = tc_svm.predict(X)
		X_pos = X[y_pred == 1]
		y_pos = [1] * X_pos.shape[0]

		# Fit one-class SVM (non-linear) from (+) subset of inputs from two-class SVM
		oc_svm = svm.OneClassSVM(kernel="rbf", nu=nu, gamma="scale")
		
		if len(X_pos) != 0:
			return oc_svm.fit(X_pos, y_pos)
		else:
			# No samples
			return False  

	# TODO: classifier convertion 
	def one_class_to_two_class_classifier(self, oc_svm, X):
		y_pred = oc_svm.predict(X)
		try:
			# return svm.SVC(gamma='scale', probability=True, class_weight='balanced').fit(X, y_pred)
			return svm.SVC(gamma='scale', class_weight='balanced').fit(X, y_pred)
		except:
			print("ValueError: The number of classes has to be greater than one; got 1 class")
			print("|-> Flipping random prediction..")
			i = random.randint(0, len(y_pred))
			y_pred[i] = y_pred[i] * -1
			# return svm.SVC(gamma='scale', probability=True, class_weight='balanced').fit(X, y_pred)
			return svm.SVC(gamma='scale', class_weight='balanced').fit(X, y_pred)
	
	# TODO: Platt Scalling module (not accurate)
	# def platt_scale(self, oc_svm, X, train_size, cv_size):
	# 	"""
	# 	Uses Platt Scalling to get probability values from a one-class SVM
		
	# 	Args:
	# 		ocsvm_: One-class SVM on positive classification of two-class SVM
	# 		X_: Input data
	# 		train_size: Proportion of data that is included in train split
	# 		cv_size: Number of folds used in cross-validation generator
	# 		outlier: Toggle to set predictions using outlier data
		
	# 	Returns:
	# 		platt scalling classifier
	# 		prediction probabilities on test data
	# 	"""
		
	# 	# Get SVM predictions
	# 	y_SVM_pred = oc_svm.predict(X)
		
	# 	# Split the data and SVM labels
	# 	X_train, X_test, y_train, y_test = train_test_split(X, y_SVM_pred, train_size=train_size)

	# 	# Train using logistic regression layer with cross validation 
	# 	lr = LRCV(cv=cv_size)
	# 	lr.fit(X_train, y_train)

	# 	return lr, lr.predict_proba(X_test)

	# TODO: utilities
	def get_rand_global_samples(self, k):
		"""Gets random k states from the global replay buffer"""
		# Replay buffer entry: (state, action, reward, next_state, terminal)
		exp = random.sample(list(self.global_solver.replay_buffer.memory), k)
		states = np.array([row[0] for row in exp])
		assert states.shape[1] == len(self.overall_mdp.init_state.features()), "OptionClass::get_rand_global_samples: Wrong size of state"
		return states

	# TODO: utilities
	def get_all_global_samples(self):
		exp_states = [row[0] for row in list(self.global_solver.replay_buffer.memory)]
		states = np.array(exp_states)
		assert states.shape[1] == len(self.overall_mdp.init_state.features()), "OptionClass::get_all_global_samples: Wrong size of state"
		return states

	# TODO: main initiation and termination classifier call
	def train_initiation_classifier(self):		
		# If no negative examples, pick first k states to be negative
		if not self.negative_examples:
			k=5
			print("      |-> No negative examples...Adding {} now!".format(k))
			
			# TODO: hack for treasure game domain
			if "treasure" in self.overall_mdp.env_name:
				self.negative_examples.append(self.get_rand_global_samples(k))
			else:
				self.negative_examples.append(self.get_rand_global_samples(k)[:,:2])
		
		# NOTE: this shouldn't ever happen
		if not self.positive_examples:
			print("      |-> No positive examples!") 

		# Create input and labels
		positive_feature_matrix = self.construct_feature_matrix(
			self.positive_examples)		
		negative_feature_matrix = self.construct_feature_matrix(
			self.negative_examples)
		positive_labels = [1] * positive_feature_matrix.shape[0]
		negative_labels = [0] * negative_feature_matrix.shape[0]
		
		# Save input and labels
		# self.X = np.concatenate((positive_feature_matrix[:,:2], negative_feature_matrix[:,:2]))
		self.X = np.concatenate((positive_feature_matrix, negative_feature_matrix))
		self.y = np.concatenate((positive_labels, negative_labels))

		# Fit classifiers
		self.optimistic_classifier = self.train_optimistic_classifier(self.X, self.y)
		self.pessimistic_classifier = self.train_pessimistic_classifier(self.optimistic_classifier, self.X, self.y, nu=self.nu)
		self.approx_pessimistic_classifier = self.one_class_to_two_class_classifier(self.pessimistic_classifier, self.X)
		
		# NOTE: Update global buffer once
		if self.X_global is None:
			self.X_global = self.get_all_global_samples()
		
		# Update variables		
		# self.optimistic_classifier_probs.append(self.get_average_predict_proba(
		# 	self.optimistic_classifier, self.X_global))
		# self.pessimistic_classifier_probs.append(self.get_average_predict_proba(self.approx_pessimistic_classifier, self.X_global))

		# Platt scalling module
		# self.psmod, _ = self.platt_scale(self.pessimistic_classifier, self.X, 0.90, 5)

		# Plotting variables
		# option_id = self.option_idx
		# num_colors = 100
		# cmaps = [cm.get_cmap('Blues', num_colors), cm.get_cmap('Greens', num_colors)]
		# colors = ['blue', 'green']
		# clfs = {'optimistic_classifier':self.optimistic_classifier,
		# 		'pessimistic_classifier':self.pessimistic_classifier}

		# Plot boundaries of classifiers
		# self.plot_boundary(X_global, self.X, self.y, clfs, colors, option_id, self.train_counter, self.experiment_name, alpha=0.5)

		# Plot state probability estimates
		# self.plot_state_probs(X_global, clfs, option_id, cmaps, self.train_counter, self.experiment_name)

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
