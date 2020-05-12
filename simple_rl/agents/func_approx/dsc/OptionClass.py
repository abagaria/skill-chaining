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
				 generate_plots=False, device=torch.device("cpu"), writer=None, opt_nu=0.5, pes_nu=0.5, experiment_name=None, discrete_actions=False,
				 tensor_log=False, use_old=False, episode=0, option_idx=None):
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
			opt_nu (float)
			pes_nu (float)
			experiment_name (str)
			discrete_actions (bool)
			tensor_log (bool)
			lr_dqn (float)
			use_old (bool)
			episode (int)
			option_idx (int)
			
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
		self.opt_nu = opt_nu
		self.pes_nu = pes_nu
		self.experiment_name = experiment_name
		self.discrete_actions = discrete_actions
		self.tensor_log = tensor_log
		self.lr_dqn = lr_dqn
		self.use_old = use_old
		
		# Global option operates on a time-scale of 1 while child (learned) options are temporally extended
		if enable_timeout:
			self.timeout = 1 if name == "global_option" else timeout

		# TODO: changed indexing due to chain fixing options
		if self.name == "global_option":
			self.option_idx = 0
		else:
			self.option_idx = option_idx

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
		self.episode = episode
		self.X_pes = None

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

	def set_positive_examples(self, positive_examples):
		self.positive_examples = positive_examples

	# TODO: parent mutator
	def update_parent(self, parent):
		self.parent = parent

	# TODO: episode mutator
	def update_episode(self, episode):
		self.episode = episode

	def get_training_phase(self):
		if self.num_goal_hits < self.num_subgoal_hits_required:
			return "gestation"
		if self.num_goal_hits < (self.num_subgoal_hits_required + self.initiation_period):
			return "initiation"
		if self.num_goal_hits == (self.num_subgoal_hits_required + self.initiation_period):
			return "initiation_done"
		return "trained"

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

	def batched_is_init_true(self, state_matrix):
		if self.name == "global_option":
			return np.ones((state_matrix.shape[0]))

		# TODO: hack for treasure game domain
		if "treasure" in self.overall_mdp.env_name:
			states = state_matrix
		else:
			states = state_matrix[:, :2]
		
		if self.use_old:	# TODO: old toggle
			assert self.initiation_classifier != None, "OptionClass::batched_is_init_true: {}'s  initiation_classifier needs to be trained before initiating".format(self.name)
			return self.initiation_classifier.predict(states) == 1
		else:	# TODO: robust DSC
			assert self.optimistic_classifier != None, "OptionClass::batched_is_init_true: {}'s optimistic_classifier needs to be trained before initiating".format(self.name)
			return self.optimistic_classifier.predict(states) == 1

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

		if self.use_old:	# TODO: old toggle
			assert self.initiation_classifier != None, "OptionClass::is_init_true: {}'s initiation_classifier needs to be trained before initiating".format(self.name)
			return self.initiation_classifier.predict([state])[0] == 1
		else:	# TODO: robust DSC
			assert self.optimistic_classifier != None, "OptionClass::is_init_true: {}'s optimistic_classifier needs to be trained before initiating".format(self.name)
			return self.optimistic_classifier.predict([state])[0] == 1

	def is_term_true(self, ground_state):
		if self.use_old:	# TODO: old toggle
			if self.parent is not None:
				return self.parent.is_init_true(ground_state)

			# If option does not have a parent, it must be the goal option or the global option
			# assert self.name == "overall_goal_policy" or self.name == "global_option", "{}".format(self.name)
			return self.overall_mdp.is_goal_state(ground_state)
		else:	# TODO: robust DSC
			# TODO: hack for treasure game domain
			if "treasure" in self.overall_mdp.env_name:
				state = ground_state.features() if isinstance(ground_state, State) else ground_state
				state = np.array(state)
			else:
				state = ground_state.features()[:2] if isinstance(ground_state, State) else ground_state[:2]
				state = np.array(state)
			
			if self.parent is not None:
				# untrained option will not have any trained classifiers so use parent init until trained
				if self.pessimistic_classifier is None:
					return self.parent.is_init_true(ground_state)
				
				# edge case where parent isn't trained (w/ chain fix)
				if self.parent.get_training_phase() == 'gestation':
					return self.pessimistic_classifier.predict([state])[0] == 1
				
				# if parent and child are trained, check if parent's opt clf and current pes clf overlap
				if self.pessimistic_classifier is not None:
					return self.parent.is_init_true(ground_state) and self.pessimistic_classifier.predict([state])[0] == 1

			# otherwise, goal or global option
			return self.overall_mdp.is_goal_state(ground_state)

	# TODO: needed for new term method
	def batched_is_term_true(self, state_matrix):
		if self.use_old:	# TODO: old toggle	
			if self.parent is not None:
				return self.parent.batched_is_init_true(state_matrix)
			
			return self.overall_mdp.batched_is_goal_state(state_matrix)
		else:	# TODO: robust DSC
			# TODO: hack for treasure game domain
			if "treasure" in self.overall_mdp.env_name:
				states = state_matrix
			else:
				states = state_matrix[:, :2]

			if self.parent is not None:
				# untrained option will not have any trained classifiers so use parent init until trained
				if self.pessimistic_classifier is None:
					return self.parent.batched_is_init_true(state_matrix)
				
				# edge case where parent isn't trained (w/ chain fix)
				if self.parent.get_training_phase() == 'gestation':
					return self.pessimistic_classifier.predict(states)[0] == 1
				else:
					return self.parent.batched_is_init_true(state_matrix)

				# if parent and child trained, check if parent's opt clf and current pes overlap
				if self.pessimistic_classifier is not None:
					return np.array(self.parent.batched_is_init_true(states) * self.pessimistic_classifier.predict(states) == 1, dtype=bool)

			# otherwise, goal or gobal option
			return self.overall_mdp.batched_is_goal_state(state_matrix)
			

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
		self.X = positive_feature_matrix = self.construct_feature_matrix(self.positive_examples)

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
		if self.negative_examples == []:
			self.negative_examples.append(self.get_neg_examples(1)) # TODO: edge case where there's no negative samples, just add 1 at random
		negative_feature_matrix = self.construct_feature_matrix(self.negative_examples)
		positive_labels = [1] * positive_feature_matrix.shape[0]
		negative_labels = [0] * negative_feature_matrix.shape[0]

		# TODO: added class reference
		self.X = X = np.concatenate((positive_feature_matrix, negative_feature_matrix))
		self.y = Y = np.concatenate((positive_labels, negative_labels))

		# if len(self.negative_examples) >= 10:
		kwargs = {"kernel": "rbf", "gamma": "scale", "class_weight": "balanced"}
		# else:
		# 	kwargs = {"kernel": "linear", "gamma": "scale"}

		# We use a 2-class balanced SVM which sets class weights based on their ratios in the training data
		initiation_classifier = svm.SVC(**kwargs)
		initiation_classifier.fit(X, Y)

		training_predictions = initiation_classifier.predict(X)
		positive_training_examples = self.X[training_predictions == 1]

		if len(positive_training_examples) > 0:
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

		# TODO: (future work) maybe don't set option solver to global since this will hinder chain fix options that need to fix for far salient events?
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
		print("    |-> Option::train: call ({})".format(self.name))
		
		self.add_initiation_experience(state_buffer)
		self.add_experience_buffer(experience_buffer)
		self.num_goal_hits += 1

		# TODO: remove
		print("      |-> num_goal_hits: {}, num_subgoal_hits_required: {} ".format(
			self.num_goal_hits, self.num_subgoal_hits_required))

		if self.num_goal_hits >= self.num_subgoal_hits_required:
			if self.use_old:	# TODO: old toggle
				self.old_train_initiation_classifier()
			else:	# TODO: robust DSC
				self.train_initiation_classifiers()
			self.initialize_option_policy()
			return True
		
		return False

	def get_subgoal_reward(self, state):

		if self.is_term_true(state):
			print("~~~~~ Warning: subgoal query at goal ~~~~~")
			return 0.

		# Return step penalty in sparse reward domain
		if not self.dense_reward:
			return -1.

		# TODO: hack for treasure game domain
		if "treasure" in self.overall_mdp.env_name:
			position_vector = state.features() if isinstance(state, State) else state
		else:
			# Rewards based on position only
			position_vector = state.features()[:2] if isinstance(state, State) else state[:2]
			

		# For global and parent option, we use the negative distance to the goal state
		if self.parent is None:
			return -0.1 * self.overall_mdp.distance_to_goal(position_vector)

		# For every other option, we use the negative distance to the parent's initiation set classifier
		if self.use_old:
			dist = self.parent.initiation_classifier.decision_function(position_vector.reshape(1, -1))[0]
		else:
			dist = self.parent.optimistic_classifier.decision_function(position_vector.reshape(1, -1))[0]

		# Decision_function returns a negative distance for points not inside the classifier
		subgoal_reward = 0. if dist >= 0 else dist
		return subgoal_reward

	def off_policy_update(self, state, action, reward, next_state):
		""" Make off-policy updates to the current option's low level solver. """
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
				print("[{}]: {} is_terminal() but not term_true()".format(self.name, s_prime))
				self.solver.step(s.features(), a, self.subgoal_reward, s_prime.features(), True, -1)
			else:
				subgoal_reward = self.get_subgoal_reward(s_prime)
				self.solver.step(s.features(), a, subgoal_reward, s_prime.features(), False, -1)
		else:
			if self.is_term_true(s_prime):
				print("{} execution successful".format(self.name))
				self.solver.step(s.features(), a, self.subgoal_reward, s_prime.features(), True)
			elif s_prime.is_terminal():
				print("[{}]: {} is_terminal() but not term_true()".format(self.name, s_prime))
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
				if self.name != "global_option":
					if self.discrete_actions:
						self.global_solver.step(state.features(), action, reward, next_state.features(), next_state.is_terminal(), -1)
					else:
						self.global_solver.step(state.features(), action, reward, next_state.features(), next_state.is_terminal())
					
					# TODO: (future work) dynamic epsilon schedule
					# - scale relative to number of steps per episode
					# - reduce global epsilon by the number of goals hits if in gestation
					# if not self.use_old and self.get_training_phase() == 'gestation':
					# 	self.global_solver.update_gestation_epsilon(self.num_subgoal_hits_required)
					# else:
					# 	self.global_solver.update_epsilon()

					# self.global_solver.update_epsilon()

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

			# TODO: disabled fixed learning
			# if self.get_training_phase() == "initiation" and self.name != "global_option":
			# 	self.refine_option_classifiers(visited_states, start_state, state, num_steps, step_number)
				
			# TODO: continously refine after execution
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
			negative_states = [start_state]
			# TODO: hack for treasure game domain
			if "treasure" in self.overall_mdp.env_name:
				negative_examples = negative_states
				assert np.array(negative_examples).shape[1] == len(self.overall_mdp.init_state.features()), "OptionClass::refine_option_classifiers: Wrong size of state (negative)"
			else:
				negative_examples = [state.position for state in negative_states]
			self.negative_examples.append(negative_examples)
		else:
			assert final_state.is_terminal() or outer_step_number == self.max_steps, \
				"Hit else case, but {} was not terminal".format(final_state)

		# Refine option classifiers
		if len(self.negative_examples) > 0:
			if self.use_old:	# TODO: old toggle
				self.train_two_class_classifier()
			else:	# TODO: robust DSC
				self.train_initiation_classifiers()

	# TODO: utilities
	def get_rand_global_states(self, k):
		"""Gets random k states from the global replay buffer."""
		# Replay buffer entry: (state, action, reward, next_state, terminal)
		exp = random.sample(list(self.global_solver.replay_buffer.memory), k)
		states = np.array([row[0] for row in exp])
		assert states.shape[1] == len(self.overall_mdp.init_state.features()), "OptionClass::get_rand_global_samples: Wrong size of state"
		return states

	# TODO: utilities
	def get_all_global_states(self):
		"""Get the entire global replay buffer."""
		exp_states = [row[0] for row in list(self.global_solver.replay_buffer.memory)]
		states = np.array(exp_states)
		assert states.shape[1] == len(self.overall_mdp.init_state.features()), "OptionClass::get_all_global_samples: Wrong size of state"
		return states

	def get_neg_examples(self, k):
		# If no negative examples, pick k random states to be negative
		if k > len(self.get_all_global_states()):
			k = len(self.get_all_global_states())
		
		# TODO: hack for treasure game domain
		if "treasure" in self.overall_mdp.env_name:
			return np.array(self.get_rand_global_states(k))
		else:
			return np.array(self.get_rand_global_states(k)[:,:2])

	# TODO: train robust initiation set classifiers
	def train_initiation_classifiers(self):
		# create input and labels
		positive_feature_matrix = self.construct_feature_matrix(self.positive_examples)
		positive_labels = [1] * positive_feature_matrix.shape[0]
		negative_feature_matrix = self.construct_feature_matrix(self.negative_examples)
		
		# add parent's pos pes predictions as negative samples (but only for predictions)
		if self.parent is not None and self.parent.get_training_phase() != 'gestation':
			if self.parent.X_pes.ndim != 1:
				if negative_feature_matrix.ndim == 1:
					negative_feature_matrix = self.parent.X_pes
				else:
					negative_feature_matrix = np.concatenate((negative_feature_matrix, self.parent.X_pes))
		
		# edge case where there's no neg samples
		if negative_feature_matrix.ndim == 1:
			k=1
			print("      |-> No negative examples...Adding {} now!".format(k))
			negative_feature_matrix = self.get_neg_examples(k)	
		negative_labels = [0] * negative_feature_matrix.shape[0]
		
		# class reference to data and labels
		self.X = np.concatenate((positive_feature_matrix, negative_feature_matrix))
		self.y = np.concatenate((positive_labels, negative_labels))
		
		# extract positive samples
		X_pos = self.X[self.y == 1]
		if X_pos.ndim == 1:
			X_pos = [X_pos]

		# fit one-class opt clf w/ positive samples 
		self.optimistic_classifier = svm.OneClassSVM(kernel="rbf", nu=self.opt_nu, gamma="scale").fit(X_pos)
		
		# fit two-class opt clf w/ one-class opt clf predictions (-1 or +1 is treated as two classes)
		opt_preds = self.optimistic_classifier.predict(self.X)
		self.optimistic_classifier = svm.SVC(gamma='scale', class_weight='balanced').fit(self.X, opt_preds)
		
		# extract positive samples from opt clf
		opt_preds = self.optimistic_classifier.predict(self.X)
		X_pos = self.X[opt_preds == 1]
		if X_pos.ndim == 1:
			X_pos = [X_pos]

		# fit one-class pes clf w/ positive samples from the opt clf
		self.pessimistic_classifier = svm.OneClassSVM(kernel="rbf", nu=self.pes_nu, gamma="scale").fit(self.X[opt_preds == 1])
		
		# reference for child option to use as pes predictions as negative samples
		pes_preds = self.pessimistic_classifier.predict(X_pos)
		self.X_pes = X_pos[pes_preds == 1]

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
