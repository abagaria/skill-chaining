# Python imports.
from __future__ import print_function
from collections import defaultdict
import matplotlib.pyplot as plt
import random
from sklearn import svm
import numpy as np
import pdb
from copy import deepcopy
import itertools
import torch
from scipy.spatial import distance
import time

# Other imports.
from simple_rl.mdp.StateClass import State
from simple_rl.agents.func_approx.ddpg.DDPGAgentClass import DDPGAgent
from simple_rl.agents.func_approx.dsc.utils import Experience, render_sampled_initiation_classifier

class Option(object):

	def __init__(self, overall_mdp, name, global_solver, buffer_length=20, pretrained=False,
				 num_subgoal_hits_required=3, subgoal_reward=1., max_steps=20000, seed=0, parent=None, children=[],
                 classifier_type="ocsvm", enable_timeout=True, timeout=250, initiation_period=3, generate_plots=False,
				 device=torch.device("cpu")):
		'''
		Args:
			overall_mdp (MDP)
			name (str)
			global_solver (DDPGAgent)
			buffer_length (int)
			pretrained (bool)
			num_subgoal_hits_required (int)
			subgoal_reward (float)
			max_steps (int)
			seed (int)
			parent (Option)
			children (list)
			classifier_type (str)
			enable_timeout (bool)
			timeout (int)
			initiation_period (int)
			generate_plots (bool)
		'''
		self.name = name
		self.buffer_length = buffer_length
		self.pretrained = pretrained
		self.num_subgoal_hits_required = num_subgoal_hits_required
		self.subgoal_reward = subgoal_reward
		self.max_steps = max_steps
		self.seed = seed
		self.parent = parent
		self.children = children
		self.classifier_type = classifier_type
		self.enable_timeout = enable_timeout
		self.generate_plots = generate_plots

		self.timeout = np.inf

		# Global option operates on a time-scale of 1 while child (learned) options are temporally extended
		if enable_timeout:
			self.timeout = 1 if name == "global_option" else timeout

		self.initiation_period = initiation_period

		if self.name == "global_option":
			self.option_idx = 0
		elif self.name == "overall_goal_policy":
			self.option_idx = 1
		else:
			self.option_idx = self.parent.option_idx + 1

		print("Creating {} with enable_timeout={}".format(name, enable_timeout))

		random.seed(seed)
		np.random.seed(seed)

		if classifier_type == "bocsvm" and parent is None:
			raise AssertionError("{}'s parent cannot be none".format(self.name))

		state_size = overall_mdp.state_space_size()
		action_size = overall_mdp.action_space_size()

		self.global_solver = DDPGAgent(state_size, action_size, seed, device) if name == "global_option" else global_solver
		self.solver = DDPGAgent(state_size, action_size, seed, device)

		if name != "global_option":
			self.initialize_with_global_ddpg()

		self.initiation_classifier = None

		# List of buffers: will use these to train the initiation classifier and the local policy respectively
		self.positive_examples = []
		self.negative_examples = []
		self.experience_buffer = []
		self.final_transitions = []

		self.overall_mdp = overall_mdp
		self.num_goal_hits = 0

		# Debug member variables
		self.num_executions = 0
		self.starting_points = []
		self.ending_points 	 = []

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

	def initialize_with_global_ddpg(self):
		for my_param, global_param in zip(self.solver.actor.parameters(), self.global_solver.actor.parameters()):
			my_param.data.copy_(global_param.data)
		for my_param, global_param in zip(self.solver.critic.parameters(), self.global_solver.critic.parameters()):
			my_param.data.copy_(global_param.data)
		for my_param, global_param in zip(self.solver.target_actor.parameters(), self.global_solver.target_actor.parameters()):
			my_param.data.copy_(global_param.data)
		for my_param, global_param in zip(self.solver.target_critic.parameters(), self.global_solver.target_critic.parameters()):
			my_param.data.copy_(global_param.data)

		for state, action, reward, next_state, done in self.global_solver.replay_buffer.memory:
			if self.is_init_true(state):
				if self.is_term_true(next_state):
					self.solver.step(state, action, self.subgoal_reward, next_state, True)
				else:
					self.solver.step(state, action, reward, next_state, done)

	def distance_to_closest_positive_example(self, state):
		XA = state.features() if isinstance(state, State) else state
		XB = self._construct_feature_matrix(self.positive_examples)
		distances = distance.cdist(XA[None, ...], XB, "euclidean")
		return np.min(distances)

	def batched_is_init_true(self, state_matrix):

		if self.name == "global_option":
			return np.ones((state_matrix.shape[0]))

		if self.classifier_type == "tcsvm":
			svm_predictions = self.initiation_classifier.predict(state_matrix)

			positive_example_matrix = self._construct_feature_matrix(self.positive_examples)
			distance_matrix = distance.cdist(state_matrix, positive_example_matrix, "euclidean")
			closest_distances = np.min(distance_matrix, axis=1)
			distance_predictions = closest_distances < 0.1

			predictions = np.logical_and(svm_predictions, distance_predictions)
			return predictions

		if self.classifier_type == "ocsvm":
			return self.initiation_classifier.predict(state_matrix)

		raise NotImplementedError("Classifier type {} not supported".format(self.classifier_type))

	def is_init_true(self, ground_state):
		# The global option is available everywhere
		if self.name == "global_option":
			return True

		input_features = ground_state.features() if isinstance(ground_state, State) else ground_state

		# Child options are available in learned sub-parts of the state space
		svm_decision = self.initiation_classifier.predict([input_features])[0] == 1

		if self.classifier_type == "ocsvm":
			return svm_decision
		if self.classifier_type == "tcsvm":
			dist = self.distance_to_closest_positive_example(input_features)
			return svm_decision and dist < 0.1
		raise NotImplementedError("Classifier type {} not supported".format(self.classifier_type))

	# TODO: Does it make more sense to return true for entering *any* parent's initiation set
	# e.g, if I enter my grand-parent's initiation set, should that be a terminal transition?
	# TODO: Write a batched version of this function and then use it in the DQNAgentClass
	def is_term_true(self, ground_state):
		if self.parent is not None:
			return self.parent.is_init_true(ground_state)

		# If option does not have a parent, it must be the goal option or the global option
		assert self.name == "overall_goal_policy" or self.name == "global_option", "{}".format(self.name)
		return self.overall_mdp.is_goal_state(ground_state)

	def get_final_positive_examples(self):
		positive_trajectories = self.positive_examples
		return [trajectory[-1] for trajectory in positive_trajectories]

	def get_training_phase(self):
		if self.num_goal_hits < self.num_subgoal_hits_required:
			return "gestation"
		if self.num_goal_hits < (self.num_subgoal_hits_required + self.initiation_period):
			return "initiation"
		if self.num_goal_hits == (self.num_subgoal_hits_required + self.initiation_period):
			return "initiation_done"
		return "trained"

	def add_initiation_experience(self, states):
		"""
		SkillChaining class will give us a queue of states that correspond to its most recently successful experience.
		Args:
			states (list)
		"""
		assert type(states) == list, "Expected initiation experience sample to be a queue"
		starting_idx = int(len(states) // 4)
		segmented_states = states[starting_idx:]

		self.positive_examples.append(segmented_states)

	def add_experience_buffer(self, experience_queue):
		"""
		Skill chaining class will give us a queue of (sars') tuples that correspond to its most recently successful experience.
		Args:
			experience_queue (list)
		"""
		assert type(experience_queue) == list, "Expected initiation experience sample to be a list"

		starting_idx = int(len(experience_queue) // 4)
		segmented_experiences = experience_queue[starting_idx:]

		experiences = [Experience(*exp) for exp in segmented_experiences]
		self.experience_buffer.append(experiences)

	@staticmethod
	def _construct_feature_matrix(examples_matrix):
		states = list(itertools.chain.from_iterable(examples_matrix))
		n_samples = len(states)
		n_features = states[0].get_num_feats()
		X = np.zeros((n_samples, n_features))
		for row in range(X.shape[0]):
			X[row, :] = states[row].features()
		return X

	def train_one_class_svm(self):
		assert len(self.positive_examples) == self.num_subgoal_hits_required, "Expected init data to be a list of lists"
		positive_feature_matrix = self._construct_feature_matrix(self.positive_examples)

		self.initiation_classifier = svm.OneClassSVM(nu=0.01, gamma="scale")
		self.initiation_classifier.fit(positive_feature_matrix)

	def train_two_class_classifier(self):
		positive_feature_matrix = self._construct_feature_matrix(self.positive_examples)
		negative_feature_matrix = self._construct_feature_matrix(self.negative_examples)
		positive_labels = [1] * positive_feature_matrix.shape[0]
		negative_labels = [0] * negative_feature_matrix.shape[0]

		X = np.concatenate((positive_feature_matrix, negative_feature_matrix))
		Y = np.concatenate((positive_labels, negative_labels))

		# We use a 2-class balanced SVM which sets class weights based on their ratios in the training data
		self.initiation_classifier = svm.SVC(kernel="rbf", gamma="scale", class_weight="balanced")
		self.initiation_classifier.fit(X, Y)
		self.classifier_type = "tcsvm"

	def train_initiation_classifier(self):
		self.train_one_class_svm()

	@staticmethod
	def get_center_of_initiation_data(initiation_data):
		initiation_states = list(itertools.chain.from_iterable(initiation_data))
		x_positions = [state.x for state in initiation_states]
		y_positions = [state.y for state in initiation_states]
		x_center = (max(x_positions) + min(x_positions)) / 2.
		y_center = (max(y_positions) + min(y_positions)) / 2.
		return np.array([x_center, y_center])

	def off_policy_update(self, state, action, reward, next_state):
		""" Make off-policy updates to the current option's low level DDPG solver. """
		assert self.overall_mdp.is_primitive_action(action), "option should be markov: {}".format(action)
		assert not state.is_terminal(), "Terminal state did not terminate at some point"

		# Don't make updates while walking around the termination set of an option
		if self.is_term_true(state):
			return

		# Off-policy updates for states outside tne initiation set were discarded
		if self.is_init_true(state) and self.is_term_true(next_state):
			self.solver.step(state.features(), action, self.subgoal_reward, next_state.features(), True)
		elif self.is_init_true(state):
			self.solver.step(state.features(), action, reward, next_state.features(), next_state.is_terminal())

	def update_option_solver(self, s, a, r, s_prime):
		""" Make on-policy updates to the current option's low-level DDPG solver. """
		assert self.overall_mdp.is_primitive_action(a), "Option solver should be over primitive actions: {}".format(a)
		assert not s.is_terminal(), "Terminal state did not terminate at some point"

		if self.is_term_true(s): #, "Don't call on-policy updater on {} term states: {}".format(self.name, s)
			pdb.set_trace()

		if self.pretrained:
			return

		if self.is_term_true(s_prime):
			print("{} execution successful".format(self.name))
			self.solver.step(s.features(), a, self.subgoal_reward, s_prime.features(), True)
		elif s_prime.is_terminal():
			self.solver.step(s.features(), a, r, s_prime.features(), True)
		else:
			self.solver.step(s.features(), a, r, s_prime.features(), False)

	def initialize_option_policy(self):
		# Initialize the local DDPG solver with the weights of the global option's DDPG solver
		self.initialize_with_global_ddpg()

		# TODO: How should we maintain everyone's epsilons?
		self.solver.epsilon = self.global_solver.epsilon

		# Fitted Q-iteration on the experiences that led to triggering the current option's termination condition
		experience_buffer = list(itertools.chain.from_iterable(self.experience_buffer))
		for experience in experience_buffer:
			state, a, r, s_prime = experience.serialize()
			self.off_policy_update(state, a, r, s_prime)

	def train(self, experience_buffer, state_buffer):
		"""
		Called every time the agent hits the current option's termination set.
		Args:
			experience_buffer (list)
			state_buffer (list)

		Returns:
			trained (bool): whether or not we actually trained this option
		"""
		self.add_initiation_experience(state_buffer)
		self.add_experience_buffer(experience_buffer)
		self.num_goal_hits += 1

		if self.num_goal_hits >= self.num_subgoal_hits_required:
			self.train_initiation_classifier()
			self.initialize_option_policy()
			return True
		return False

	def execute_option_in_mdp(self, mdp, step_number):
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
			# print("\rExecuting {}".format(self.name))
			option_transitions = []
			visited_states = []
			total_reward = 0.
			self.num_executions += 1
			num_steps = 0

			while not self.is_term_true(state) and not state.is_terminal() and \
					step_number < self.max_steps and num_steps < self.timeout:

				action = self.solver.act(state.features(), evaluation_mode=False)
				reward, next_state = mdp.execute_agent_action(action, option_idx=self.option_idx)

				if not self.pretrained:
					self.update_option_solver(state, action, reward, next_state)

				# Note: We are not using the option augmented subgoal reward while making off-policy updates to global DQN
				assert mdp.is_primitive_action(action), "Option solver should be over primitive actions: {}".format(action)

				if self.name != "global_option":
					self.global_solver.step(state.features(), action, reward, next_state.features(), next_state.is_terminal())
					self.global_solver.update_epsilon()

				self.solver.update_epsilon()
				option_transitions.append((state, action, reward, next_state))
				visited_states.append(state)

				total_reward += reward
				state = next_state

				# step_number is to check if we exhaust the episodic step budget
				# num_steps is to appropriately discount the rewards during option execution (and check for timeouts)
				step_number += 1
				num_steps += 1

			# Don't forget to add the final state to the followed trajectory
			visited_states.append(state)

			if self.get_training_phase() == "initiation":
				self.refine_initiation_set_classifier(visited_states, start_state, state, num_steps, step_number)

			if self.solver.tensor_log:
				self.solver.writer.add_scalar("ExecutionLength", len(option_transitions), self.num_executions)

			return option_transitions, total_reward

		raise Warning("Wanted to execute {}, but initiation condition not met".format(self))

	def refine_initiation_set_classifier(self, visited_states, start_state, final_state, num_steps, outer_step_number):
		if self.is_term_true(final_state):  # success
			self.num_goal_hits += 1
			if not self.pretrained:
				positive_states = [start_state] + visited_states[-self.buffer_length:]
				self.positive_examples.append(positive_states)

		elif num_steps == self.timeout:
			negative_examples = [start_state]
			if self.parent is not None and len(self.parent.positive_examples) > 0:
				parent_sampled_negative = random.choice(self.parent.get_final_positive_examples())
				negative_examples.append(parent_sampled_negative)
			self.negative_examples.append(negative_examples)
		else:
			assert final_state.is_terminal() or outer_step_number == self.max_steps,\
				"Hit else case, but {} was not terminal".format(final_state)

		# Refine the initiation set classifier
		if len(self.negative_examples) > 0:
			self.train_two_class_classifier()
			if self.generate_plots:
				if self.name != "global_option":
					render_sampled_initiation_classifier(self)
				else:
					render_sampled_initiation_classifier(self)

	def trained_option_execution(self, mdp, outer_step_counter):
		state = mdp.cur_state
		score, step_number = 0., deepcopy(outer_step_counter)
		num_steps = 0

		while not self.is_term_true(state) and not state.is_terminal()\
				and not state.is_out_of_frame() and step_number < self.max_steps and num_steps < self.timeout:
			action = self.solver.act(state.features(), evaluation_mode=True)
			reward, state = mdp.execute_agent_action(action, option_idx=self.option_idx)
			score += reward
			step_number += 1
			num_steps += 1
		return score, state, step_number
