# Python imports.
from __future__ import print_function
import random
import numpy as np
import pdb
from copy import deepcopy
import torch
from sklearn import svm
import itertools

# Other imports.
from simple_rl.tasks.point_maze.PortablePointMazeStateClass import PortablePointMazeState
from simple_rl.agents.func_approx.ddpg.DDPGAgentClass import DDPGAgent
from simple_rl.agents.func_approx.dsc.utils import Experience

INIT_SET_DISTANCE_THRESHOLD = 100
DOOR_OPTION_COMPLETION_DISTANCE = 1.0
DOOR_OPTION_INIT_DISTANCE = 2.0

class Option(object):

	def __init__(self, overall_mdp, name, global_solver, lr_actor, lr_critic, ddpg_batch_size, classifier_type="ocsvm",
				 subgoal_reward=0., max_steps=20000, seed=0, parent=None, num_subgoal_hits_required=3, buffer_length=20,
				 enable_timeout=True, timeout=100, initiation_period=15,  generate_plots=False,
				 device=torch.device("cpu"), writer=None):
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
		self.enable_timeout = enable_timeout
		self.initiation_period = initiation_period
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

		random.seed(seed)
		np.random.seed(seed)

		state_size = self.policy_features(overall_mdp.init_state).shape[0]
		action_size = overall_mdp.action_space_size()
		action_bound = overall_mdp.action_space_bound()

		solver_name = "{}_ddpg_agent".format(self.name)
		self.global_solver = DDPGAgent(state_size, action_size, action_bound, seed, device, lr_actor, lr_critic, ddpg_batch_size, name=solver_name) if name == "global_option" else global_solver
		self.solver = DDPGAgent(state_size, action_size, action_bound, seed, device, lr_actor, lr_critic, ddpg_batch_size, tensor_log=(writer is not None), writer=writer, name=solver_name)

		# Attributes related to initiation set classifiers
		self.num_goal_hits = 0
		self.positive_examples = []
		self.negative_examples = []
		self.parent_positive_examples = []
		self.experience_buffer = []
		self.aspace_initiation_classifier = None
		self.pspace_initiation_classifier = None
		self.num_subgoal_hits_required = num_subgoal_hits_required
		self.buffer_length = buffer_length
		self.two_class_classifier = None
		self.learn_only_pspace_classifiers = False

		# If we are not the global or the goal option, we want to include the positive examples of all options
		# above us as our own negative examples
		parent_option = self.parent
		while parent_option is not None:
			print("Adding {}'s positive examples".format(parent_option.name))
			self.parent_positive_examples += list(itertools.chain.from_iterable(self.parent.positive_examples))
			parent_option = parent_option.parent

		print("Creating {} with enable_timeout={}, buffer_len={}, subgoal_hits={}".format(name, enable_timeout,
																						  self.buffer_length,
																						  self.num_subgoal_hits_required))

		self.overall_mdp = overall_mdp
		self.final_transitions = []

		# After the first time we complete the initiation phase of the option, make sure that it is able
		# to progress to the next training phase of `trained`.
		self.completed_initiation_phase = False

		# Debug member variables
		self.num_executions = 0
		self.starting_points = []
		self.ending_points 	 = []
		self.num_steps_per_execution = []

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

	def clear_for_new_task(self):
		self.pspace_initiation_classifier = None
		self.learn_only_pspace_classifiers = True
		self.completed_initiation_phase = False
		self.num_goal_hits = 0
		self.initiation_period = 2 * self.initiation_period // 3  # Easier to just learn pspace classifier
		self.positive_examples = []
		self.negative_examples = []
		self.experience_buffer = []
		self.classifier_type = "ocsvm"
		self.final_transitions = []
		self.num_executions = 0
		self.starting_points = []
		self.ending_points = []

	# TODO: To learn option policies in agent space, we need to keep around a buffer w/ PortableState objects
	def initialize_with_global_ddpg(self):
		# for my_param, global_param in zip(self.solver.actor.parameters(), self.global_solver.actor.parameters()):
		# 	my_param.data.copy_(global_param.data)
		# for my_param, global_param in zip(self.solver.critic.parameters(), self.global_solver.critic.parameters()):
		# 	my_param.data.copy_(global_param.data)
		# for my_param, global_param in zip(self.solver.target_actor.parameters(), self.global_solver.target_actor.parameters()):
		# 	my_param.data.copy_(global_param.data)
		# for my_param, global_param in zip(self.solver.target_critic.parameters(), self.global_solver.target_critic.parameters()):
		# 	my_param.data.copy_(global_param.data)

		# Not using off_policy_update() because we have numpy arrays not state objects here
		for state, action, reward, next_state, done in self.global_solver.replay_buffer.memory:
			if self.is_init_true(state):
				if self.is_term_true(next_state) or self.overall_mdp.is_goal_state(next_state):
					self.solver.step(state, action, self.subgoal_reward, next_state, True)
				else:
					subgoal_reward = self.get_subgoal_reward(next_state)
					self.solver.step(state, action, subgoal_reward, next_state, done)

	def batched_is_init_true(self, aspace_state_matrix, pspace_state_matrix):
		if self.name == "global_option":
			return np.ones((aspace_state_matrix.shape[0]))

		aspace_feature_idx = PortablePointMazeState.initiation_classifier_feature_indices()
		aspace_feature_matrix = aspace_state_matrix[:, aspace_feature_idx]

		pspace_feature_idx = PortablePointMazeState.pspace_initiation_classifier_feature_indices_in_pspace_state()
		pspace_feature_matrix = pspace_state_matrix[:, pspace_feature_idx]

		aspace_decisions = self.aspace_initiation_classifier.predict(aspace_feature_matrix) == 1
		pspace_decisions = self.pspace_initiation_classifier.predict(pspace_feature_matrix) == 1
		return np.logical_and(aspace_decisions, pspace_decisions)

	def is_init_true(self, ground_state):
		if self.name == "global_option":
			return True

		if isinstance(ground_state, PortablePointMazeState):
			aspace_features = ground_state.initiation_classifier_features()
			pspace_features = ground_state.pspace_initiation_classifier_features()
		else:
			assert ground_state.shape == (26,)
			aspace_features = ground_state[PortablePointMazeState.initiation_classifier_feature_indices()]
			pspace_features = ground_state[PortablePointMazeState.pspace_initiation_classifier_feature_indices_in_flat_state()]

		agent_space_decision = self.aspace_initiation_classifier.predict([aspace_features])[0] == 1
		problem_space_decision = self.pspace_initiation_classifier.predict([pspace_features])[0] == 1
		return agent_space_decision and problem_space_decision

	def is_term_true(self, ground_state):
		if self.parent is not None:
			parent_option = self.parent
			while parent_option is not None:
				if parent_option.is_init_true(ground_state):
					return True
				parent_option = parent_option.parent
			return False
		else:
			# If option does not have a parent, it must be the goal option or the global option
			assert self.name == "overall_goal_policy" or self.name == "global_option", "{}".format(self.name)
			return self.overall_mdp.is_goal_state(ground_state)

	@staticmethod
	def get_features_for_initiation_classifier(segmented_states):
		aspace_features = [segmented_state.initiation_classifier_features() for segmented_state in segmented_states]
		pspace_features = [segmented_state.pspace_initiation_classifier_features() for segmented_state in segmented_states]
		return aspace_features, pspace_features

	def add_initiation_experience(self, states):
		assert type(states) == list, "Expected initiation experience sample to be a queue"
		segmented_states = deepcopy(states)
		if len(states) >= self.buffer_length:
			segmented_states = segmented_states[-self.buffer_length:]
		self.positive_examples.append(segmented_states)

	def add_experience_buffer(self, experience_queue):
		assert type(experience_queue) == list, "Expected initiation experience sample to be a list"
		segmented_experiences = deepcopy(experience_queue)
		if len(segmented_experiences) >= self.buffer_length:
			segmented_experiences = segmented_experiences[-self.buffer_length:]
		experiences = [Experience(*exp) for exp in segmented_experiences]
		self.experience_buffer.append(experiences)

	@staticmethod
	def construct_feature_matrix(examples):
		aspace_feature_list = []
		pspace_feature_list = []
		for trajectory in examples:
			aspace_features, pspace_features = Option.get_features_for_initiation_classifier(trajectory)
			aspace_feature_list.append(aspace_features)
			pspace_feature_list.append(pspace_features)
		aspace_feature_matrix = np.array(list(itertools.chain.from_iterable(aspace_feature_list)))
		pspace_feature_matrix = np.array(list(itertools.chain.from_iterable(pspace_feature_list)))
		return aspace_feature_matrix, pspace_feature_matrix

	def get_source_task_training_phase(self):
		if self.num_goal_hits < self.num_subgoal_hits_required:
			return "gestation"
		if self.num_goal_hits < (self.num_subgoal_hits_required + self.initiation_period) \
				or len(self.negative_examples) <= self.initiation_period:
			return "initiation"
		if self.num_goal_hits >= (self.num_subgoal_hits_required + self.initiation_period) \
				and len(self.negative_examples) > self.initiation_period and not self.completed_initiation_phase:
			self.completed_initiation_phase = True
			return "initiation_done"
		return "trained"

	def get_transfer_task_training_phase(self):
		if self.num_goal_hits < self.num_subgoal_hits_required:
			return "gestation"
		if self.num_goal_hits < (self.num_subgoal_hits_required + self.initiation_period):
			return "initiation"
		if self.num_goal_hits >= (self.num_subgoal_hits_required + self.initiation_period) and \
				not self.completed_initiation_phase:
			self.completed_initiation_phase = True
			return "initiation_done"
		return "trained"

	def get_training_phase(self):
		# Transfer Task
		if self.learn_only_pspace_classifiers:
			return self.get_transfer_task_training_phase()
		# Source Task
		return self.get_source_task_training_phase()

	def train_one_class_svm(self):
		assert len(self.positive_examples) == self.num_subgoal_hits_required, "Expected init data to be a list of lists"
		positive_aspace_feature_matrix, positive_pspace_feature_matrix = self.construct_feature_matrix(self.positive_examples)

		# Smaller gamma -> influence of example reaches farther. Using scale leads to smaller gamma than auto.
		if not self.learn_only_pspace_classifiers:
			self.aspace_initiation_classifier = svm.OneClassSVM(kernel="rbf", nu=0.1, gamma="auto")
			self.aspace_initiation_classifier.fit(positive_aspace_feature_matrix)

		self.pspace_initiation_classifier = svm.OneClassSVM(kernel="rbf", nu=0.1, gamma="auto")
		self.pspace_initiation_classifier.fit(positive_pspace_feature_matrix)

	def train_initiation_classifier(self):
		if self.classifier_type == "ocsvm":
			self.train_one_class_svm()
		else:
			raise NotImplementedError("{} not supported".format(self.classifier_type))

	def train_two_class_classifier(self):
		positive_aspace_matrix, positive_pspace_matrix = self.construct_feature_matrix(self.positive_examples)
		negative_aspace_matrix, negative_pspace_matrix = self.construct_feature_matrix(self.negative_examples)

		# -- Learn Initiation Classifier in Agent-Space
		positive_labels = [1] * positive_aspace_matrix.shape[0]
		negative_labels = [0] * negative_aspace_matrix.shape[0]

		X = np.concatenate((positive_aspace_matrix, negative_aspace_matrix))
		Y = np.concatenate((positive_labels, negative_labels))

		if len(list(itertools.chain.from_iterable(self.positive_examples))) > len(self.negative_examples) >= (self.initiation_period - 1):
			kwargs = {"gamma": "scale", "kernel": "rbf", "class_weight": "balanced", "random_state": self.seed}
		else:
			kwargs = {"gamma": "scale", "kernel": "rbf", "random_state": self.seed}

		if not self.learn_only_pspace_classifiers:
			self.aspace_initiation_classifier = svm.SVC(**kwargs)
			self.aspace_initiation_classifier.fit(X, Y)

		# -- Learn Initiation Set Classifier in Problem-Space
		positive_labels = [1] * positive_pspace_matrix.shape[0]
		negative_labels = [0] * negative_pspace_matrix.shape[0]

		X = np.concatenate((positive_pspace_matrix, negative_pspace_matrix))
		Y = np.concatenate((positive_labels, negative_labels))

		self.pspace_initiation_classifier = svm.SVC(**kwargs)
		self.pspace_initiation_classifier.fit(X, Y)
		self.classifier_type = "tcsvm"

	def initialize_option_policy(self):
		# TODO: Will need a new data structure to hold PortableState objects from global agent's experiences
		# Initialize the local DDPG solver with the weights of the global option's DDPG solver
		# self.initialize_with_global_ddpg()

		# self.solver.epsilon = self.global_solver.epsilon

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
		self.add_initiation_experience(state_buffer)
		self.add_experience_buffer(experience_buffer)
		self.num_goal_hits += 1

		if self.num_goal_hits >= self.num_subgoal_hits_required:
			self.train_initiation_classifier()
			self.initialize_option_policy()
			return True
		return False

	def get_subgoal_reward(self, state):

		if self.is_term_true(state):
			print("~~~~~ Warning: subgoal query at goal ~~~~~")
			return self.subgoal_reward
		else:
			return -1.

	def is_portable(self):
		return "global" not in self.name.lower()

	def policy_features(self, state):
		if not self.is_portable():
			return state.pspace_features()
		return state.aspace_features()

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
			self.solver.step(self.policy_features(state), action, self.subgoal_reward, self.policy_features(next_state), True)
		elif self.is_init_true(state):
			subgoal_reward = self.get_subgoal_reward(next_state)
			self.solver.step(self.policy_features(state), action, subgoal_reward, self.policy_features(next_state), next_state.is_terminal())

	def update_option_solver(self, s, a, r, s_prime):
		""" Make on-policy updates to the current option's low-level DDPG solver. """
		assert self.overall_mdp.is_primitive_action(a), "Option solver should be over primitive actions: {}".format(a)
		assert not s.is_terminal(), "Terminal state did not terminate at some point"

		if self.is_term_true(s):
			print("[update_option_solver] Warning: called updater on {} term states: {}".format(self.name, s))
			return

		if self.is_term_true(s_prime):
			print("{} execution successful".format(self.name))
			self.solver.step(self.policy_features(s), a, self.subgoal_reward, self.policy_features(s_prime), True)
		elif s_prime.is_terminal():
			print("[{}]: {} is_terminal() but not term_true()".format(self.name, s))
			self.solver.step(self.policy_features(s), a, self.subgoal_reward, self.policy_features(s_prime), True)
		else:
			subgoal_reward = self.get_subgoal_reward(s_prime)
			self.solver.step(self.policy_features(s), a, subgoal_reward, self.policy_features(s_prime), False)

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
			option_transitions = []
			total_reward = 0.
			self.num_executions += 1
			num_steps = 0
			visited_states = []
			if "global" not in self.name:
				print("Executing ", self.name)

			while not self.is_term_true(state) and not state.is_terminal() and \
					step_number < self.max_steps and num_steps < self.timeout:

				# if self.get_training_phase() == "initiation":
				# 	action = self.solver.act(state.features(), evaluation_mode=True)
				# else:
				action = self.solver.act(self.policy_features(state), evaluation_mode=False)
				self.solver.update_epsilon()

				reward, next_state = mdp.execute_agent_action(action, option_idx=self.option_idx)

				self.update_option_solver(state, action, reward, next_state)

				# Note: We are not using the option augmented subgoal reward while making off-policy updates to global DQN
				assert mdp.is_primitive_action(action), "Option solver should be over primitive actions: {}".format(action)

				if self.name != "global_option" and self.global_solver is not None:
					self.global_solver.step(state.pspace_features(), action, reward, next_state.pspace_features(), next_state.is_terminal())
					self.global_solver.update_epsilon()

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
			self.num_steps_per_execution.append(num_steps)

			if self.is_term_true(state):
				self.num_goal_hits += 1

			if self.get_training_phase() == "initiation" and self.name != "global_option":
				self.refine_initiation_set_classifier(visited_states, start_state, state, num_steps, step_number)

			if self.writer is not None:
				self.writer.add_scalar("{}_ExecutionLength".format(self.name), len(option_transitions), self.num_executions)

			return option_transitions, total_reward

		raise Warning("Wanted to execute {}, but initiation condition not met".format(self))

	def refine_initiation_set_classifier(self, visited_states, start_state, final_state, num_steps,
										 outer_step_number):
		if self.is_term_true(final_state):  # success
			positive_states = [start_state] + visited_states[-self.buffer_length // 4:]
			self.positive_examples.append(positive_states)

		elif num_steps == self.timeout:
			self.negative_examples.append([start_state])
		else:
			assert final_state.is_terminal() or outer_step_number == self.max_steps, \
				"Hit else case, but {} was not terminal".format(final_state)

		# Refine the initiation set classifier
		if len(self.negative_examples) > 0 and len(self.positive_examples) > 0:
			self.train_two_class_classifier()

	def trained_option_execution(self, mdp, outer_step_counter):
		state = mdp.cur_state
		score, step_number = 0., deepcopy(outer_step_counter)
		num_steps = 0

		while not self.is_term_true(state) and not state.is_terminal()\
				and step_number < self.max_steps and num_steps < self.timeout:
			action = self.solver.act(self.policy_features(state), evaluation_mode=True)
			reward, state = mdp.execute_agent_action(action, option_idx=self.option_idx)
			score += reward
			step_number += 1
			num_steps += 1
		return score, state, step_number

	def get_num_parents(self):
		num_parents = 0
		parent_option = self.parent
		while parent_option is not None:
			num_parents += 1
			parent_option = parent_option.parent
		return num_parents
