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
import torch.nn as nn
from torch import optim
from tqdm import tqdm

# Other imports.
#from simple_rl.mdp.StateClass import State
from simple_rl.agents.func_approx.ddpg.DDPGAgentClass import DDPGAgent
from simple_rl.agents.func_approx.dsc.utils import Experience
from simple_rl.agents.func_approx.dqn.model import ConvInitiationClassifier
from simple_rl.agents.func_approx.dqn.DQNAgentClass import DQNAgent
from simple_rl.tasks.gym.wrappers import LazyFrames


class Option(object):

	def __init__(self, overall_mdp, name, global_solver, lr_actor, lr_critic, ddpg_batch_size, classifier_type="ocbcn",
				 subgoal_reward=0., max_steps=20000, seed=0, parent=None, num_subgoal_hits_required=3, buffer_length=20,
				 dense_reward=False, enable_timeout=True, timeout=100, initiation_period=2, pixel_observations=True,
				 generate_plots=False, device=torch.device("cpu"), writer=None, num_stacks=4):
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
		self.device = device
		self.pixel_observations = pixel_observations
		self.num_stacks = num_stacks

		# TODO: Get rid of this so that it can apply to both DQN and DDPG
		self.lr_actor = lr_actor
		self.lr_critic = lr_critic
		self.batch_size = ddpg_batch_size

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

		if overall_mdp.is_action_space_discrete():
			solver_name = "{}_ddqn_agent".format(self.name)
			self.global_solver = DQNAgent(state_size, action_size, [], seed, device,
										  name=solver_name, pixel_observation=pixel_observations) \
				if name == "global_option" else global_solver
			self.solver = DQNAgent(state_size, action_size, [], seed, device, tensor_log=(writer is not None),
								   writer=writer, name=solver_name, pixel_observation=pixel_observations)
		else:
			solver_name = "{}_ddpg_agent".format(self.name)
			self.global_solver = DDPGAgent(state_size, action_size, seed, device, lr_actor, lr_critic, ddpg_batch_size,
										   name=solver_name, pixel_observation=pixel_observations, num_stacks=4) if name == "global_option" else global_solver
			self.solver = DDPGAgent(state_size, action_size, seed, device, lr_actor, lr_critic, ddpg_batch_size,
									tensor_log=(writer is not None), writer=writer, name=solver_name, pixel_observation=pixel_observations, num_stacks=4)

		# Attributes related to initiation set classifiers
		self.num_goal_hits = 0
		self.positive_examples = []
		self.negative_examples = []
		self.experience_buffer = []
		self.initiation_classifier = None
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

	def initialize_with_global_dqn(self):
		assert isinstance(self.solver, DQNAgent), self.solver
		assert isinstance(self.global_solver, DQNAgent), self.global_solver

		for my_param, global_param in zip(self.solver.policy_network.parameters(), self.global_solver.policy_network.parameters()):
			my_param.data.copy_(global_param.data)
		for my_param, global_param in zip(self.solver.target_network.parameters(), self.global_solver.target_network.parameters()):
			my_param.data.copy_(global_param.data)

		# # Not using off_policy_update() because we have numpy arrays not state objects here
		# for state, action, reward, next_state, done, num_steps in self.global_solver.replay_buffer.memory:
		# 	if self.is_init_true(state):
		# 		if self.is_term_true(next_state):
		# 			self.solver.step(state, action, self.subgoal_reward, next_state, True, num_steps)
		# 		else:
		# 			subgoal_reward = self.get_subgoal_reward(next_state)
		# 			self.solver.step(state, action, subgoal_reward, next_state, done, num_steps)

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
		# for state, action, reward, next_state, done in self.global_solver.replay_buffer.memory:
		# 	print("initialize with global ddpg")
		# 	print(next_state)
		# 	print(type(next_state))
		# 	if self.is_init_true(state):
		# 		if self.is_term_true():
		# 			self.solver.step(state, action, self.subgoal_reward, next_state, True)
		# 		else:
		#
		# 			subgoal_reward = self.get_subgoal_reward(next_state)
		# 			self.solver.step(state, action, subgoal_reward, next_state, done)

	def batched_is_init_true(self, state_matrix):

		# Lazy frame -> numpy array
		if isinstance(state_matrix, LazyFrames):
			state_matrix = np.array(state_matrix)

		if self.name == "global_option":
			return np.ones((state_matrix.shape[0]))

		if self.pixel_observations:
			images = state_matrix[:, -1, :, :]
			tensor = torch.from_numpy(images).to(self.device)
			predictions = self.initiation_classifier.predict(tensor)
			return predictions.cpu().numpy() == 1
		else:
			position_matrix = state_matrix[:, :2]
			return self.initiation_classifier.predict(position_matrix) == 1

	def is_init_true(self, ground_state):
		if self.name == "global_option":
			return True

		if self.pixel_observations:
			ground_state = ground_state.features() if isinstance(ground_state, State) else ground_state

			# Lazy frame -> numpy array
			if isinstance(ground_state, LazyFrames):
				ground_state = np.array(ground_state)

			image = ground_state[-1, :, :]  # (4, 84, 84) -> (84, 84)
			tensor = torch.from_numpy(image).unsqueeze(0).to(self.device)
			label = self.initiation_classifier.predict(tensor) == 1
			return label

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
		if self.pixel_observations:
			print('check good')
			examples = [self.get_features(segmented_state, 1) for segmented_state in segmented_states]
		else:
			examples = [segmented_state.position for segmented_state in segmented_states]
		self.positive_examples.append(examples)

	def get_features(self, state, n):
		assert n <= self.num_stacks, "Number of states desired exceed number of states in stacks"
		return np.array(state.features())[-n, :, :]

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
	def construct_image_feature_matrix(examples):
		states = list(itertools.chain.from_iterable(examples))
		state_matrix = np.array(states)  # (N, 84, 84)
		return state_matrix

	def construct_negative_data_for_one_class_classifier(self):
		""" Randomly sample states from the global solver's replay buffer and assume they are negative examples. """
		states = [np.array(experience[0])[-1, :, :] for experience in self.global_solver.replay_buffer.memory]
		num_positive_examples = len(list(itertools.chain.from_iterable(self.positive_examples)))
		sampled_states = random.sample(states, k=num_positive_examples)
		return np.array(sampled_states)

	def get_distances_to_goal(self, position_matrix):
		if self.parent is None:
			goal_position = self.overall_mdp.goal_position
			return distance.cdist(goal_position[None, ...], position_matrix, "euclidean")

		distances = -self.parent.initiation_classifier.decision_function(position_matrix)
		distances[distances <= 0.] = 0.
		return distances

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

	def train_conv_image_classifier(self, loader, n_epochs):

		# Model
		loss_function = nn.BCEWithLogitsLoss().to(self.device)  # TODO: Add pos_weight to balance classes
		initiation_classifier = ConvInitiationClassifier(device=self.device, seed=self.seed)
		optimizer = optim.Adam(initiation_classifier.parameters())

		initiation_classifier.train()

		for epoch in range(n_epochs):

			random.shuffle(loader)  # Shuffle data for iid
			epoch_loss = []

			for data, label in loader:
				# Move the data to the GPU
				classifier_input = torch.from_numpy(data).unsqueeze(0).to(self.device)
				classifier_label = torch.FloatTensor([label]).unsqueeze(0).to(self.device)

				optimizer.zero_grad()

				logits = initiation_classifier(classifier_input)
				loss = loss_function(logits, classifier_label)

				loss.backward()
				optimizer.step()

				epoch_loss.append(loss.item())

			print("Epoch {} \t Loss: {}".format(epoch, np.mean(epoch_loss)))

		return initiation_classifier

	def train_two_class_image_classifier(self, n_epochs=5):

		# Data
		positive_feature_matrix = self.construct_image_feature_matrix(self.positive_examples)
		negative_feature_matrix = self.construct_image_feature_matrix(self.negative_examples)
		positive_labels = [1] * positive_feature_matrix.shape[0]
		negative_labels = [0] * negative_feature_matrix.shape[0]

		def prepare_data(pos_data, neg_data, pos_labels, neg_labels):
			training_data = []
			for pos_datum, pos_label in zip(pos_data, pos_labels):
				training_data.append((pos_datum, pos_label))
			for neg_datum, neg_label in zip(neg_data, neg_labels):
				training_data.append((neg_datum, neg_label))
			return training_data

		loader = prepare_data(positive_feature_matrix, negative_feature_matrix, positive_labels, negative_labels)

		print("\rTraining 2-class Binary Convolutional Classifier...")
		self.initiation_classifier = self.train_conv_image_classifier(loader, n_epochs)
		self.classifier_type = "tcbcn"

	def train_one_class_image_classifier(self, n_epochs=5):

		# Data
		positive_feature_matrix = self.construct_image_feature_matrix(self.positive_examples)
		negative_feature_matrix = self.construct_negative_data_for_one_class_classifier()
		positive_labels = [1] * positive_feature_matrix.shape[0]
		negative_labels = [0] * negative_feature_matrix.shape[0]

		def prepare_data(pos_data, neg_data, pos_labels, neg_labels):
			training_data = []
			for pos_datum, pos_label in zip(pos_data, pos_labels):
				training_data.append((pos_datum, pos_label))
			for neg_datum, neg_label in zip(neg_data, neg_labels):
				training_data.append((neg_datum, neg_label))
			return training_data

		loader = prepare_data(positive_feature_matrix, negative_feature_matrix, positive_labels, negative_labels)

		print("\rTraining 1-class Binary Convolutional Classifier")
		self.initiation_classifier = self.train_conv_image_classifier(loader, n_epochs)
		self.classifier_type = "ocbcn"

	def train_initiation_classifier(self):
		if self.classifier_type == "ocsvm":
			self.train_one_class_svm()
		elif self.classifier_type == "elliptic":
			self.train_elliptic_envelope_classifier()
		elif self.classifier_type == "ocbcn":
			self.train_one_class_image_classifier()
		elif self.classifier_type == "tcbcn":
			self.train_two_class_image_classifier()
		else:
			raise NotImplementedError("{} not supported".format(self.classifier_type))

	def initialize_option_policy(self):
		# Initialize the local DDPG solver with the weights of the global option's DDPG solver
		if self.overall_mdp.is_action_space_discrete():
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
		self.add_initiation_experience(state_buffer)
		self.add_experience_buffer(experience_buffer)
		self.num_goal_hits += 1

		if self.num_goal_hits >= self.num_subgoal_hits_required:
			self.train_initiation_classifier()
			self.initialize_option_policy()
			return True
		return False

	def get_subgoal_reward(self, state):
		if self.name in ("overall_goal_policy", "global_option"):
			return -1.

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

			while not self.is_term_true(state) and not state.is_terminal() and \
					step_number < self.max_steps and num_steps < self.timeout:

				if isinstance(self.solver, DDPGAgent):
					action = self.solver.act(state.features(), evaluation_mode=False)
				else:
					action = self.solver.act(state.features(), train_mode=True)

				reward, next_state = mdp.execute_agent_action(action)

				self.update_option_solver(state, action, reward, next_state)

				# Note: We are not using the option augmented subgoal reward while making off-policy updates to global DQN
				# assert mdp.is_primitive_action(action), "Option solver should be over primitive actions: {}".format(action)

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
										 outer_step_number):
		if self.is_term_true(final_state):  # success
			positive_states = [start_state] + visited_states[-self.buffer_length:]
			positive_examples = [np.array(state.features())[-1, :, :] for state in positive_states]
			self.positive_examples.append(positive_examples)

		elif num_steps == self.timeout:
			negative_examples = [np.array(start_state.features())[-1, :, :]]
			self.negative_examples.append(negative_examples)
		else:
			assert final_state.is_terminal() or outer_step_number == self.max_steps, \
				"Hit else case, but {} was not terminal".format(final_state)

		# Refine the initiation set classifier
		if len(self.negative_examples) > 0:
			if self.pixel_observations:
				self.train_two_class_image_classifier()
			else:
				self.train_two_class_classifier()

	def trained_option_execution(self, mdp, outer_step_counter):
		state = mdp.cur_state
		score, step_number = 0., deepcopy(outer_step_counter)
		num_steps = 0
		state_option_trajectory = []

		while not self.is_term_true(state) and not state.is_terminal()\
				and step_number < self.max_steps and num_steps < self.timeout:
			state_option_trajectory.append((self.option_idx, deepcopy(state)))
			if isinstance(self.solver, DDPGAgent):
				action = self.solver.act(state.features(), evaluation_mode=True)
			else:
				action = self.solver.act(state.features(), train_mode=False)
			reward, state = mdp.execute_agent_action(action)
			score += reward
			step_number += 1
			num_steps += 1
		return score, state, step_number, state_option_trajectory
