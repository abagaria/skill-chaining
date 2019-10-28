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
from simple_rl.mdp.StateClass import State
from simple_rl.agents.func_approx.ddpg.DDPGAgentClass import DDPGAgent
from simple_rl.agents.func_approx.dsc.utils import Experience
from simple_rl.agents.func_approx.dqn.model import ConvInitiationClassifier
from simple_rl.agents.func_approx.dqn.DQNAgentClass import DQNAgent
from simple_rl.tasks.gym.wrappers import LazyFrames
from simple_rl.agents.func_approx.DeepRandomAgentClass import DeepRandomAgent


class Option(object):

	def __init__(self, overall_mdp, name, global_solver, lr_actor, lr_critic, ddpg_batch_size,
				 option_idx, classifier_type="ocsvm", random_global_agent=True,
				 subgoal_reward=0., max_steps=20000, seed=0, parent=None, num_subgoal_hits_required=3, buffer_length=20,
				 dense_reward=False, enable_timeout=True, timeout=100, initiation_period=10, pixel_observations=True,
				 pixel_initiations=False, generate_plots=False, device=torch.device("cpu"), writer=None):
		'''
		Args:
			overall_mdp (MDP)
			name (str)
			global_solver (DDPGAgent)
			lr_actor (float)
			lr_critic (float)
			ddpg_batch_size (int)
			option_idx (int)
			classifier_type (str)
			subgoal_reward (float)
			max_steps (int)
			seed (int)
			parent (Option)
			dense_reward (bool)
			enable_timeout (bool)
			timeout (int)
			initiation_period (int)
			pixel_observations (bool)
			pixel_initiations (bool)
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
		self.pixel_initiations = pixel_initiations
		self.random_global_agent = random_global_agent

		# TODO: Get rid of this so that it can apply to both DQN and DDPG
		self.lr_actor = lr_actor
		self.lr_critic = lr_critic
		self.batch_size = ddpg_batch_size

		self.timeout = np.inf

		# Global option operates on a time-scale of 1 while child (learned) options are temporally extended
		if enable_timeout:
			self.timeout = 1 if name == "global_option" else timeout

		self.option_idx = option_idx

		print("Creating {} with enable_timeout={}".format(name, enable_timeout))

		random.seed(seed)
		np.random.seed(seed)

		state_size = overall_mdp.state_space_size()
		action_size = overall_mdp.action_space_size()

		if overall_mdp.is_action_space_discrete():
			solver_name = "{}_ddqn_agent".format(self.name)
			if name == "global_option":
				if self.random_global_agent:
					self.global_solver = DeepRandomAgent(overall_mdp.actions, device, pixel_observations, seed)
					self.solver = DeepRandomAgent(overall_mdp.actions, device, pixel_observations, seed)
				else:
					self.global_solver = DQNAgent(state_size, action_size, [], seed, device, tensor_log=(writer is not None),
									   writer=writer, name=solver_name, pixel_observation=pixel_observations)
					self.solver = DQNAgent(state_size, action_size, [], seed, device, tensor_log=(writer is not None),
									   writer=writer, name=solver_name, pixel_observation=pixel_observations)
			else:
				self.global_solver = global_solver
				self.solver = DQNAgent(state_size, action_size, [], seed, device, tensor_log=(writer is not None),
									   writer=writer, name=solver_name, pixel_observation=pixel_observations)
		else:
			solver_name = "{}_ddpg_agent".format(self.name)
			self.global_solver = DDPGAgent(state_size, action_size, seed, device, lr_actor, lr_critic, ddpg_batch_size,
										   name=solver_name) if name == "global_option" else global_solver
			self.solver = DDPGAgent(state_size, action_size, seed, device, lr_actor, lr_critic, ddpg_batch_size,
									tensor_log=(writer is not None), writer=writer, name=solver_name)

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
		if self.num_goal_hits >= (self.num_subgoal_hits_required + self.initiation_period):
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
		for state, action, reward, next_state, done in self.global_solver.replay_buffer.memory:
			if self.is_init_true(state):
				if self.is_term_true(next_state):
					self.solver.step(state, action, self.subgoal_reward, next_state, True)
				else:
					subgoal_reward = self.get_subgoal_reward(next_state)
					self.solver.step(state, action, subgoal_reward, next_state, done)

	def batched_is_init_true(self, state_matrix, position_matrix):

		# Lazy frame -> numpy array
		if isinstance(state_matrix, LazyFrames):
			state_matrix = np.array(state_matrix)

		if self.name == "global_option":
			return np.ones((state_matrix.shape[0]))

		x = position_matrix[:, 0]
		y = position_matrix[:, 1]

		if self.pixel_initiations:
			if self.name == "overall_goal_policy":
				return np.logical_and(np.logical_and(x >= 120, x <= 150), y >= 160)
			if self.name == "option_1":
				return np.logical_and(np.logical_and(x >= 90, x <= 150), y >= 190)
			if self.name == "option_2":
				return np.logical_and(np.logical_and(x >= 60, x <= 100), y >= 230)
			if self.name == "goal_option_2":
				return np.logical_and(x <= 25, y < 180)

			images = state_matrix[:, -1, :, :]
			tensor = torch.from_numpy(images).to(self.device)
			predictions = self.initiation_classifier.predict(tensor)
			return predictions.cpu().numpy() == 1
		else:
			# position_matrix = state_matrix[:, :2]
			return self.initiation_classifier.predict(position_matrix) == 1

	def is_init_true(self, ground_state, position=None):
		if self.name == "global_option":
			return True

		if position is None:
			position = ground_state.position

		if self.pixel_initiations:
			image_features = ground_state.features() if isinstance(ground_state, State) else ground_state

			# Hard coding for experiment
			if self.name == "overall_goal_policy":
				return 120 <= position[0] <= 150 and position[1] >= 170
			if self.name == "option_1":
				return 90 <= position[0] <= 150 and position[1] >= 190
			if self.name == "option_2":
				return 60 <= position[0] <= 100 and position[1] >= 230
			if self.name == "goal_option_2":
				return position[0] <= 25 and position[1] < 180

			# Lazy frame -> numpy array
			if isinstance(image_features, LazyFrames):
				image_features = np.array(image_features)

			image = image_features[-1, :, :]  # (4, 84, 84) -> (84, 84)
			tensor = torch.from_numpy(image).unsqueeze(0).to(self.device)
			label = self.initiation_classifier.predict(tensor) == 1
			return label

		# features = ground_state.features()[:2] if isinstance(ground_state, State) else ground_state[:2]
		return self.initiation_classifier.predict([position])[0] == 1

	def is_term_true(self, ground_state, position=None):

		if position is None:
			position = ground_state.position

		if self.parent is not None:
			return self.parent.is_init_true(ground_state, position)

		# If option does not have a parent, it must be the goal option or the global option
		if self.name == "overall_goal_policy":
			return self.overall_mdp.term_function_1(ground_state)
		if self.name == "global_option":
			return self.overall_mdp.term_function_2(ground_state)
		if self.name == "goal_option_2":
			return self.overall_mdp.term_function_2(ground_state)
		assert self.name == "overall_goal_policy" or self.name == "global_option" or \
			   self.name == "goal_option_2", "{}".format(self.name)
		raise Warning(self.name)

	@staticmethod
	def reject_outliers(data, m=2.):
		d = np.linalg.norm(data - np.median(data, axis=0), axis=1)
		mdev = np.median(d)
		s = d / (mdev if mdev else 1.)
		return data[s < m, :]

	def add_initiation_experience(self, states):
		assert type(states) == list, "Expected initiation experience sample to be a queue"
		segmented_states = deepcopy(states)
		if len(states) >= self.buffer_length:
			segmented_states = segmented_states[-self.buffer_length:]
		if self.pixel_initiations:
			examples = [np.array(segmented_state.features())[-1, :, :] for segmented_state in segmented_states]
		else:
			examples = [segmented_state.position for segmented_state in segmented_states]
		self.positive_examples.append(examples)

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
		states = [np.array(experience.state)[-1, :, :] for experience in self.global_solver.replay_buffer.memory]
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
		positive_feature_matrix = self.reject_outliers(positive_feature_matrix)

		# Smaller gamma -> influence of example reaches farther. Using scale leads to smaller gamma than auto.
		self.initiation_classifier = svm.OneClassSVM(kernel="rbf", nu=0.01, gamma="scale")
		self.initiation_classifier.fit(positive_feature_matrix)

	def train_elliptic_envelope_classifier(self):
		assert len(self.positive_examples) == self.num_subgoal_hits_required, "Expected init data to be a list of lists"
		positive_feature_matrix = self.construct_feature_matrix(self.positive_examples)

		self.initiation_classifier = EllipticEnvelope(contamination=0.2)
		self.initiation_classifier.fit(positive_feature_matrix)

	def train_two_class_classifier(self):
		positive_feature_matrix = self.construct_feature_matrix(self.positive_examples)
		positive_feature_matrix = self.reject_outliers(positive_feature_matrix)
		negative_feature_matrix = self.construct_feature_matrix(self.negative_examples)
		positive_labels = [1] * positive_feature_matrix.shape[0]
		negative_labels = [0] * negative_feature_matrix.shape[0]

		X = np.concatenate((positive_feature_matrix, negative_feature_matrix))
		Y = np.concatenate((positive_labels, negative_labels))

		if self.initiation_period // 2 < len(self.negative_examples) < len(self.positive_examples):
			kwargs = {"kernel": "linear", "gamma": "scale", "class_weight": "balanced"}
		else:
			kwargs = {"kernel": "linear", "gamma": "scale"}

		# We use a 2-class balanced SVM which sets class weights based on their ratios in the training data
		initiation_classifier = svm.SVC(**kwargs)
		initiation_classifier.fit(X, Y)

		training_predictions = initiation_classifier.predict(X)
		positive_training_examples = X[training_predictions == 1]

		if positive_training_examples.shape[0] > 0:
			self.initiation_classifier = svm.OneClassSVM(kernel="rbf", nu=0.01, gamma="scale")
			self.initiation_classifier.fit(positive_training_examples)
			self.classifier_type = "tcsvm"

	def get_pos_weight(self, loader):
		""" How much more important is it to correctly classify +ive data vs -ive data. """
		num_positive_examples = len([1 for data, label in loader if label == 1])
		num_negative_examples = len([1 for data, label in loader if label != 1])

		# We want the classification of negative data to be at most 10 times more important
		ratio = max(num_negative_examples / num_positive_examples, 0.1)
		return torch.tensor([ratio], dtype=torch.float)

	def train_conv_image_classifier(self, loader, n_epochs):

		# Model
		pos_label_weight = self.get_pos_weight(loader)
		print("Positive data weight = {}".format(pos_label_weight.item()))
		loss_function = nn.BCEWithLogitsLoss(pos_weight=pos_label_weight).to(self.device)
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
		# if not self.random_global_agent:
		# 	if self.overall_mdp.is_action_space_discrete():
		# 		self.initialize_with_global_dqn()
		# 	else:
		# 		self.initialize_with_global_ddpg()

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
			return -1.  # STEP PENALTY

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

	def update_option_solver(self, s, a, r, s_prime):
		""" Make on-policy updates to the current option's low-level DDPG solver. """
		assert self.overall_mdp.is_primitive_action(a), "Option solver should be over primitive actions: {}".format(a)
		assert not s.is_terminal(), "Terminal state did not terminate at some point"

		if self.is_term_true(s):
			print("[update_option_solver] Warning: called updater on {} term states: {}".format(self.name, s))
			return

		if self.is_term_true(s_prime):
			print("{} execution successful".format(self.name))
			self.solver.step(s.features(), s.position, a, self.subgoal_reward, s_prime.features(), s_prime.position, True)
		elif s_prime.is_terminal():  # This could be something like "game_over"
			self.solver.step(s.features(), s.position, a, r, s_prime.features(), s_prime.position, True)
		else:
			subgoal_reward = self.get_subgoal_reward(s_prime)
			self.solver.step(s.features(), s.position, a, subgoal_reward, s_prime.features(), s_prime.position, False)

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
			if self.name != 'global_option':
				print("\r Executing {}".format(self.name))
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
					action = self.solver.act(state.features(), state.position, train_mode=True)

				reward, next_state = mdp.execute_agent_action(action)

				self.update_option_solver(state, action, reward, next_state)

				# Note: We are not using the option augmented subgoal reward while making off-policy updates to global DQN
				# assert mdp.is_primitive_action(action), "Option solver should be over primitive actions: {}".format(action)

				if self.name != "global_option":
					self.global_solver.step(state.features(), state.position, action, reward, next_state.features(),
											next_state.position, next_state.is_terminal())
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
			positive_states = [start_state] + visited_states[-self.buffer_length: ]

			if self.name == "overall_goal_policy":
				term_states = [trajectory[-1].next_state for trajectory in self.experience_buffer]
				sampled_negative_state = random.choice(term_states)
				sampled_negative_example = sampled_negative_state.position
				if sampled_negative_example not in list(itertools.chain.from_iterable(self.negative_examples)):
					self.negative_examples.append([sampled_negative_example])

			if self.pixel_initiations:
				positive_examples = [np.array(state.features())[-1, :, :] for state in positive_states]
			else:
				positive_examples = [state.position for state in positive_states]

			self.positive_examples.append(positive_examples)

		elif num_steps == self.timeout:
			if self.pixel_initiations:
				negative_examples = [np.array(start_state.features())[-1, :, :]]
			else:
				negative_examples = [start_state.position]
			self.negative_examples.append(negative_examples)
		else:
			assert final_state.is_terminal() or outer_step_number == self.max_steps, \
				"Hit else case, but {} was not terminal".format(final_state)

		# Refine the initiation set classifier
		if len(self.negative_examples) > 0:
			if self.pixel_initiations:
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
