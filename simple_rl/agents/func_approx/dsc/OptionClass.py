# Python imports.
from __future__ import print_function
import random

import ipdb
import numpy as np
import pdb
from copy import deepcopy
import torch
from sklearn import svm
from tqdm import tqdm
import itertools
from scipy.spatial import distance

# Other imports.
from simple_rl.mdp.StateClass import State
from simple_rl.agents.func_approx.ddpg.DDPGAgentClass import DDPGAgent
from simple_rl.agents.func_approx.dsc.utils import Experience
from simple_rl.agents.func_approx.dsc.BaseSalientEventClass import BaseSalientEvent


class Option(object):

	def __init__(self, overall_mdp, name, global_solver, lr_actor, lr_critic, ddpg_batch_size, classifier_type="ocsvm",
				 subgoal_reward=0., max_steps=20000, seed=0, parent=None, num_subgoal_hits_required=3, buffer_length=20,
				 dense_reward=False, enable_timeout=True, timeout=100, initiation_period=5, option_idx=None,
				 chain_id=None, initialize_everywhere=True, intersecting_options=[], max_num_children=3,
				 init_salient_event=None, target_salient_event=None,
				 gestation_init_predicates=[], is_backward_option=False, generate_plots=False, device=torch.device("cpu"), writer=None):
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
			chain_id (int)
			initialize_everywhere (bool)
			init_salient_event (BaseSalientEvent)
			target_salient_event (BaseSalientEvent)
			gestation_init_predicates (list)
			is_backward_option (bool)
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
		self.enable_timeout = enable_timeout
		self.initiation_period = initiation_period
		self.max_num_children = max_num_children
		self.classifier_type = classifier_type
		self.generate_plots = generate_plots
		self.writer = writer
		self.initialize_everywhere = initialize_everywhere
		self.init_salient_event = init_salient_event
		self.target_salient_event = target_salient_event
		self.gestation_init_predicates = gestation_init_predicates
		self.overall_mdp = overall_mdp

		self.timeout = np.inf

		# Global option operates on a time-scale of 1 while child (learned) options are temporally extended
		if enable_timeout:
			self.timeout = 1 if name == "global_option" else timeout

		self.option_idx = option_idx
		self.chain_id = chain_id
		self.backward_option = is_backward_option
		self.intersecting_options = intersecting_options

		print("Creating {} in chain {} with enable_timeout={}".format(name, chain_id, enable_timeout))

		random.seed(seed)
		np.random.seed(seed)

		state_size = overall_mdp.state_space_size()
		action_size = overall_mdp.action_space_size()

		solver_name = "{}_ddpg_agent".format(self.name)
		self.global_solver = DDPGAgent(state_size, action_size, seed, device, lr_actor, lr_critic, ddpg_batch_size, name=solver_name) if name == "global_option" else global_solver
		self.solver = None
		self.reset_ddpg_solver()

		# Attributes related to initiation set classifiers
		self.num_goal_hits = 0
		self.positive_examples = []
		self.negative_examples = []
		self.experience_buffer = []
		self.initiation_classifier = None
		self.num_subgoal_hits_required = num_subgoal_hits_required
		self.buffer_length = buffer_length
		self.last_episode_term_triggered = -1
		self.nu = 0.01

		self.final_transitions = []
		self.terminal_states = []
		self.children = []

		# Debug member variables
		self.num_executions = 0
		self.taken_or_not = []
		self.n_taken_or_not = 0
		self.option_start_states = []
		self.num_test_executions = 0
		self.num_successful_test_executions = 0

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

	def reset_ddpg_solver(self):
		state_size = self.overall_mdp.state_space_size()
		action_size = self.overall_mdp.action_space_size()
		seed = self.seed
		device = self.global_solver.device
		lr_actor = self.global_solver.actor_learning_rate
		lr_critic = self.global_solver.critic_learning_rate
		ddpg_batch_size = self.global_solver.batch_size
		writer = self.writer
		solver_name = "{}_ddpg_agent".format(self.name)
		exploration = "shaping" if self.name == "global_option" else ""
		self.solver = DDPGAgent(state_size, action_size, seed, device, lr_actor, lr_critic, ddpg_batch_size,
								tensor_log=(writer is not None), writer=writer, name=solver_name, exploration=exploration)

	def sample_state(self):
		""" Return a state from the option's initiation set. """
		if self.get_training_phase() != "initiation_done":
			return None

		sampled_state = None
		while sampled_state is None:
			sampled_experience = random.choice(self.solver.replay_buffer.memory)
			sampled_state = sampled_experience[0] if self.is_init_true(sampled_experience[0]) else None
		return sampled_state

	def get_sibling_options(self):
		siblings = []
		if self.parent is not None:
			parent_children = self.parent.children
			for child in parent_children:
				if child != self:
					siblings.append(child)
		return siblings

	def is_valid_init_data(self, state_buffer):
		siblings = self.get_sibling_options()  # type: list

		if len(siblings) == 0:
			return True

		if len(state_buffer) == 0:
			return False

		def _get_sibling_count(sibling, buffer):
			if sibling.initiation_classifier is not None:
				sibling_count = 0.
				for state in buffer:
					sibling_count += sibling.is_init_true(state)
				return sibling_count
			return 0

		sibling_counts = [_get_sibling_count(sibling, state_buffer) for sibling in siblings]

		sibling_condition = 0 <= (max(sibling_counts) / len(state_buffer)) <= 0.2
		#
		# # # TODO: Hack - preventing chain id 3 from chaining to the start state
		# # def _start_hallway_count(states):
		# # 	count = 0
		# # 	for s in states:
		# # 		if s.position[0] < 8:
		# # 			count += 1
		# # 	return count
		# #
		# # if self.chain_id == 3:
		# # 	backward_fraction = _start_hallway_count(state_buffer) / len(state_buffer)
		# # 	backward_condition = 0 < backward_fraction <= 0.1
		# # 	return backward_condition and sibling_condition
		#
		return sibling_condition

	def get_training_phase(self):
		if self.num_goal_hits < self.num_subgoal_hits_required:
			return "gestation"
		if self.num_goal_hits < (self.num_subgoal_hits_required + self.initiation_period):
			return "initiation"
		if self.num_goal_hits >= (self.num_subgoal_hits_required + self.initiation_period):
			return "initiation_done"
		return "trained"

	def initialize_with_global_ddpg(self, reset_option_buffer=True):
		"""" Initialize option policy - unrestricted support because not checking if I_o(s) is true. """
		# Not using off_policy_update() because we have numpy arrays not state objects here
		for state, action, reward, next_state, done in tqdm(self.global_solver.replay_buffer.memory, desc=f"Initializing {self.name} policy"):
			if self.is_term_true(state):
				continue
			if self.is_term_true(next_state):
				self.solver.step(state, action, self.subgoal_reward, next_state, True)
			else:
				subgoal_reward = self.get_subgoal_reward(next_state)
				self.solver.step(state, action, subgoal_reward, next_state, done)

		# To keep the support of the option policy restricted to the initiation region
		# of the option, we can clear the option's replay buffer after we have pre-trained its policy
		if reset_option_buffer:
			self.solver.replay_buffer.clear()

	def initialize_option_ddpg_with_restricted_support(self):  # TODO: THIS DOESN"T WORK

		assert self.initiation_classifier is not None, f"{self.name} in phase {self.get_training_phase()}"

		for state, action, reward, next_state, done in tqdm(self.global_solver.replay_buffer.memory,
															desc=f"Initializing {self.name} policy with {self.global_solver.name}"):

			# Adding terminal self transitions causes the option value function to explode
			if self.is_term_true(state):
				continue

			# Give sub-goal reward for triggering your own subgoal
			if self.is_term_true(next_state):
				self.solver.step(state, action, self.subgoal_reward, next_state, True)

			# Restricted support - only add transitions that begin from the initiation set of the option
			elif self.is_init_true(state):
				subgoal_reward = self.get_subgoal_reward(next_state)
				self.solver.step(state, action, subgoal_reward, next_state, done)

	def batched_is_init_true(self, state_matrix):

		if self.name == "global_option":
			return np.ones((state_matrix.shape[0]))

		if self.initialize_everywhere and (self.initiation_classifier is None or self.get_training_phase() == "gestation"):
			if self.backward_option:
				# TODO: Kshitij deleted
				# state_matrix = state_matrix[:, :2]

				# When we treat the start state as a salient event, we pass in the
				# init predicate that we are going to use during gestation
				if self.init_salient_event is not None:
					return self.init_salient_event(state_matrix)

				# When we create intersection salient events, then we treat the union of all salient
				# events as the joint initiation set of the backward options
				salients = [event(state_matrix) for event in self.overall_mdp.get_current_salient_events()]

				if len(salients) > 1:
					gestation_inits = np.logical_or.reduce(tuple(salients))
					assert gestation_inits.shape == (state_matrix.shape[0],), gestation_inits.shape
					return gestation_inits
				elif len(salients) == 1:
					return salients[0]
				elif len(salients) == 0:
					return np.zeros((state_matrix.shape[0]))

			return np.ones((state_matrix.shape[0]))

		# TODO: Kshitij deleted
		# position_matrix = state_matrix[:, :2]
		position_matrix = state_matrix[:, :]
		return self.initiation_classifier.predict(position_matrix) == 1

	def batched_is_term_true(self, state_matrix):
		if self.parent is not None:
			return self.parent.batched_is_init_true(state_matrix)

		# Extract the relevant dimensions from the state matrix (x, y)
		# TODO: Kshitij deleted
		# state_matrix = state_matrix[:, :2]

		if self.target_salient_event is not None:
			return self.target_salient_event(state_matrix)

		# If the option does not have a parent, it must be targeting a pre-specified salient event
		if self.name == "global_option" or self.name == "exploration_option":
			return np.zeros((state_matrix.shape[0]))
		elif self.name == "goal_option_1":
			return self.overall_mdp.get_original_salient_events()[0](state_matrix)
		elif self.name == "goal_option_2":
			return self.overall_mdp.get_original_salient_events()[1](state_matrix)
		else:
			assert len(self.intersecting_options) > 0
			o1, o2 = self.intersecting_options[0], self.intersecting_options[1]
			return np.logical_and(o1.batched_is_init_true(state_matrix), o2.batched_is_init_true(state_matrix))

	def is_init_true(self, ground_state):

		if self.name == "global_option":
			return True

		if self.initialize_everywhere and (self.initiation_classifier is None or self.get_training_phase() == "gestation"):
			# For options in the backward chain, the default initiation set is the
			# union of the discovered salient events in the MDP
			if self.backward_option:

				# Backward chain is just the reverse of the start and target regions
				# During gestation, we use the old target salient events as initiation sets for
				# options in the backward chain
				if self.init_salient_event is not None:
					return self.init_salient_event(ground_state)

				# Intersection salient event
				salients = [event(ground_state) for event in self.overall_mdp.get_current_salient_events()]
				return any(salients)

			return True

		# print('using svm for predicting if in initiation set.')
		features = ground_state.features() if isinstance(ground_state, State) else ground_state[:5]
		return self.initiation_classifier.predict([features])[0] == 1

	def is_term_true(self, ground_state):
		if self.parent is not None:
			return self.parent.is_init_true(ground_state)

		# ip the option is targeting a salient event, check if we have satisfied the
		# initiation condition of that salient event
		if self.target_salient_event is not None:
			return self.target_salient_event(ground_state)

		# If option does not have a parent, it must be the goal option or the global option
		if self.name == "global_option" or self.name == "exploration_option":
			return self.overall_mdp.is_goal_state(ground_state)
		else:
			assert len(self.intersecting_options) > 0, "{}, {}".format(self, self.intersecting_options)
			o1, o2 = self.intersecting_options[0], self.intersecting_options[1]
			return o1.is_init_true(ground_state) and o2.is_init_true(ground_state)

	def should_target_with_bonus(self):
		"""
		Used by policy over options and the global DDPG agent to decide if the
		skill chaining agent should target the initiation set of the current
		option by means of exploration bonuses.
		"""
		if self.name == "global_option":
			return False

		# During the gestation period, the initiation set of options in the backward chain
		# are set to be the union of the salient events in the MDP. As a result, we want to
		# encourage the policy over options to execute skills that trigger those salient
		# events. From there, we hope to execute the new backward option so that it can
		# trigger the intersection salient event. However, if the backward option has progressed
		# to the initiation_done phase, we no longer need to give exploration bonuses to target it
		if self.backward_option and self.get_training_phase() != "initiation_done":
			return True

		# If the current option is not in the backward chain, then we give exploration bonus
		# only when it is in the initiation phase. When it is in the gestation phase,
		# it can initiate anywhere (consequence of initialize_everywhere). When it is in
		# initiation phase, this term will yield exploration bonuses
		return self.get_training_phase() == "initiation"

	def get_init_predicates_during_gestation(self):
		assert self.backward_option
		assert len(self.gestation_init_predicates) > 0, self.gestation_init_predicates

		my_original_init_predicates = self.gestation_init_predicates
		overall_current_init_predicates = self.overall_mdp.get_current_salient_events()
		my_current_init_predicates = [pred for pred in my_original_init_predicates if pred in overall_current_init_predicates]
		return my_current_init_predicates

	def get_batched_init_predicates_during_gestation(self):
		assert self.backward_option
		assert len(self.gestation_init_predicates) > 0, self.gestation_init_predicates
		pass

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
		# TODO: Why is this check failing???
		# assert len(self.positive_examples) == self.num_subgoal_hits_required, "Expected init data to be a list of lists"
		positive_feature_matrix = self.construct_feature_matrix(self.positive_examples)

		# Smaller gamma -> influence of example reaches farther. Using scale leads to smaller gamma than auto.
		self.initiation_classifier = svm.OneClassSVM(kernel="rbf", nu=self.nu, gamma="scale")
		self.initiation_classifier.fit(positive_feature_matrix)

		return True

	def train_two_class_classifier(self):
		positive_feature_matrix = self.construct_feature_matrix(self.positive_examples)
		negative_feature_matrix = self.construct_feature_matrix(self.negative_examples)
		positive_labels = [1] * positive_feature_matrix.shape[0]
		negative_labels = [0] * negative_feature_matrix.shape[0]

		X = np.concatenate((positive_feature_matrix, negative_feature_matrix))
		Y = np.concatenate((positive_labels, negative_labels))

		if len(self.negative_examples) >= 5:
			kwargs = {"kernel": "rbf", "gamma": "scale", "class_weight": "balanced"}
		else:
			kwargs = {"kernel": "rbf", "gamma": "scale"}

		# We use a 2-class balanced SVM which sets class weights based on their ratios in the training data
		initiation_classifier = svm.SVC(**kwargs)
		initiation_classifier.fit(X, Y)

		training_predictions = initiation_classifier.predict(X)
		positive_training_examples = X[training_predictions == 1]

		if positive_training_examples.shape[0] > 0:
			self.initiation_classifier = svm.OneClassSVM(kernel="rbf", nu=self.nu, gamma="scale")
			self.initiation_classifier.fit(positive_training_examples)

			self.classifier_type = "tcsvm"
			return True
		else:
			print("\r Couldn't fit first SVM for {}".format(self.name))
			kwargs = {"kernel": "rbf", "gamma": "scale"}

			# We use a 2-class balanced SVM which sets class weights based on their ratios in the training data
			initiation_classifier = svm.SVC(**kwargs)
			initiation_classifier.fit(X, Y)

			training_predictions = initiation_classifier.predict(X)
			positive_training_examples = X[training_predictions == 1]

			if positive_training_examples.shape[0] > 0:
				self.initiation_classifier = svm.OneClassSVM(kernel="rbf", nu=self.nu, gamma="scale")
				self.initiation_classifier.fit(positive_training_examples)

				self.classifier_type = "tcsvm"
				return True
			else:
				print("Couldn't fit any classifier for ", self.name)
				return False

	def train_initiation_classifier(self):
		if self.classifier_type == "ocsvm" and len(self.negative_examples) == 0:
			trained = self.train_one_class_svm()
		else:
			trained = self.train_two_class_classifier()
		return trained

	def initialize_option_policy(self):
		# Initialize the local DDPG solver with the weights of the global option's DDPG solver
		if self.chain_id < 3:
			self.initialize_with_global_ddpg()
			self.solver.epsilon = self.global_solver.epsilon

		# Fitted Q-iteration on the experiences that led to triggering the current option's termination condition
		experience_buffer = list(itertools.chain.from_iterable(self.experience_buffer))
		for experience in experience_buffer:
			state, action, reward, next_state = experience.serialize()
			self.update_option_solver(state, action, reward, next_state)

	def train(self, experience_buffer, state_buffer, episode):
		"""
		Called every time the agent hits the current option's termination set.
		Args:
			experience_buffer (list)
			state_buffer (list)
			episode (int)
		Returns:
			trained (bool): whether or not we actually trained this option
		"""
		self.add_initiation_experience(state_buffer)
		self.add_experience_buffer(experience_buffer)
		self.num_goal_hits += 1

		self.last_episode_term_triggered = episode

		if self.num_goal_hits >= self.num_subgoal_hits_required:
			trained = self.train_initiation_classifier()
			if trained:
				self.initialize_option_policy()
			return trained
		return False

	def get_subgoal_reward(self, state):

		if self.is_term_true(state):
			print("~~~~~ Warning: subgoal query at goal ~~~~~")
			return self.subgoal_reward

		# Return step penalty in sparse reward domain
		if not self.dense_reward:
			return -1.

		# Rewards based on position only
		position_vector = state.features() if isinstance(state, State) else state[:5]

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

	def execute_option_in_mdp(self, mdp, episode, step_number, eval_mode=False):
		"""
		Option main control loop.

		Args:
			mdp (MDP): environment where actions are being taken
			episode (int)
			step_number (int): how many steps have already elapsed in the outer control loop.
			eval_mode (bool): Added for the SkillGraphPlanning agent so that we can perform
							  option policy rollouts with eval_epsilon but still keep training
							  option policies based on new experiences seen during test time.

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

			if self.name != "global_option":
				print("Executing {}".format(self.name))

			self.option_start_states.append(start_state)
			while not self.is_term_true(state) and not state.is_terminal() and \
					step_number < self.max_steps and num_steps < self.timeout:

				action = self.solver.act(state.features(), evaluation_mode=eval_mode)

				reward, next_state = mdp.execute_agent_action(action, option_idx=self.option_idx)
				# if step_number == 0 or step_number == 1 or step_number == 2 or step_number == 3 or step_number == 4:
				# 	mdp.temp.append((episode, step_number, self.name, state.position, next_state.position))
				# 	if len(mdp.temp) > 50:
				# 		ipdb.set_trace()

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

			# TODO: When initialize_everywhere is true, self.train() needs to be called from the OptionClass and not from SkillChainingClass

			if self.is_term_true(state) and self.last_episode_term_triggered != episode:  # and self.is_valid_init_data(visited_states):
				self.num_goal_hits += 1
				self.last_episode_term_triggered = episode

			if self.name != "global_option" and self.get_training_phase() != "initiation_done":
				self.refine_initiation_set_classifier(visited_states, start_state, state, num_steps, step_number)

			if self.writer is not None:
				self.writer.add_scalar("{}_ExecutionLength".format(self.name), len(option_transitions), self.num_executions)

			return option_transitions, total_reward

		raise Warning("Wanted to execute {}, but initiation condition not met".format(self))

	def refine_initiation_set_classifier(self, visited_states, start_state, final_state, num_steps, outer_step_number):
		if self.is_term_true(final_state):  # success

			# No need to add the start state of option execution to positive examples during
			# gestation because there are no negative examples yet to balance
			if self.get_training_phase() == "gestation":
				positive_states = visited_states[-self.buffer_length:]
			else:
				positive_states = [start_state] + visited_states[-self.buffer_length:]

			# if self.is_valid_init_data(positive_states):
			positive_examples = [state.features() for state in positive_states]
			self.positive_examples.append(positive_examples)

		elif num_steps == self.timeout and self.get_training_phase() == "initiation":
			negative_examples = [start_state.position]
			self.negative_examples.append(negative_examples)
		# else:
		# 	assert final_state.is_terminal() or outer_step_number == self.max_steps, \
		# 		"Hit else case, but {} was not terminal".format(final_state)

		# Refine the initiation set classifier
		if len(self.negative_examples) > 0 and len(self.positive_examples) > 0:
			self.train_two_class_classifier()
		elif len(self.positive_examples) > 0:
			self.train_one_class_svm()

	def trained_option_execution(self, mdp, outer_step_counter):
		state = mdp.cur_state
		score, step_number = 0., deepcopy(outer_step_counter)
		num_steps = 0
		state_option_trajectory = []
		self.num_test_executions += 1

		while not self.is_term_true(state) and not state.is_terminal()\
				and step_number < self.max_steps and num_steps < self.timeout:
			state_option_trajectory.append((self.option_idx, deepcopy(state)))
			action = self.solver.act(state.features(), evaluation_mode=True)
			reward, state = mdp.execute_agent_action(action, option_idx=self.option_idx)

			score += reward
			step_number += 1
			num_steps += 1

		if self.is_term_true(state):
			self.num_successful_test_executions += 1

		return score, state, step_number, state_option_trajectory

	def get_option_success_rate(self):
		if self.num_test_executions > 0:
			return self.num_successful_test_executions / self.num_test_executions
		return 1.
