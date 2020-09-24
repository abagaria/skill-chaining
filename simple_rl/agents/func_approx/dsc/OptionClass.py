# Python imports.
from __future__ import print_function
import random
import numpy as np
import ipdb
from copy import deepcopy
import torch
from sklearn import svm
from tqdm import tqdm
import itertools
from scipy.spatial import distance

# Other imports.
from simple_rl.mdp.StateClass import State
from simple_rl.mdp.GoalDirectedMDPClass import GoalDirectedMDP
from simple_rl.agents.func_approx.ddpg.DDPGAgentClass import DDPGAgent
from simple_rl.agents.func_approx.td3.TD3AgentClass import TD3
from simple_rl.agents.func_approx.dsc.utils import Experience
from simple_rl.agents.func_approx.dsc.SalientEventClass import SalientEvent


class Option(object):

	def __init__(self, overall_mdp, name, global_solver, lr_actor, lr_critic, ddpg_batch_size, classifier_type="ocsvm",
				 subgoal_reward=0., max_steps=20000, seed=0, parent=None, num_subgoal_hits_required=3, buffer_length=20,
				 dense_reward=False, enable_timeout=True, timeout=200, initiation_period=5, option_idx=None,
				 chain_id=None, initialize_everywhere=True, max_num_children=3,
				 init_salient_event=None, target_salient_event=None, update_global_solver=False, use_warmup_phase=True,
				 gestation_init_predicates=[], is_backward_option=False, solver_type="ddpg", use_her=True,
				 generate_plots=False, device=torch.device("cpu"), writer=None):
		'''
		Args:
			overall_mdp (GoalDirectedMDP)
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
			init_salient_event (SalientEvent)
			target_salient_event (SalientEvent)
			update_global_solver (bool)
			use_warmup_phase (bool)
			gestation_init_predicates (list)
			is_backward_option (bool)
			use_her (bool)
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
		self.update_global_solver = update_global_solver
		self.use_warmup_phase = use_warmup_phase
		self.gestation_init_predicates = gestation_init_predicates
		self.overall_mdp = overall_mdp
		self.device = device
		self.lr_actor = lr_actor
		self.lr_critic = lr_critic
		self.ddpg_batch_size = ddpg_batch_size
		self.solver_type = solver_type
		self.use_her = use_her

		if self.name not in ("global_option", "exploration_option"):
			assert self.init_salient_event is not None
			assert self.target_salient_event is not None

		self.state_size = self.overall_mdp.state_space_size() + 2 if use_her else self.overall_mdp.state_space_size()
		self.action_size = self.overall_mdp.action_space_size()

		self.timeout = np.inf

		# Global option operates on a time-scale of 1 while child (learned) options are temporally extended
		if enable_timeout:
			self.timeout = 1 if name == "global_option" else timeout

		self.option_idx = option_idx
		self.chain_id = chain_id
		self.backward_option = is_backward_option

		print("Creating {} in chain {} with enable_timeout={}".format(name, chain_id, enable_timeout))

		random.seed(seed)
		np.random.seed(seed)

		if name != "global_option": assert global_solver is not None

		self.global_solver = global_solver
		self.solver = None

		if self.name == "global_option":
			self.reset_global_solver()
		else:
			self.reset_option_solver()

		# Attributes related to initiation set classifiers
		self.num_goal_hits = 0
		self.effect_set = []
		self.positive_examples = []
		self.negative_examples = []
		self.experience_buffer = []
		self.initiation_classifier = None
		self.num_subgoal_hits_required = num_subgoal_hits_required
		self.buffer_length = buffer_length
		self.last_episode_term_triggered = -1
		self.nu = 0.01
		self.pretrained_option_policy = False
		self.started_at_target_salient_event = False
		self.salient_event_for_initiation = None

		self.final_transitions = []
		self.terminal_states = []
		self.children = []

		# Debug member variables
		self.num_executions = 0
		self.num_test_executions = 0
		self.num_successful_test_executions = 0
		self.pursued_goals = []

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

	def __getstate__(self):
		excluded_keys = ("parent", "global_solver", "children", "overall_mdp")
		parent_option_idx = self.parent.option_idx if self.parent is not None else None
		children_option_idx = [child.option_idx for child in self.children if child is not None]
		state_dictionary = {x: self.__dict__[x] for x in self.__dict__ if x not in excluded_keys}
		state_dictionary["parent_option_idx"] = parent_option_idx
		state_dictionary["children_option_idx"] = children_option_idx
		return state_dictionary

	def __setstate__(self, state_dictionary):
		excluded_keys = ("parent", "global_solver", "children", "overall_mdp")
		for key in state_dictionary:
			if key not in excluded_keys:
				self.__dict__[key] = state_dictionary[key]

	def _get_epsilon_greedy_epsilon(self):
		if "point" in self.overall_mdp.env_name:
			return 0.1
		elif "ant" in self.overall_mdp.env_name:
			return 0.25

	def act(self, state_features, eval_mode, warmup_phase):
		""" Epsilon greedy action selection when in training mode. """

		if warmup_phase and self.use_warmup_phase:
			return self.overall_mdp.sample_random_action()

		if random.random() < self._get_epsilon_greedy_epsilon() and not eval_mode:
			return self.overall_mdp.sample_random_action()

		return self.solver.act(state_features, evaluation_mode=eval_mode)

	def reset_global_solver(self):
		if self.solver_type == "ddpg":
			self.solver = DDPGAgent(self.state_size, self.action_size, self.seed, self.device,
										   self.lr_actor, self.lr_critic, self.ddpg_batch_size, name="global_option")
		elif self.solver_type == "td3":
			self.solver = TD3(state_dim=self.state_size, action_dim=self.action_size,
									 max_action=self.overall_mdp.env.action_space.high[0], device=self.device)

		else:
			raise NotImplementedError(self.solver_type)

	def reset_option_solver(self):
		if self.solver_type == "ddpg":
			seed = self.seed
			device = self.global_solver.device
			lr_actor = self.global_solver.actor_learning_rate
			lr_critic = self.global_solver.critic_learning_rate
			ddpg_batch_size = self.global_solver.batch_size
			writer = self.writer
			solver_name = "{}_ddpg_agent".format(self.name)
			self.solver = DDPGAgent(self.state_size, self.action_size, seed, device, lr_actor, lr_critic, ddpg_batch_size,
									tensor_log=(writer is not None), writer=writer, name=solver_name)
		elif self.solver_type == "td3":
			exploration_method = "shaping" if self.name == "global_option" else ""
			self.solver = TD3(state_dim=self.state_size, action_dim=self.action_size,
							  max_action=self.overall_mdp.env.action_space.high[0],
							  device=self.device, exploration_method=exploration_method)

		else:
			raise NotImplementedError(self.solver_type)

	@staticmethod
	def is_between(x1, x2, x3):
		""" is x3 in between x1 and x2? """
		d0 = x2 - x1
		d0m = np.linalg.norm(d0)
		v = d0 / d0m
		d1 = x3 - x1
		r = v.dot(d1)
		return 0 <= r <= d0m

	def sample_state(self):
		""" Return a state from the option's initiation set. """

		def _get_state_from_experience(experience):
			if isinstance(experience, list):
				experience = experience[0]
			if isinstance(experience, Experience):
				return experience.state
			return experience[0]

		if self.get_training_phase() != "initiation_done":
			return None

		sampled_state = None
		num_tries = 0

		while sampled_state is None and num_tries < 50:
			if isinstance(self.solver, DDPGAgent):
				if len(self.solver.replay_buffer) > 0:
					sampled_experience = random.choice(self.solver.replay_buffer.memory)
				elif len(self.experience_buffer) > 0:
					sampled_experience = random.choice(self.experience_buffer)
				else:
					continue
				# ipdb.set_trace()
				sampled_state = _get_state_from_experience(sampled_experience)
				sampled_state = sampled_state if self.is_init_true(sampled_state) else None
			elif isinstance(self.solver, TD3):
				sampled_idx = random.randint(0, len(self.solver.replay_buffer) - 1)
				sampled_experience = self.solver.replay_buffer[sampled_idx]
				sampled_state = sampled_experience[0] if self.is_init_true(sampled_experience[0]) else None
			num_tries += 1
		return sampled_state

	def sample_parent_positive(self):
		num_tries = 0
		sampled_positive = None
		parent_positives = list(itertools.chain.from_iterable(self.parent.positive_examples))
		while sampled_positive is None and num_tries < 50 and len(parent_positives) > 0:
			num_tries += 1
			sampled_positive = random.sample(parent_positives, k=1)[0]
		return sampled_positive

	def get_sibling_options(self):
		siblings = []
		if self.parent is not None:
			parent_children = self.parent.children
			for child in parent_children:
				if child != self:
					siblings.append(child)
		return siblings

	def is_valid_init_data(self, state_buffer):
		# TODO: Use should_expand_initiation_classifier() from ChainClass

		def get_points_in_between(init_point, final_point, positions):
			return [position for position in positions if self.is_between(init_point, final_point, position)]

		# Use the data if it could complete the chain
		if self.init_salient_event is not None:
			if any([self.is_init_true(s) for s in self.init_salient_event.trigger_points]):
				return True

		length_condition = len(state_buffer) >= self.buffer_length // 10

		if not length_condition:
			return False

		position_buffer = [self.overall_mdp.get_position(state) for state in state_buffer]
		points_in_between = get_points_in_between(self.init_salient_event.get_target_position(),
												  self.target_salient_event.get_target_position(),
												  position_buffer)
		in_between_condition = len(points_in_between) >= self.buffer_length // 10
		return in_between_condition

	def get_training_phase(self):
		if self.num_goal_hits < self.num_subgoal_hits_required:
			return "gestation"
		if self.num_goal_hits < (self.num_subgoal_hits_required + self.initiation_period):
			return "initiation"
		if self.num_goal_hits >= (self.num_subgoal_hits_required + self.initiation_period):
			return "initiation_done"
		return "trained"

	def initialize_with_global_solver(self, reset_option_buffer=True):
		"""" Initialize option policy - unrestricted support because not checking if I_o(s) is true. """
		# Not using off_policy_update() because we have numpy arrays not state objects here
		if not self.pretrained_option_policy:  # Make sure that we only do this once
			for state, action, reward, next_state, done in tqdm(self.global_solver.replay_buffer,
																desc=f"Initializing {self.name} policy"):
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

			self.pretrained_option_policy = True

	def initialize_option_with_global_her_policy(self):
		""" Work in progress: initialize option policy with the global option's goal-conditioned policy. """

		for target_param, param in zip(self.global_solver.target_actor.parameters(), self.solver.actor.parameters()):
			target_param.data.copy_(param.data)

		for target_param, param in zip(self.global_solver.target_critic.parameters(), self.solver.critic.parameters()):
			target_param.data.copy_(param.data)

		self.use_her = True

	def initialize_option_solver_with_restricted_support(self):  # TODO: THIS DOESN"T WORK

		assert self.initiation_classifier is not None, f"{self.name} in phase {self.get_training_phase()}"

		for state, action, reward, next_state, done in tqdm(self.global_solver.replay_buffer,
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
				state_matrix = state_matrix[:, :2]

				# When we treat the start state as a salient event, we pass in the
				# init predicate that we are going to use during gestation
				if self.init_salient_event is not None:
					return self.init_salient_event(state_matrix)

			return np.ones((state_matrix.shape[0]))
		position_matrix = state_matrix[:, :2]

		event_decision = self.salient_event_for_initiation(position_matrix) \
							if self.salient_event_for_initiation is not None else np.zeros((state_matrix.shape[0]))

		return np.logical_or(self.initiation_classifier.predict(position_matrix) == 1, event_decision)

	def batched_is_term_true(self, state_matrix):
		if self.parent is not None:
			return self.parent.batched_is_init_true(state_matrix)

		# Extract the relevant dimensions from the state matrix (x, y)
		state_matrix = state_matrix[:, :2]

		if self.name == "global_option" or self.name == "exploration_option":
			return np.zeros((state_matrix.shape[0]))  # TODO: Create and use a batched version of MDP::is-goal-state()

		assert self.target_salient_event is not None
		return self.target_salient_event(state_matrix)

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

			return True

		features = ground_state.features()[:2] if isinstance(ground_state, State) else ground_state[:2]

		# If an option's initiation set is "expanded", we union its classifier with the init_salient_event
		event_decision = self.salient_event_for_initiation(features) if self.salient_event_for_initiation is not None else False

		return self.initiation_classifier.predict([features])[0] == 1 or event_decision

	def is_term_true(self, ground_state):
		if self.parent is not None:
			return self.parent.is_init_true(ground_state)

		if self.name == "global_option" or self.name == "exploration_option":
			return self.overall_mdp.is_goal_state(ground_state)

		assert self.target_salient_event is not None, self.target_salient_event
		return self.target_salient_event(ground_state)

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

	def expand_initiation_classifier(self, salient_event):
		self.salient_event_for_initiation = salient_event

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
	def distance_to_weights(distances, tol=0.6):
		weights = np.copy(distances)
		for row in range(weights.shape[0]):
			if weights[row] > tol:
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

		weights = None

		if self.parent is None and self.target_salient_event.get_target_position() is not None:
			goal = self.target_salient_event.get_target_position()[None, ...]
			distances = distance.cdist(positive_feature_matrix, goal)
			weights = self.distance_to_weights(distances, tol=self.target_salient_event.tolerance).squeeze(1)

		self.initiation_classifier.fit(positive_feature_matrix, sample_weight=weights)

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
		# self.initialize_with_global_solver()
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
			if trained and not self.use_her:
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
		position_vector = state.features()[:2] if isinstance(state, State) else state[:2]

		# For global and parent option, we use the negative distance to the goal state
		if self.parent is None:
			goal_state = self.target_salient_event.target_state
			goal_position = goal_state.features()[:2] if isinstance(goal_state, State) else goal_state[:2]
			distance_to_goal = np.linalg.norm(position_vector - goal_position)
			return -distance_to_goal

		# For every other option, we use the negative distance to the parent's initiation set classifier
		dist = self.parent.initiation_classifier.decision_function(position_vector.reshape(1, -1))[0]

		# Decision_function returns a negative distance for points not inside the classifier
		subgoal_reward = 0. if dist >= 0 else dist
		return subgoal_reward

	def off_policy_update(self, state, action, reward, next_state):
		""" Make off-policy updates to the current option's low level DDPG solver. """
		assert not state.is_terminal(), "Terminal state did not terminate at some point"

		# Don't make updates while walking around the termination set of an option
		if self.is_term_true(state):
			print("[off_policy_update] Warning: called updater on {} term states: {}".format(self.name, state))
			return

		# Off-policy updates for states outside tne initiation set were discarded
		if self.is_init_true(state) and self.is_term_true(next_state):
			self.solver.step(state.features(), action, self.subgoal_reward, next_state.features(), True)
		elif self.is_init_true(state):
			subgoal_reward = self.get_subgoal_reward(next_state) if self.name != "global_option" else reward
			self.solver.step(state.features(), action, subgoal_reward, next_state.features(), next_state.is_terminal())

	def update_option_solver(self, s, a, r, s_prime):
		""" Make on-policy updates to the current option's low-level DDPG solver. """
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
			subgoal_reward = self.get_subgoal_reward(s_prime) if self.name != "global_option" else r
			self.solver.step(s.features(), a, subgoal_reward, s_prime.features(), False)

	def relabel_transitions_with_option_reward_function(self, transitions):
		assert not self.use_her, "Not implemented for HER transitions yet b/c that requires picking a goal-state"
		relabeled_transitions = []
		for state, action, _, next_state in transitions:
			done = self.is_term_true(next_state) or next_state.is_terminal()
			reward = self.subgoal_reward if done else self.get_subgoal_reward(next_state)
			relabeled_transitions.append((state, action, reward, next_state, done))
		return relabeled_transitions

	def get_mean_td_error(self, transitions):
		relabeled_transitions = self.relabel_transitions_with_option_reward_function(transitions)
		batched_transitions = DDPGAgent.batchify_transitions(relabeled_transitions)
		td_errors = self.solver.get_td_error(*batched_transitions)
		return np.mean(np.abs(td_errors))

	def get_goal_for_option_rollout(self, method="use_effect_set"):
		assert method in ("use_effect_set", "use_term_set"), method

		def _sample_from_parent_initiation_set():
			if self.parent is not None:
				parent_positive = self.sample_parent_positive()
				if parent_positive is not None:
					g = self.overall_mdp.get_position(parent_positive)
					return g

		if self.parent is None:  # Pick a specific state for root options
			goal = self.target_salient_event.get_target_position()
			if goal is not None: return goal

		if self.parent is not None and method == "use_term_set":
			goal = _sample_from_parent_initiation_set()
			if goal is not None: return goal

		if len(self.effect_set) > 0 and method == "use_effect_set":
			sample = random.sample(self.effect_set, k=1)[0]
			return self.overall_mdp.get_position(sample)

		# If the effect set is empty, then sample from the parent option's initiation set classifier
		if self.parent is not None:
			goal = _sample_from_parent_initiation_set()
			if goal is not None: return goal

		# If not a root option, and can't sample from either the term or effect sets,
		# then just use the chain's target salient event as the goal
		if self.target_salient_event is not None:
			return self.target_salient_event.get_target_position()

		return self.overall_mdp.get_position(self.overall_mdp.goal_state)

	def get_augmented_state(self, state, goal):
		if goal is not None and self.use_her:
			return np.concatenate((state.features(), goal))
		return state.features()

	def get_goal_for_current_rollout(self, goal_for_policy_over_options):
		if goal_for_policy_over_options is not None and self.name == "global_option":
			return goal_for_policy_over_options
		return self.get_goal_for_option_rollout(method="use_effect_set")

	def execute_option_in_mdp(self, mdp, episode, step_number, poo_goal=None, eval_mode=False):
		"""
		Option main control loop.

		Args:
			mdp (MDP): environment where actions are being taken
			episode (int)
			step_number (int): how many steps have already elapsed in the outer control loop.
			poo_goal (np.ndarray): Current goal that the policy over options (poo) is pursuing right now
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
			goal = None
			option_transitions = []
			total_reward = 0.
			self.num_executions += 1
			num_steps = 0
			visited_states = []

			# Pick goal for a goal-conditioned option rollout
			if self.use_her:
				goal = self.get_goal_for_current_rollout(goal_for_policy_over_options=poo_goal)
				self.pursued_goals.append(goal)

			if self.name != "global_option":
				print(f"Executing {self.name} targeting {goal}")

			warmup_phase = self.get_training_phase() == "gestation"

			while not self.is_at_local_goal(state, goal) and step_number < self.max_steps and num_steps < self.timeout:

				# Goal-conditioned option acting
				augmented_state = self.get_augmented_state(state, goal)
				action = self.act(augmented_state, eval_mode=eval_mode, warmup_phase=warmup_phase)
				reward, next_state = mdp.execute_agent_action(action, option_idx=self.option_idx)

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

			if self.is_at_local_goal(state, goal):
				print(f"{self} successfully reached {goal}")

			if self.is_term_true(state) and self.last_episode_term_triggered != episode:
				print(f"{self} successful (Reached Goal State: {self.is_at_local_goal(state, goal)})")
				self.num_goal_hits += 1
				self.last_episode_term_triggered = episode
				self.effect_set.append(state)

			if self.parent is None and self.is_term_true(state):
				print(f"[{self}] Adding {state.position} to {self.target_salient_event}'s trigger points")
				self.target_salient_event.trigger_points.append(state)

			if self.name != "global_option" and self.get_training_phase() != "initiation_done":
				self.refine_initiation_set_classifier(visited_states, start_state, state, num_steps, step_number)

			# Experience Replay
			if self.name != "global_option" and not warmup_phase:
				self.local_option_experience_replay(option_transitions, goal_state=goal)

				if self.use_her:
					self.local_option_experience_replay(option_transitions, goal_state=state.position)

			return option_transitions, total_reward

		raise Warning("Wanted to execute {}, but initiation condition not met".format(self))

	def is_at_local_goal(self, state, goal_state):
		assert isinstance(state, State)
		global_done = self.is_term_true(state) or state.is_terminal() or self.name == "global_option"
		local_done = self.overall_mdp.sparse_gc_reward_function(state, goal_state, {})[1] if self.use_her else True
		done = global_done and local_done
		return done

	def local_option_experience_replay(self, trajectory, goal_state):
		for state, action, reward, next_state in trajectory:
			augmented_state = self.get_augmented_state(state, goal=goal_state)
			augmented_next_state = self.get_augmented_state(next_state, goal=goal_state)
			done = self.is_at_local_goal(next_state, goal_state)
			if self.use_her:
				reward_func = self.overall_mdp.dense_gc_reward_function if self.dense_reward \
					else self.overall_mdp.sparse_gc_reward_function
				reward, _ = reward_func(next_state, goal_state, info={})
				self.solver.step(augmented_state, action, reward, augmented_next_state, done)
			else:
				self.update_option_solver(state, action, reward, next_state)

	def is_eligible_for_off_policy_triggers(self):
		if "point" in self.overall_mdp.env_name:
			eligible_phase = self.get_training_phase() == "gestation"
		elif "ant" in self.overall_mdp.env_name:
			eligible_phase = self.get_training_phase() != "initiation_done"
		else:
			raise NotImplementedError(self.overall_mdp)

		return eligible_phase and not self.backward_option and not self.started_at_target_salient_event

	def trigger_termination_condition_off_policy(self, trajectory, episode):
		trajectory_so_far = []
		states_so_far = []

		if not self.is_eligible_for_off_policy_triggers():
			return False

		for state, action, reward, next_state in trajectory:  # type: State, np.ndarray, float, State
			trajectory_so_far.append((state, action, reward, next_state))
			states_so_far.append(state)

			if self.is_term_true(next_state) and self.is_valid_init_data(states_so_far) and \
					episode != self.last_episode_term_triggered:

				print(f"Triggered termination condition for {self}")
				self.effect_set.append(next_state)

				if self.parent is None and self.is_term_true(next_state):
					print(f"[{self}] Adding {next_state.position} to {self.target_salient_event}'s trigger points")
					self.target_salient_event.trigger_points.append(next_state)

				states_so_far.append(next_state)  # We want the terminal state to be part of the initiation classifier
				if self.train(trajectory_so_far, states_so_far, episode):
					return True

		return False

	def refine_initiation_set_classifier(self, visited_states, start_state, final_state, num_steps, outer_step_number):
		if self.is_term_true(final_state):  # success

			# No need to add the start state of option execution to positive examples during
			# gestation because there are no negative examples yet to balance
			if self.get_training_phase() == "gestation":
				positive_states = visited_states[-self.buffer_length:]
			else:
				positive_states = [start_state] + visited_states[-self.buffer_length:]

			# if self.is_valid_init_data(positive_states):
			positive_examples = [state.position for state in positive_states]
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
		elif self.num_executions > 0:
			return self.num_goal_hits / self.num_executions
		return 1.
