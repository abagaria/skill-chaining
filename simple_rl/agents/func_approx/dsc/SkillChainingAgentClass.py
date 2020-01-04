# Python imports.
from __future__ import print_function

import sys
sys.path = [""] + sys.path

from collections import deque, defaultdict
from copy import deepcopy
import pdb
import argparse
import os
import random
import numpy as np
from tensorboardX import SummaryWriter
import torch
import itertools

# Other imports.
from simple_rl.mdp.StateClass import State
from simple_rl.agents.func_approx.dsc.OptionClass import Option
from simple_rl.agents.func_approx.dsc.utils import *
from simple_rl.agents.func_approx.ddpg.utils import *
from simple_rl.agents.func_approx.exploration.utils import *
from simple_rl.agents.func_approx.dqn.DQNAgentClass import DQNAgent
from simple_rl.agents.func_approx.dsc.ChainClass import SkillChain


class SkillChaining(object):
	def __init__(self, mdp, max_steps, lr_actor, lr_critic, ddpg_batch_size, device, max_num_options=5,
				 subgoal_reward=0., enable_option_timeout=True, buffer_length=20, num_subgoal_hits_required=3,
				 classifier_type="ocsvm", init_q=None, generate_plots=False, use_full_smdp_update=False,
				 log_dir="", seed=0, tensor_log=False):
		"""
		Args:
			mdp (MDP): Underlying domain we have to solve
			max_steps (int): Number of time steps allowed per episode of the MDP
			lr_actor (float): Learning rate for DDPG Actor
			lr_critic (float): Learning rate for DDPG Critic
			ddpg_batch_size (int): Batch size for DDPG agents
			device (str): torch device {cpu/cuda:0/cuda:1}
			subgoal_reward (float): Hitting a subgoal must yield a supplementary reward to enable local policy
			enable_option_timeout (bool): whether or not the option times out after some number of steps
			buffer_length (int): size of trajectories used to train initiation sets of options
			num_subgoal_hits_required (int): number of times we need to hit an option's termination before learning
			classifier_type (str): Type of classifier we will train for option initiation sets
			init_q (float): If not none, we use this value to initialize the value of a new option
			generate_plots (bool): whether or not to produce plots in this run
			use_full_smdp_update (bool): sparse 0/1 reward or discounted SMDP reward for training policy over options
			log_dir (os.path): directory to store all the scores for this run
			seed (int): We are going to use the same random seed for all the DQN solvers
			tensor_log (bool): Tensorboard logging enable
		"""
		self.mdp = mdp
		self.original_actions = deepcopy(mdp.actions)
		self.max_steps = max_steps
		self.subgoal_reward = subgoal_reward
		self.enable_option_timeout = enable_option_timeout
		self.init_q = init_q
		self.use_full_smdp_update = use_full_smdp_update
		self.generate_plots = generate_plots
		self.buffer_length = buffer_length
		self.num_subgoal_hits_required = num_subgoal_hits_required
		self.log_dir = log_dir
		self.seed = seed
		self.device = torch.device(device)
		self.max_num_options = max_num_options
		self.classifier_type = classifier_type
		self.dense_reward = mdp.dense_reward

		tensor_name = "runs/{}_{}".format(args.experiment_name, seed)
		self.writer = SummaryWriter(tensor_name) if tensor_log else None

		print("Initializing skill chaining with option_timeout={}, seed={}".format(self.enable_option_timeout, seed))

		random.seed(seed)
		np.random.seed(seed)

		self.validation_scores = []

		# This option has an initiation set that is true everywhere and is allowed to operate on atomic timescale only
		self.global_option = Option(overall_mdp=self.mdp, name="global_option", global_solver=None,
									lr_actor=lr_actor, lr_critic=lr_critic, buffer_length=buffer_length,
									ddpg_batch_size=ddpg_batch_size, num_subgoal_hits_required=num_subgoal_hits_required,
									subgoal_reward=self.subgoal_reward, seed=self.seed, max_steps=self.max_steps,
									enable_timeout=self.enable_option_timeout, classifier_type=classifier_type,
									generate_plots=self.generate_plots, writer=self.writer, device=self.device,
									dense_reward=self.dense_reward, chain_id=None)

		self.trained_options = [self.global_option]

		# This is our first untrained option - one that gets us to the goal state from nearby the goal
		# We pick this option when self.agent_over_options thinks that we should
		# Once we pick this option, we will use its internal DDPG solver to take primitive actions until termination
		# Once we hit its termination condition N times, we will start learning its initiation set
		# Once we have learned its initiation set, we will create its child option
		goal_option_1 = Option(overall_mdp=self.mdp, name='goal_option_1', global_solver=self.global_option.solver,
							 lr_actor=lr_actor, lr_critic=lr_critic, buffer_length=buffer_length,
							 ddpg_batch_size=ddpg_batch_size, num_subgoal_hits_required=num_subgoal_hits_required,
							 subgoal_reward=self.subgoal_reward, seed=self.seed, max_steps=self.max_steps,
							 enable_timeout=self.enable_option_timeout, classifier_type=classifier_type,
							 generate_plots=self.generate_plots, writer=self.writer, device=self.device,
							 dense_reward=self.dense_reward, chain_id=1)
		goal_option_2 = Option(overall_mdp=self.mdp, name='goal_option_2', global_solver=self.global_option.solver,
							   lr_actor=lr_actor, lr_critic=lr_critic, buffer_length=buffer_length,
							   ddpg_batch_size=ddpg_batch_size, num_subgoal_hits_required=num_subgoal_hits_required,
							   subgoal_reward=self.subgoal_reward, seed=self.seed, max_steps=self.max_steps,
							   enable_timeout=self.enable_option_timeout, classifier_type=classifier_type,
							   generate_plots=self.generate_plots, writer=self.writer, device=self.device,
							   dense_reward=self.dense_reward, chain_id=2)

		# This is our policy over options
		# We use (double-deep) (intra-option) Q-learning to learn the Q-values of *options* at any queried state Q(s, o)
		# We start with this DQN Agent only predicting Q-values for taking the global_option, but as we learn new
		# options, this agent will predict Q-values for them as well
		self.agent_over_options = DQNAgent(self.mdp.state_space_size(), 1, trained_options=self.trained_options,
										   seed=seed, lr=1e-4, name="GlobalDQN", eps_start=1.0, tensor_log=tensor_log,
										   use_double_dqn=True, writer=self.writer, device=self.device,
										   exploration_strategy="counts", use_position_only_for_exploration=True)

		# Pointer to the current option:
		# 1. This option has the termination set which defines our current goal trigger
		# 2. This option has an untrained initialization set and policy, which we need to train from experience
		self.untrained_options = [goal_option_1, goal_option_2]

		# Keep track of which chain each created option belongs to
		s0 = self.mdp.env._init_positions
		self.s0 = s0
		self.chains = [SkillChain(start_states=s0, target_predicate=target, options=[], chain_id=(i+1),
		 intersecting_options=[], mdp_start_states=s0) for i, target in enumerate(self.mdp.get_target_events())]

		# List of init states seen while running this algorithm
		self.init_states = []

		self.current_option_idx = 1

		# This is our temporally extended exploration option
		# We use it to drive the agent towards unseen parts of the state-space
		# The temporally extended nature of the option allows the agent to solve the "option sink" problem
		# Note that we use a smaller timeout for this option so as to permit creating new options after it
		target_predicate = lambda s: goal_option_1.is_term_true(s) or goal_option_2.is_term_true(s)
		batched_target_predicate = lambda s: np.logical_or(goal_option_1.batched_is_term_true(s),
														   goal_option_2.batched_is_term_true(s))
		# exploration_option = Option(overall_mdp=self.mdp, name="exploration_option", global_solver=self.global_option.solver,
		# 							lr_actor=self.global_option.solver.actor_learning_rate,
		# 							lr_critic=self.global_option.solver.critic_learning_rate,
		# 							buffer_length=self.global_option.buffer_length,
		# 							ddpg_batch_size=self.global_option.solver.batch_size,
		# 							num_subgoal_hits_required=self.num_subgoal_hits_required,
		# 							subgoal_reward=self.subgoal_reward, seed=self.seed, max_steps=self.max_steps,
		# 							enable_timeout=self.enable_option_timeout, classifier_type=self.classifier_type,
		# 							generate_plots=self.generate_plots, writer=self.writer, device=self.device,
		# 							dense_reward=self.dense_reward, chain_id=None, timeout=30,
		# 							init_predicate=target_predicate, batched_init_predicate=batched_target_predicate)

		# # Add the exploration option to the policy over options
		# self._augment_agent_with_new_option(exploration_option, 0.)

		# Debug variables
		self.global_execution_states = []
		self.num_option_executions = defaultdict(lambda : [])
		self.option_rewards = defaultdict(lambda : [])
		self.option_qvalues = defaultdict(lambda : [])
		self.num_options_history = []

	def add_negative_examples(self, new_option):
		# TODO: Cannot do this in skill graphs because the forward and backward skills have to overlap

		if not new_option.initialize_everywhere:
			for option in self.trained_options[1:]:
				new_option.negative_examples += option.positive_examples

		else:

			# get the pair of intersecting options
			chain = self.chains[2]
			intersecting_options = chain.intersecting_options

			# go through all the children of the intersecting options and add their positive examples
			for option in intersecting_options:
				for child_option in option.children:
					if child_option in self.trained_options:
						new_option.negative_examples += child_option.positive_examples

	def init_state_in_option(self, option):
		for start_state in self.init_states:
			if option.is_init_true(start_state):
				return True
		return False

	def state_in_any_option(self, state):
		for option in self.trained_options[1:]:
			if option.is_init_true(state):
				return True
		return False

	def create_children_options(self, option):
		o1 = self.create_child_option(option)
		o2 = self.create_child_option(option.parent) if option.parent is not None else None
		return o1, o2

	def create_child_option(self, parent_option):

		# Don't create a child option if you already include the init state of the MDP
		# Also enforce the max branching factor of our skill tree
		if self.init_state_in_option(parent_option) or len(parent_option.children) >= parent_option.max_num_children:
			return None

		# Create new option whose termination is the initiation of the option we just trained
		if parent_option.name == "goal_option_1":
			prefix = "option_1"
		elif parent_option.name == "goal_option_2":
			prefix = "option_2"
		else:
			prefix = parent_option.name
		name = prefix + "_{}".format(len(parent_option.children))
		print("Creating {} with parent {}".format(name, parent_option.name))

		old_untrained_option_id = id(parent_option)
		new_untrained_option = Option(self.mdp, name=name, global_solver=self.global_option.solver,
									  lr_actor=parent_option.solver.actor_learning_rate,
									  lr_critic=parent_option.solver.critic_learning_rate,
									  ddpg_batch_size=parent_option.solver.batch_size,
									  subgoal_reward=self.subgoal_reward,
									  buffer_length=self.buffer_length,
									  classifier_type=self.classifier_type,
									  num_subgoal_hits_required=self.num_subgoal_hits_required,
									  seed=self.seed, parent=parent_option,  max_steps=self.max_steps,
									  enable_timeout=self.enable_option_timeout, chain_id=parent_option.chain_id,
									  writer=self.writer, device=self.device, dense_reward=self.dense_reward,
									  initialize_everywhere=parent_option.initialize_everywhere)

		new_untrained_option_id = id(new_untrained_option)
		assert new_untrained_option_id != old_untrained_option_id, "Checking python references"
		assert id(new_untrained_option.parent) == old_untrained_option_id, "Checking python references"

		parent_option.children.append(new_untrained_option)

		return new_untrained_option

	def make_off_policy_updates_for_options(self, state, action, reward, next_state):
		for option in self.trained_options: # type: Option
			option.off_policy_update(state, action, reward, next_state)\


	def make_smdp_update(self, state, action, total_discounted_reward, next_state, option_transitions):
		"""
		Use Intra-Option Learning for sample efficient learning of the option-value function Q(s, o)
		Args:
			state (State): state from which we started option execution
			action (int): option taken by the global solver
			total_discounted_reward (float): cumulative reward from the overall SMDP update
			next_state (State): state we landed in after executing the option
			option_transitions (list): list of (s, a, r, s') tuples representing the trajectory during option execution
		"""
		assert self.subgoal_reward == 0, "This kind of SMDP update only makes sense when subgoal reward is 0"

		def get_reward(transitions):
			gamma = self.global_option.solver.gamma
			raw_rewards = [tt[2] for tt in transitions]
			return sum([(gamma ** idx) * rr for idx, rr in enumerate(raw_rewards)])

		selected_option = self.trained_options[action]  # type: Option
		for i, transition in enumerate(option_transitions):
			start_state = transition[0]
			if selected_option.is_init_true(start_state):
				if self.use_full_smdp_update:
					sub_transitions = option_transitions[i:]
					option_reward = get_reward(sub_transitions)
					self.agent_over_options.step(start_state.features(), action, option_reward, next_state.features(),
												 next_state.is_terminal(), num_steps=len(sub_transitions))
				else:
					option_reward = self.subgoal_reward if selected_option.is_term_true(next_state) else -1.
					self.agent_over_options.step(start_state.features(), action, option_reward, next_state.features(),
												 next_state.is_terminal(), num_steps=1)

	def get_init_q_value_for_new_option(self, newly_trained_option):
		global_solver = self.agent_over_options  # type: DQNAgent
		state_option_pairs = newly_trained_option.final_transitions
		q_values = []
		for state, option_idx in state_option_pairs:
			q_value = global_solver.get_qvalue(state.features(), option_idx)
			q_values.append(q_value)
		return np.max(q_values)

	def _augment_agent_with_new_option(self, newly_trained_option, init_q_value):
		"""
		Train the current untrained option and initialize a new one to target.
		Add the newly_trained_option as a new node to the Q-function over options
		Args:
			newly_trained_option (Option)
			init_q_value (float): if given use this, else compute init_q optimistically
		"""
		# Add the trained option to the action set of the global solver
		if newly_trained_option not in self.trained_options:
			self.trained_options.append(newly_trained_option)

		# Add the newly trained option to the corresponding skill chain
		if newly_trained_option.chain_id is not None:
			chain_idx = newly_trained_option.chain_id - 1
			self.chains[chain_idx].options.append(newly_trained_option)

		# Set the option_idx for the newly added option
		newly_trained_option.option_idx = self.current_option_idx
		self.current_option_idx += 1

		# Augment the global DQN with the newly trained option
		num_actions = len(self.trained_options)
		new_global_agent = DQNAgent(self.agent_over_options.state_size, num_actions, self.trained_options,
									seed=self.seed, name=self.agent_over_options.name,
									eps_start=self.agent_over_options.epsilon,
									tensor_log=self.agent_over_options.tensor_log,
									use_double_dqn=self.agent_over_options.use_ddqn,
									lr=self.agent_over_options.learning_rate,
									writer=self.writer, device=self.device,
									exploration_strategy="counts", use_position_only_for_exploration=True)
		new_global_agent.replay_buffer = self.agent_over_options.replay_buffer

		init_q = self.get_init_q_value_for_new_option(newly_trained_option) if init_q_value is None else init_q_value
		print("Initializing new option node with q value {}".format(init_q))
		new_global_agent.policy_network.initialize_with_smaller_network(self.agent_over_options.policy_network, init_q)
		new_global_agent.target_network.initialize_with_smaller_network(self.agent_over_options.target_network, init_q)

		self.agent_over_options = new_global_agent

	def act(self, state):
		# Query the global Q-function to determine which option to take in the current state
		option_idx = self.agent_over_options.act(state.features(), train_mode=True)
		self.agent_over_options.update_epsilon()

		# Selected option
		selected_option = self.trained_options[option_idx]  # type: Option

		return selected_option

	def take_action(self, state, episode_number, step_number, episode_option_executions):
		"""
		Either take a primitive action from `state` or execute a closed-loop option policy.
		Args:
			state (State)
			episode_number (int)
			step_number (int): which iteration of the control loop we are on
			episode_option_executions (defaultdict)

		Returns:
			experiences (list): list of (s, a, r, s') tuples
			reward (float): sum of all rewards accumulated while executing chosen action
			next_state (State): state we landed in after executing chosen action
		"""
		selected_option = self.act(state)

		if selected_option.is_term_true(state):
			selected_option = self.global_option

		option_transitions, discounted_reward = selected_option.execute_option_in_mdp(self.mdp, episode_number, step_number)

		option_reward = self.get_reward_from_experiences(option_transitions)
		next_state = self.get_next_state_from_experiences(option_transitions)

		# If we triggered the untrained option's termination condition, add to its buffer of terminal transitions
		for untrained_option in self.untrained_options:
			if untrained_option.is_term_true(next_state) and not untrained_option.is_term_true(state):
				untrained_option.final_transitions.append((state, selected_option.option_idx))

		# Add data to train Q(s, o)
		self.make_smdp_update(state, selected_option.option_idx, discounted_reward, next_state, option_transitions)

		# Debug logging
		episode_option_executions[selected_option.name] += 1
		self.option_rewards[selected_option.name].append(discounted_reward)

		sampled_q_value = self.sample_qvalue(selected_option)
		self.option_qvalues[selected_option.name].append(sampled_q_value)
		if self.writer is not None:
			self.writer.add_scalar("{}_q_value".format(selected_option.name),
								   sampled_q_value, selected_option.num_executions)

		return option_transitions, option_reward, next_state, len(option_transitions)

	def sample_qvalue(self, option):
		if len(option.solver.replay_buffer) > 500:
			sample_experiences = option.solver.replay_buffer.sample(batch_size=500)
			sample_states = torch.from_numpy(sample_experiences[0]).float().to(self.device)
			sample_actions = torch.from_numpy(sample_experiences[1]).float().to(self.device)
			sample_qvalues = option.solver.get_qvalues(sample_states, sample_actions)
			return sample_qvalues.mean().item()
		return 0.0

	@staticmethod
	def get_next_state_from_experiences(experiences):
		return experiences[-1][-1]

	@staticmethod
	def get_reward_from_experiences(experiences):
		total_reward = 0.
		for experience in experiences:
			reward = experience[2]
			total_reward += reward
		return total_reward

	def should_create_more_options(self):
		""" Continue chaining as long as any chain is still accepting new options. """
		return any([chain.should_continue_chaining(self.chains) for chain in self.chains])

	def get_intersecting_options(self):
		for chain in self.chains:  # type: SkillChain
			intersecting_options = chain.detect_intersection_with_other_chains(self.chains)
			if intersecting_options is not None:
				return intersecting_options
		return None

	def get_current_salient_events(self):
		""" Get a list of states that satisfy final target events (i.e not init of an option) in the MDP so far. """

		def get_option_terminal_states(root_option):
			""" Given a root option, get the final (goal) states from successful trajectories. """
			successful_trajectories = root_option.experience_buffer
			option_goal_states = [trajectory[-1].next_state for trajectory in successful_trajectories]
			return option_goal_states

		root_options = [chain.options[0] for chain in self.chains if chain.options[0].parent is None]
		salient_states = [get_option_terminal_states(option) for option in root_options]
		salient_states = list(itertools.chain.from_iterable(salient_states))
		return salient_states

	def create_intersection_salient_event(self, option1, option2):
		"""
		When 2 chains intersect, we treat the region of intersection as a target region. This method
		returns an Option whose termination region is such a region of intersection.
		Args:
			option1 (Option): intersecting option from chain 1
			option2 (Option): intersecting option from chain 2

		Returns:
			new_option (Option): whose termination set is the intersection of initiation sets of o1 and o2
		"""

		new_option = Option(self.mdp, name="intersection_option", global_solver=self.global_option.solver,
							  lr_actor=option1.solver.actor_learning_rate,
							  lr_critic=option1.solver.critic_learning_rate,
							  ddpg_batch_size=option1.solver.batch_size,
							  subgoal_reward=self.subgoal_reward,
							  buffer_length=self.buffer_length,
							  classifier_type=self.classifier_type,
							  num_subgoal_hits_required=self.num_subgoal_hits_required,
							  seed=self.seed, parent=None,  max_steps=self.max_steps,
							  enable_timeout=self.enable_option_timeout, chain_id=3,
							  intersecting_options=[option1, option2],
							  initialize_everywhere=True,
							  writer=self.writer, device=self.device, dense_reward=self.dense_reward)

		current_salient_states = self.get_current_salient_events()
		new_chain = SkillChain(start_states=current_salient_states, target_predicate=None, chain_id=3,
							   options=[], intersecting_options=[option1, option2],
							   mdp_start_states=self.s0)
		self.chains.append(new_chain)

		# Allow agent to use this option to encourage chaining to it
		self._augment_agent_with_new_option(new_option, 0.)

		# Also reset the exploration module to encourage adding skills to the new chain
		# self.agent_over_options.density_model.reset()

		return new_option

	def skill_chaining(self, num_episodes, num_steps):

		# For logging purposes
		per_episode_scores = []
		per_episode_durations = []
		last_10_scores = deque(maxlen=10)
		last_10_durations = deque(maxlen=10)

		for episode in range(num_episodes):

			self.mdp.reset()
			score = 0.
			step_number = 0

			# TODO: Does this flag's logic need to be changed? YES when there is a list of target events in the MDP
			uo_episode_terminated = False

			state = deepcopy(self.mdp.init_state)
			self.init_states.append(deepcopy(state))
			experience_buffer = []
			state_buffer = []
			episode_option_executions = defaultdict(lambda : 0)

			while step_number < num_steps:
				# pdb.set_trace()
				experiences, reward, state, steps = self.take_action(state, episode, step_number, episode_option_executions)
				score += reward
				step_number += steps
				for experience in experiences:
					experience_buffer.append(experience)
					state_buffer.append(experience[0])

				# Don't forget to add the last s' to the buffer
				if state.is_terminal() or (step_number == num_steps - 1):
					state_buffer.append(state)

				# In the skill-tree setting we could be learning many options at the current time
				# We must iterate through all such options and check if the current transition
				# triggered the termination condition of any such option
				for untrained_option in self.untrained_options:

					if untrained_option.is_term_true(state) and (not uo_episode_terminated) and \
							self.max_num_options > 0 and untrained_option.get_training_phase() == "gestation" and \
							untrained_option.is_valid_init_data(state_buffer):

						print("\rTriggered termination condition of ", untrained_option)
						uo_episode_terminated = True
						if untrained_option.train(experience_buffer, state_buffer, episode):

							# If we are in chain # 3, we have already augmented with the option
							if not untrained_option.initialize_everywhere:
								self._augment_agent_with_new_option(untrained_option, self.init_q)

							# Visualize option you just learned
							if untrained_option.classifier_type == "ocsvm":
								plot_one_class_initiation_classifier(untrained_option, episode, args.experiment_name)
							else:
								plot_two_class_classifier(untrained_option, episode, args.experiment_name)

					# In the skill-graph setting, we have to check if the current option's chain
					# is still accepting new options
					if self.should_create_more_options() and untrained_option.get_training_phase() == "initiation_done"\
							and self.chains[untrained_option.chain_id - 1].should_continue_chaining(self.chains):

						# Debug visualization
						plot_two_class_classifier(untrained_option, episode, args.experiment_name)

						# We fix the learned option's initiation set and remove it from the list of target events
						self.untrained_options.remove(untrained_option)
						new_option_1, new_option_2 = self.create_children_options(untrained_option)

						if new_option_1 is not None:
							self.untrained_options.append(new_option_1)
							self.add_negative_examples(new_option_1)
							if new_option_1.initialize_everywhere:
								self._augment_agent_with_new_option(new_option_1, 0.)

						if new_option_2 is not None:
							self.untrained_options.append(new_option_2)
							self.add_negative_examples(new_option_2)
							if new_option_2.initialize_everywhere:
								self._augment_agent_with_new_option(new_option_2, 0.)

				# Detect intersection salience to start building skill graph
				intersecting_options = self.get_intersecting_options()
				if intersecting_options is not None and len(self.chains) < 3:
					new_option = self.create_intersection_salient_event(intersecting_options[0], intersecting_options[1])
					self.untrained_options.append(new_option)

				if state.is_terminal():
					break

			last_10_scores.append(score)
			last_10_durations.append(step_number)
			per_episode_scores.append(score)
			per_episode_durations.append(step_number)

			self._log_dqn_status(episode, last_10_scores, episode_option_executions, last_10_durations)

		return per_episode_scores, per_episode_durations

	def _log_dqn_status(self, episode, last_10_scores, episode_option_executions, last_10_durations):

		print('\rEpisode {}\tAverage Score: {:.2f}\tDuration: {:.2f} steps\tGO Eps: {:.2f}'.format(
			episode, np.mean(last_10_scores), np.mean(last_10_durations), self.global_option.solver.epsilon))

		self.num_options_history.append(len(self.trained_options))

		if self.writer is not None:
			self.writer.add_scalar("Episodic scores", last_10_scores[-1], episode)

		if episode % 10 == 0:
			print('\rEpisode {}\tAverage Score: {:.2f}\tDuration: {:.2f} steps\tGO Eps: {:.2f}'.format(
				episode, np.mean(last_10_scores), np.mean(last_10_durations), self.global_option.solver.epsilon))
			# visualize_bonus_for_actions(self.agent_over_options.density_model, self.trained_options, episode,
			# 							args.experiment_name, args.seed)

		if episode > 0 and episode % 100 == 0:
			# eval_score, trajectory = self.trained_forward_pass(render=False)
			eval_score, trajectory = 0., []
			self.validation_scores.append(eval_score)
			print("\rEpisode {}\tValidation Score: {:.2f}".format(episode, eval_score))

		if self.generate_plots and episode % 10 == 0:
			render_sampled_value_function(self.global_option.solver, episode, args.experiment_name)

		for trained_option in self.trained_options:  # type: Option
			self.num_option_executions[trained_option.name].append(episode_option_executions[trained_option.name])
			if self.writer is not None:
				self.writer.add_scalar("{}_executions".format(trained_option.name),
									   episode_option_executions[trained_option.name], episode)

	def save_all_models(self):
		for option in self.trained_options: # type: Option
			save_model(option.solver, args.episodes, best=False)

	def save_all_scores(self, pretrained, scores, durations):
		print("\rSaving training and validation scores..")
		training_scores_file_name = "sc_pretrained_{}_training_scores_{}.pkl".format(pretrained, self.seed)
		training_durations_file_name = "sc_pretrained_{}_training_durations_{}.pkl".format(pretrained, self.seed)
		validation_scores_file_name = "sc_pretrained_{}_validation_scores_{}.pkl".format(pretrained, self.seed)
		num_option_history_file_name = "sc_pretrained_{}_num_options_per_epsiode_{}.pkl".format(pretrained, self.seed)

		if self.log_dir:
			training_scores_file_name = os.path.join(self.log_dir, training_scores_file_name)
			training_durations_file_name = os.path.join(self.log_dir, training_durations_file_name)
			validation_scores_file_name = os.path.join(self.log_dir, validation_scores_file_name)
			num_option_history_file_name = os.path.join(self.log_dir, num_option_history_file_name)

		with open(training_scores_file_name, "wb+") as _f:
			pickle.dump(scores, _f)
		with open(training_durations_file_name, "wb+") as _f:
			pickle.dump(durations, _f)
		with open(validation_scores_file_name, "wb+") as _f:
			pickle.dump(self.validation_scores, _f)
		with open(num_option_history_file_name, "wb+") as _f:
			pickle.dump(self.num_options_history, _f)

	def perform_experiments(self):
		for option in self.trained_options:
			visualize_dqn_replay_buffer(option.solver, args.experiment_name)

		for i, o in enumerate(self.trained_options):
			plt.subplot(1, len(self.trained_options), i + 1)
			plt.plot(self.option_qvalues[o.name])
			plt.title(o.name)
		plt.savefig("value_function_plots/{}/sampled_q_so_{}.png".format(args.experiment_name, self.seed))
		plt.close()

		for option in self.trained_options:
			visualize_next_state_reward_heat_map(option.solver, args.episodes, args.experiment_name)

		for i, o in enumerate(self.trained_options):
			plt.subplot(1, len(self.trained_options), i + 1)
			plt.plot(o.taken_or_not)
			plt.title(o.name)
		plt.savefig("value_function_plots/{}_taken_or_not_{}.png".format(args.experiment_name, self.seed))
		plt.close()

	def trained_forward_pass(self, render=True):
		"""
		Called when skill chaining has finished training: execute options when possible and then atomic actions
		Returns:
			overall_reward (float): score accumulated over the course of the episode.
		"""
		self.mdp.reset()
		state = deepcopy(self.mdp.init_state)
		overall_reward = 0.
		self.mdp.render = render
		num_steps = 0
		option_trajectories = []

		while not state.is_terminal() and num_steps < self.max_steps:
			selected_option = self.act(state)

			option_reward, next_state, num_steps, option_state_trajectory = selected_option.trained_option_execution(self.mdp, num_steps)
			overall_reward += option_reward

			# option_state_trajectory is a list of (o, s) tuples
			option_trajectories.append(option_state_trajectory)

			state = next_state

		return overall_reward, option_trajectories

def create_log_dir(experiment_name):
	path = os.path.join(os.getcwd(), experiment_name)
	try:
		os.mkdir(path)
	except OSError:
		print("Creation of the directory %s failed" % path)
	else:
		print("Successfully created the directory %s " % path)
	return path


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument("--experiment_name", type=str, help="Experiment Name")
	parser.add_argument("--device", type=str, help="cpu/cuda:0/cuda:1")
	parser.add_argument("--env", type=str, help="name of gym environment", default="Pendulum-v0")
	parser.add_argument("--pretrained", type=bool, help="whether or not to load pretrained options", default=False)
	parser.add_argument("--seed", type=int, help="Random seed for this run (default=0)", default=0)
	parser.add_argument("--episodes", type=int, help="# episodes", default=200)
	parser.add_argument("--steps", type=int, help="# steps", default=1000)
	parser.add_argument("--subgoal_reward", type=float, help="SkillChaining subgoal reward", default=0.)
	parser.add_argument("--lr_a", type=float, help="DDPG Actor learning rate", default=1e-4)
	parser.add_argument("--lr_c", type=float, help="DDPG Critic learning rate", default=1e-3)
	parser.add_argument("--ddpg_batch_size", type=int, help="DDPG Batch Size", default=64)
	parser.add_argument("--render", type=bool, help="Render the mdp env", default=False)
	parser.add_argument("--option_timeout", type=bool, help="Whether option times out at 200 steps", default=False)
	parser.add_argument("--generate_plots", type=bool, help="Whether or not to generate plots", default=False)
	parser.add_argument("--tensor_log", type=bool, help="Enable tensorboard logging", default=False)
	parser.add_argument("--control_cost", type=bool, help="Penalize high actuation solutions", default=False)
	parser.add_argument("--dense_reward", type=bool, help="Use dense/sparse rewards", default=False)
	parser.add_argument("--max_num_options", type=int, help="Max number of options we can learn", default=5)
	parser.add_argument("--num_subgoal_hits", type=int, help="Number of subgoal hits to learn an option", default=3)
	parser.add_argument("--buffer_len", type=int, help="buffer size used by option to create init sets", default=20)
	parser.add_argument("--classifier_type", type=str, help="ocsvm/elliptic for option initiation clf", default="ocsvm")
	parser.add_argument("--init_q", type=str, help="compute/zero", default="zero")
	parser.add_argument("--use_smdp_update", type=bool, help="sparse/SMDP update for option policy", default=False)
	args = parser.parse_args()

	if "reacher" in args.env.lower():
		from simple_rl.tasks.dm_fixed_reacher.FixedReacherMDPClass import FixedReacherMDP
		overall_mdp = FixedReacherMDP(seed=args.seed, difficulty=args.difficulty, render=args.render)
		state_dim = overall_mdp.init_state.features().shape[0]
		action_dim = overall_mdp.env.action_spec().minimum.shape[0]
	elif "maze" in args.env.lower():
		from simple_rl.tasks.point_maze.PointMazeMDPClass import PointMazeMDP
		overall_mdp = PointMazeMDP(dense_reward=args.dense_reward, seed=args.seed, render=args.render)
		state_dim = 6
		action_dim = 2
	elif "point" in args.env.lower():
		from simple_rl.tasks.point_env.PointEnvMDPClass import PointEnvMDP
		overall_mdp = PointEnvMDP(control_cost=args.control_cost, render=args.render)
		state_dim = 4
		action_dim = 2
	else:
		from simple_rl.tasks.gym.GymMDPClass import GymMDP
		overall_mdp = GymMDP(args.env, render=args.render)
		state_dim = overall_mdp.env.observation_space.shape[0]
		action_dim = overall_mdp.env.action_space.shape[0]
		overall_mdp.env.seed(args.seed)

	# Create folders for saving various things
	logdir = create_log_dir(args.experiment_name)
	create_log_dir("saved_runs")
	create_log_dir("value_function_plots")
	create_log_dir("initiation_set_plots")
	create_log_dir("value_function_plots/{}".format(args.experiment_name))
	create_log_dir("initiation_set_plots/{}".format(args.experiment_name))

	print("Training skill chaining agent from scratch with a subgoal reward {}".format(args.subgoal_reward))
	print("MDP InitState = ", overall_mdp.init_state)

	q0 = 0. if args.init_q == "zero" else None

	chainer = SkillChaining(overall_mdp, args.steps, args.lr_a, args.lr_c, args.ddpg_batch_size,
							seed=args.seed, subgoal_reward=args.subgoal_reward,
							log_dir=logdir, num_subgoal_hits_required=args.num_subgoal_hits,
							enable_option_timeout=args.option_timeout, init_q=q0, use_full_smdp_update=args.use_smdp_update,
							generate_plots=args.generate_plots, tensor_log=args.tensor_log, device=args.device)
	episodic_scores, episodic_durations = chainer.skill_chaining(args.episodes, args.steps)

	# Log performance metrics
	# chainer.save_all_models()
	# chainer.perform_experiments()
	chainer.save_all_scores(args.pretrained, episodic_scores, episodic_durations)
