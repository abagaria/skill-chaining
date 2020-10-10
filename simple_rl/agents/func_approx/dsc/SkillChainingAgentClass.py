# Python imports.
from __future__ import print_function

import sys
sys.path = [""] + sys.path

from collections import deque, defaultdict
from copy import deepcopy
import ipdb
import argparse
import os
import random
import numpy as np
from tensorboardX import SummaryWriter
import torch
import itertools
from tqdm import tqdm

# Other imports.
from simple_rl.mdp.StateClass import State
from simple_rl.agents.func_approx.dsc.OptionClass import Option
from simple_rl.agents.func_approx.dsc.utils import *
from simple_rl.agents.func_approx.ddpg.utils import *
from simple_rl.agents.func_approx.exploration.utils import *
from simple_rl.agents.func_approx.dqn.DQNAgentClass import DQNAgent
from simple_rl.agents.func_approx.dsc.ChainClass import SkillChain
from simple_rl.agents.func_approx.dsc.SalientEventClass import SalientEvent, LearnedSalientEvent


class SkillChaining(object):
	def __init__(self, mdp, max_steps, lr_actor, lr_critic, ddpg_batch_size, device, max_num_options=5,
				 subgoal_reward=0., enable_option_timeout=True, buffer_length=20, num_subgoal_hits_required=3,
				 classifier_type="ocsvm", init_q=None, generate_plots=False, use_full_smdp_update=False,
				 start_state_salience=False, option_intersection_salience=False, event_intersection_salience=False,
				 pretrain_option_policies=False, create_backward_options=False, learn_backward_options_offline=False,
				 update_global_solver=False, use_warmup_phase=False, dense_reward=False, use_her=False,
				 use_her_locally=False, off_policy_update_type="none", allow_her_initialization=False,
				 log_dir="", seed=0, tensor_log=False, experiment_name="", use_hardcoded_init_sets=False, use_hardcoded_goal_regions=False):
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
			start_state_salience (bool): Treat the start state of the MDP as a salient event OR create intersection events
			option_intersection_salience (bool): Should we treat the intersection b/w options as a salient event
			event_intersection_salience (bool): Should we treat the option-event intersections as salient events
			pretrain_option_policies (bool): Whether to pre-train option policies with the global option
			create_backward_options (bool): Whether to spend time learning back options for skill graphs
			learn_backward_options_offline (bool): Whether to learn initiation sets for back option or reuse forward ones
			update_global_solver (bool): Whether to learn a global option policy (this can increase wall clock time in ant)
			use_warmup_phase (bool): If true, then option will act randomly during its gestation period
			dense_reward (bool): Whether DSC will use dense rewards to train option policies
			use_her (bool): Whether DSC will use Hindsight Experience Replay
			use_her_locally (bool): Whether the local options will use Hindsight Experience Replay
			off_policy_update_type (bool): What kind of off-policy updates to make to option policies not used
			allow_her_initialization (bool): Whether option-policies are allowed to be initialized using the
											 goal-conditioned global-option policy
			log_dir (os.path): directory to store all the scores for this run
			seed (int): We are going to use the same random seed for all the DQN solvers
			tensor_log (bool): Tensorboard logging enable
			experiment_name (str)
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
		self.dense_reward = dense_reward
		self.use_her = use_her
		self.use_her_locally = use_her_locally
		self.start_state_salience = start_state_salience
		self.option_intersection_salience = option_intersection_salience
		self.event_intersection_salience = event_intersection_salience
		self.pretrain_option_policies = pretrain_option_policies
		self.create_backward_options = create_backward_options
		self.learn_backward_options_offline = learn_backward_options_offline
		self.off_policy_update_type = off_policy_update_type
		self.allow_her_initialization = allow_her_initialization
		self.update_global_solver = update_global_solver
		self.use_warmup_phase = use_warmup_phase
		self.experiment_name = experiment_name
		self.use_hardcoded_init_sets = use_hardcoded_init_sets
		self.use_hardcoded_goal_regions = use_hardcoded_goal_regions

		assert off_policy_update_type in ("none", "single", "batched"), off_policy_update_type

		tensor_name = "runs/{}_{}".format(self.experiment_name, seed)
		self.writer = SummaryWriter(tensor_name) if tensor_log else None

		print("Initializing skill chaining with option_timeout={}, seed={}".format(self.enable_option_timeout, seed))

		random.seed(seed)
		np.random.seed(seed)

		self.validation_scores = []

		# This option has an initiation set that is true everywhere and is allowed to operate on atomic timescale only
		# if not self.mdp.task_agnostic:
		# 	raise NotImplementedError("Need target_salient_event for the global-option")

		self.global_option = Option(overall_mdp=self.mdp, name="global_option",
									global_solver=None, target_salient_event=self.mdp.get_original_target_events()[0],
									lr_actor=lr_actor, lr_critic=lr_critic, buffer_length=buffer_length,
									ddpg_batch_size=ddpg_batch_size, num_subgoal_hits_required=num_subgoal_hits_required,
									subgoal_reward=self.subgoal_reward, seed=self.seed, max_steps=self.max_steps,
									enable_timeout=self.enable_option_timeout, classifier_type=classifier_type,
									generate_plots=self.generate_plots, writer=self.writer, device=self.device,
									use_warmup_phase=self.use_warmup_phase,
									update_global_solver=self.update_global_solver,
									use_her=self.use_her,
									dense_reward=self.dense_reward,
									chain_id=None, option_idx=0,
									allow_her_initialization=self.allow_her_initialization, 
									use_hardcoded_init_sets=self.use_hardcoded_init_sets, 
									use_hardcoded_goal_regions=self.use_hardcoded_goal_regions)

		self.trained_options = [self.global_option]

		# Pointer to the current option:
		# 1. This option has the termination set which defines our current goal trigger
		# 2. This option has an untrained initialization set and policy, which we need to train from experience
		self.untrained_options = []

		# This is our first untrained option - one that gets us to the goal state from nearby the goal
		# We pick this option when self.agent_over_options thinks that we should
		# Once we pick this option, we will use its internal DDPG solver to take primitive actions until termination
		# Once we hit its termination condition N times, we will start learning its initiation set
		# Once we have learned its initiation set, we will create its child option
		for i, salient_event in enumerate(self.mdp.get_original_target_events()):
			goal_option = Option(overall_mdp=self.mdp, name=f'goal_option_{i + 1}', global_solver=self.global_option.solver,
								 lr_actor=lr_actor, lr_critic=lr_critic, buffer_length=buffer_length,
								 ddpg_batch_size=ddpg_batch_size, num_subgoal_hits_required=num_subgoal_hits_required,
								 subgoal_reward=self.subgoal_reward, seed=self.seed, max_steps=self.max_steps,
								 enable_timeout=self.enable_option_timeout, classifier_type=classifier_type,
								 generate_plots=self.generate_plots, writer=self.writer, device=self.device,
								 dense_reward=self.dense_reward, chain_id=i+1, max_num_children=1,
								 use_warmup_phase=self.use_warmup_phase, update_global_solver=self.update_global_solver,
								 target_salient_event=salient_event, option_idx=i+1, use_her=use_her_locally,
								 allow_her_initialization=self.allow_her_initialization, use_hardcoded_init_sets=self.use_hardcoded_init_sets, 
								 use_hardcoded_goal_regions=self.use_hardcoded_goal_regions, init_salient_event=self.mdp.get_start_state_salient_event())

			self.untrained_options.append(goal_option)

		self.current_option_idx = 1	


		# This is our policy over options
		# We use (double-deep) (intra-option) Q-learning to learn the Q-values of *options* at any queried state Q(s, o)
		# We start with this DQN Agent only predicting Q-values for taking the global_option, but as we learn new
		# options, this agent will predict Q-values for them as well
		state_size = self.mdp.state_space_size() + 2 if self.use_her else self.mdp.state_space_size()
		self.agent_over_options = DQNAgent(state_size, 1, trained_options=self.trained_options,
										   seed=seed, lr=1e-4, name="GlobalDQN", eps_start=0.1, tensor_log=tensor_log,
										   use_double_dqn=True, writer=self.writer, device=self.device,
										   exploration_strategy="eps-greedy")

		# Keep track of which chain each created option belongs to
		start_state_salient_event = self.mdp.get_start_state_salient_event()
		self.s0 = start_state_salient_event.trigger_points
		self.chains = [SkillChain(options=[], chain_id=(i+1),
		 			   			  intersecting_options=[],
								  target_salient_event=salient_event, init_salient_event=start_state_salient_event,
								  option_intersection_salience=option_intersection_salience,
								  event_intersection_salience=event_intersection_salience,
								  mdp_init_salient_event=self.mdp.get_start_state_salient_event())
					   for i, salient_event in enumerate(self.mdp.get_original_target_events())]

		if self.use_hardcoded_init_sets:
			goal_option = self.untrained_options[0]
			option_1_0 = self.create_child_option(goal_option)
			option_1_0_0 = self.create_child_option(option_1_0)
			self.add_new_options_to_skill_chains([option_1_0, option_1_0_0])

		# List of init states seen while running this algorithm
		self.init_states = []

		self.current_intersecting_chains = []

		# If initialize_everywhere is True, also add to trained_options
		for option in self.untrained_options:  # type: Option
			if option.initialize_everywhere:
				self.augment_agent_with_new_option(option, 0.)

		# Debug variables
		self.global_execution_states = []
		self.num_option_executions = defaultdict(lambda : [])
		self.option_rewards = defaultdict(lambda : [])
		self.option_qvalues = defaultdict(lambda : [])
		self.num_options_history = []

	def create_chain_targeting_new_salient_event(self, salient_event, init_salient_event=None, eval_mode=False):
		"""
		Given a new `salient_event`, create an option and the corresponding skill chain targeting it.

		Args:
			salient_event (SalientEvent)
			init_salient_event (SalientEvent)
			eval_mode (bool)

		Returns:
			skill_chain (SkillChain)
		"""
		chain_id = len(self.chains) + 1
		allow_her_initialization = True if eval_mode else self.allow_her_initialization
		init_salient_event = self.mdp.get_start_state_salient_event() if init_salient_event is None else init_salient_event
		goal_option = Option(overall_mdp=self.mdp, name=f'goal_option_{chain_id}',
							 global_solver=self.global_option.solver,
							 lr_actor=self.global_option.solver.actor_learning_rate,
							 lr_critic=self.global_option.solver.critic_learning_rate,
							 ddpg_batch_size=self.global_option.solver.batch_size,
							 subgoal_reward=self.subgoal_reward,
							 buffer_length=self.buffer_length,
							 classifier_type=self.classifier_type,
							 num_subgoal_hits_required=self.num_subgoal_hits_required,
							 seed=self.seed, parent=None, max_steps=self.max_steps,
							 enable_timeout=self.enable_option_timeout,
							 chain_id=chain_id,
							 initialize_everywhere=True, max_num_children=1,
							 writer=self.writer, device=self.device,
							 dense_reward=self.dense_reward,
							 use_warmup_phase=self.use_warmup_phase,
							 update_global_solver=self.update_global_solver,
							 init_salient_event=init_salient_event,
							 target_salient_event=salient_event,
							 use_her=self.use_her_locally,
							 allow_her_initialization=allow_her_initialization)
		self.untrained_options.append(goal_option)

		new_chain = SkillChain(options=[], chain_id=chain_id,
							   intersecting_options=[],
							   target_salient_event=salient_event,
							   init_salient_event=init_salient_event,
							   option_intersection_salience=self.option_intersection_salience,
							   event_intersection_salience=self.event_intersection_salience,
							   mdp_init_salient_event=self.mdp.get_start_state_salient_event())
		self.add_skill_chain(new_chain)

		if goal_option.initialize_everywhere:
			init_q = +10. if eval_mode else 0.
			self.augment_agent_with_new_option(goal_option, init_q)

		# Plot the option's VF if we did some fancy initialization
		if goal_option.initialize_with_her and goal_option.target_salient_event.get_target_position() is not None:
			goal = goal_option.target_salient_event.get_target_position()
			make_chunked_goal_conditioned_value_function_plot(goal_option.solver, goal,
															  -1, self.seed, self.experiment_name,
															  replay_buffer=self.global_option.solver.replay_buffer)

		return new_chain

	def add_skill_chain(self, new_chain):
		if new_chain not in self.chains:
			self.chains.append(new_chain)

	def add_negative_examples(self, new_option):
		""" When creating an option, add the positive examples of all the options
			in the forward chain as negative examples for the current option.
		"""
		for option in self.trained_options[1:]:  # type: Option
			new_option.negative_examples += option.positive_examples

	def init_state_in_option(self, option):
		"""
		If the input `option` is part of a forward chain, we want to check if the start
		state of the MDP is in its initiation set. If so, we will not create a child
		option for the input `option`. If the input `option` is part of the backward
		chain, we want to check that ANY of the chain start states (usually existing
		salient events and optionally the start state of the MDP) are if in its
		initiation set classifier.
		Args:
			option (Option): We want to check if we want to create a child for this option

		Returns:
			any_init_in_option (bool)
		"""
		start_states = self.chains[option.chain_id - 1].init_salient_event.trigger_points
		if len(start_states) == 0:
			ipdb.set_trace()
		starts_chained = [option.is_init_true(state) for state in start_states]
		return all(starts_chained)

	def state_in_any_completed_option(self, state):
		"""
		Is the input `state` is inside the initiation classifier of an option whose initiation
		we are already done learning.

		Args:
			state (State)

		Returns:
			is_inside (bool)
		"""
		for option in self.trained_options[1:]:  # type: Option
			if option.get_training_phase() == "initiation_done" and option.initiation_classifier is not None:
				if option.is_init_true(state):
					return True
		return False

	def should_create_children_options(self, parent_option):

		if parent_option is None:
			return False

		return self.should_create_more_options() \
			   and parent_option.get_training_phase() == "initiation_done" \
			   and self.chains[parent_option.chain_id - 1].should_continue_chaining()

	def create_children_options(self, option):
		o1 = self.create_child_option(option)
		children = [o1]
		current_parent = option.parent
		while current_parent is not None:
			child_option = self.create_child_option(current_parent)
			children.append(child_option)
			current_parent = current_parent.parent
		return children

	def create_child_option(self, parent_option):

		# Don't create a child option if you already include the init state of the MDP
		# Also enforce the max branching factor of our skill tree

		# Added extra condition for chain_id b/c I got an intersection option which covered the start state,
		# but still want options to build in the other direction
		# if (self.init_state_in_option(parent_option) and parent_option.name != "intersection_option") \
		if self.init_state_in_option(parent_option)	\
				or len(parent_option.children) >= parent_option.max_num_children:
			return None

		# Create new option whose termination is the initiation of the option we just trained
		if "goal_option" in parent_option.name:
			prefix = f"option_{parent_option.name.split('_')[-1]}"
		else:
			prefix = parent_option.name
		name = prefix + "_{}".format(len(parent_option.children))
		print("Creating {} with parent {}".format(name, parent_option.name))

		chain = self.chains[parent_option.chain_id - 1]
		old_untrained_option_id = id(parent_option)
		new_untrained_option = Option(self.mdp, name=name, global_solver=self.global_option.solver,
									  lr_actor=parent_option.lr_actor,
									  lr_critic=parent_option.lr_critic,
									  ddpg_batch_size=parent_option.ddpg_batch_size,
									  subgoal_reward=self.subgoal_reward,
									  buffer_length=self.buffer_length,
									  classifier_type=self.classifier_type,
									  num_subgoal_hits_required=self.num_subgoal_hits_required,
									  seed=self.seed, parent=parent_option,  max_steps=self.max_steps,
									  enable_timeout=self.enable_option_timeout, chain_id=parent_option.chain_id,
									  writer=self.writer, device=self.device, dense_reward=self.dense_reward,
									  initialize_everywhere=parent_option.initialize_everywhere,
									  max_num_children=parent_option.max_num_children,
									  init_salient_event=chain.init_salient_event,
									  target_salient_event=chain.target_salient_event,
									  use_warmup_phase=self.use_warmup_phase,
									  update_global_solver=self.update_global_solver,
									  initiation_period=parent_option.initiation_period,
									  use_her=self.use_her_locally,
									  allow_her_initialization=self.allow_her_initialization,
									  use_hardcoded_init_sets=self.use_hardcoded_init_sets, 
									  use_hardcoded_goal_regions=self.use_hardcoded_goal_regions)

		new_untrained_option_id = id(new_untrained_option)
		assert new_untrained_option_id != old_untrained_option_id, "Checking python references"
		assert id(new_untrained_option.parent) == old_untrained_option_id, "Checking python references"

		parent_option.children.append(new_untrained_option)

		return new_untrained_option

	def make_off_policy_updates_for_options(self, state, action, reward, next_state):
		for option in self.trained_options: # type: Option
			option.off_policy_update(state, action, reward, next_state)\


	def make_smdp_update(self, *, action, next_state, option_transitions):
		"""
		Use Intra-Option Learning for sample efficient learning of the option-value function Q(s, o)
		Args:
			action (int): option taken by the global solver
			next_state (State): state we landed in after executing the option
			option_transitions (list): list of (s, a, r, s') tuples representing the trajectory during option execution
		"""

		def get_reward(transitions):
			gamma = self.global_option.solver.gamma
			raw_rewards = [tt[2] for tt in transitions]
			return sum([(gamma ** idx) * rr for idx, rr in enumerate(raw_rewards)])

		try:
			selected_option = self.trained_options[action]  # type: Option
		except:
			ipdb.set_trace()

		for i, transition in enumerate(option_transitions):
			start_state = transition[0]
			if selected_option.is_init_true(start_state):
				sub_transitions = option_transitions[i:]
				option_reward = get_reward(sub_transitions)
				self.agent_over_options.step(start_state.features(), action, option_reward, next_state.features(),
											 next_state.is_terminal(), num_steps=len(sub_transitions))

	def make_goal_conditioned_smdp_update(self, *, goal, final_state, option_trajectories, intra_option_learning=False):
		def get_goal_conditioned_reward(g, transitions):
			gamma = self.global_option.solver.gamma
			reward_func = self.mdp.dense_gc_reward_function if self.dense_reward else self.mdp.sparse_gc_reward_function
			raw_rewards = [reward_func(trans[-1], g, info={})[0] for trans in transitions]
			return sum([(gamma ** idx) * rr for idx, rr in enumerate(raw_rewards)])

		def perform_gc_experience_replay(g, o, transitions):
			if len(transitions) == 0: return

			reward_func = self.mdp.dense_gc_reward_function if self.dense_reward \
						  else self.mdp.sparse_gc_reward_function

			option_final_state = transitions[-1][-1]
			sp_augmented = np.concatenate((option_final_state.features(), g), axis=0)

			if intra_option_learning:
				for i, transition in enumerate(transitions):
					s0 = transition[0]
					s0_augmented = np.concatenate((s0.features(), g), axis=0)
					if o.is_init_true(s0):
						sub_transitions = transitions[i:]
						option_reward = get_goal_conditioned_reward(g, sub_transitions)
						done = reward_func(option_final_state, g, info={})[1]
						self.agent_over_options.step(s0_augmented, o.option_idx, option_reward, sp_augmented,
													 done=done, num_steps=len(sub_transitions))
			else:
				s0 = transitions[0][0]
				s0_augmented = np.concatenate((s0.features(), g), axis=0)
				option_reward = get_goal_conditioned_reward(g, transitions)
				done = reward_func(option_final_state, g, info={})[1]
				self.agent_over_options.step(s0_augmented, o.option_idx, option_reward, sp_augmented,
											 done=done, num_steps=len(transitions))

		goal_position = self.mdp.get_position(goal)
		reached_goal = self.mdp.get_position(final_state)

		for option, trajectory in option_trajectories:  # type: Option, list
			perform_gc_experience_replay(goal_position, option, transitions=trajectory)

		# Only perform the hindsight update if you didn't reach the goal
		if not self.mdp.sparse_gc_reward_function(reached_goal, goal_position, {})[1]:
			for option, trajectory in option_trajectories:  # type: Option, list
				perform_gc_experience_replay(reached_goal, option, transitions=trajectory)

	def get_init_q_value_for_new_option(self, newly_trained_option):
		global_solver = self.agent_over_options  # type: DQNAgent
		state_option_pairs = newly_trained_option.final_transitions
		q_values = []
		for state, option_idx in state_option_pairs:
			q_value = global_solver.get_qvalue(state.features(), option_idx)
			q_values.append(q_value)
		return np.max(q_values)

	def augment_agent_with_new_option(self, newly_trained_option, init_q_value):
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
			self.global_option.solver.trained_options.append(newly_trained_option)

		# Add the newly trained option to the corresponding skill chain
		if newly_trained_option.chain_id is not None:
			chain_idx = newly_trained_option.chain_id - 1
			self.chains[chain_idx].options.append(newly_trained_option)

		# Set the option_idx for the newly added option
		newly_trained_option.option_idx = len(self.trained_options) - 1
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
									exploration_strategy=self.agent_over_options.exploration_strategy)
		new_global_agent.replay_buffer = self.agent_over_options.replay_buffer

		init_q = self.get_init_q_value_for_new_option(newly_trained_option) if init_q_value is None else init_q_value
		print("Initializing new option node with q value {}".format(init_q))
		new_global_agent.policy_network.initialize_with_smaller_network(self.agent_over_options.policy_network, init_q)
		new_global_agent.target_network.initialize_with_smaller_network(self.agent_over_options.target_network, init_q)

		self.agent_over_options = new_global_agent

	def initialize_policy_over_options(self):
		""" Can be called after call to _augment_(). Useful if we want to reset the policy over options. """
		num_actions = len(self.trained_options)
		new_global_agent = DQNAgent(self.agent_over_options.state_size, num_actions, self.trained_options,
									seed=self.seed, name=self.agent_over_options.name,
									eps_start=self.agent_over_options.epsilon,
									tensor_log=self.agent_over_options.tensor_log,
									use_double_dqn=self.agent_over_options.use_ddqn,
									lr=self.agent_over_options.learning_rate,
									writer=self.writer, device=self.device,
									exploration_strategy=self.agent_over_options.exploration_strategy)
		self.agent_over_options = new_global_agent

	def trigger_option_termination_conditions_off_policy(self, untrained_option, executed_option, option_transitions, episode_number):
		"""
		While you were executing `selected_option`, it is possible that you triggered the
		termination condition of some other option in gestation. If so, we want to use that
		off-policy data to train that other option's first initiation classifier

		Args:
			untrained_option (Option)
			executed_option (Option)
			option_transitions (list)
			episode_number (int)

		Returns:

		"""
		if untrained_option != executed_option:
			if untrained_option.trigger_termination_condition_off_policy(option_transitions, episode_number) \
					and untrained_option.get_training_phase() == "initiation_done":
				print(f"Triggered final termination condition for {untrained_option} while executing {executed_option}")

	def make_off_policy_option_updates(self, executed_option, pre_rollout_state, option_transitions, k=3):
		if self.off_policy_update_type == "single":
			self.make_single_transition_off_policy_updates(executed_option, option_transitions, k=k)
		elif self.off_policy_update_type == "batched":
			self.make_batched_off_policy_option_updates(executed_option, pre_rollout_state, option_transitions, k=k)

	def make_batched_off_policy_option_updates(self, executed_option, pre_rollout_state, option_transitions, k=3):
		assert isinstance(executed_option, Option), f"{type(executed_option)}"
		assert not self.use_her_locally, "Not implemented off-policy option updates for HER yet"

		eligible = lambda o, s: o.is_init_true(s) and not o.is_term_true(s) and \
								o != executed_option and o.get_training_phase() == "initiation_done"

		options = [o for o in self.trained_options[1:] if eligible(o, pre_rollout_state)]
		td_errors = np.array([o.get_mean_td_error(option_transitions) for o in options])
		sorted_option_idx = np.argsort(-td_errors)[:k]
		selected_options = [options[idx] for idx in sorted_option_idx]
		for option in selected_options:  # type: Option
			print(f"Performing batched off-policy update for {option} at {pre_rollout_state.position}")
			option.local_option_experience_replay(option_transitions, goal_state=None)

	def make_single_transition_off_policy_updates(self, executed_option, option_transitions, k=3):
		assert isinstance(executed_option, Option), f"{type(executed_option)}"
		assert not self.use_her_locally, "Not implemented off-policy option updates for HER yet"

		def _numpify_transition(s, a, r, sp, done):
			return s.features(), a, r, sp.features(), done

		eligible = lambda o, s: o.is_init_true(s) and not o.is_term_true(s) and \
								o != executed_option and o.get_training_phase() == "initiation_done"

		for transition in option_transitions:
			options = [o for o in self.trained_options[1:] if eligible(o, transition[0])]
			td_errors = []
			for option in options:
				relabeled_transition = option.relabel_transitions_with_option_reward_function([transition])[0]
				relabeled_transition_np = _numpify_transition(*relabeled_transition)
				td_error = option.solver.get_td_error(*relabeled_transition_np)
				td_errors.append(td_error.item())
			td_errors = np.array(td_errors)
			sorted_option_idx = np.argsort(-td_errors)[:k]
			selected_options = [options[idx] for idx in sorted_option_idx]
			for option in selected_options:  # type: Option
				option.update_option_solver(*transition)

	def get_augmented_state(self, state, goal):
		if goal is not None and self.use_her:
			return np.concatenate((state.features(), goal))
		return state.features()

	def act(self, state, goal=None, train_mode=True):

		# Query the global Q-function to determine which option to take in the current state
		augmented_state = self.get_augmented_state(state, goal)
		option_idx = self.agent_over_options.act(augmented_state, train_mode=train_mode)

		if train_mode:
			self.agent_over_options.update_epsilon()

		# Selected option
		selected_option = self.trained_options[option_idx]  # type: Option

		return selected_option

	def update_policy_over_options_after_option_rollout(self, state_before_rollout, executed_option, option_transitions):
		"""

		Args:
			state_before_rollout (State)
			executed_option (Option)
			option_transitions (list)

		Returns:

		"""

		state_after_rollout = deepcopy(self.mdp.cur_state)

		# If we triggered the untrained option's termination condition, add to its buffer of terminal transitions
		for untrained_option in self.untrained_options:
			if untrained_option.is_term_true(state_after_rollout) and not untrained_option.is_term_true(state_before_rollout):
				untrained_option.final_transitions.append((state_before_rollout, executed_option.option_idx))
				untrained_option.terminal_states.append(state_after_rollout)

		# Add data to train Q(s, o)
		if not self.use_her:
			self.make_smdp_update(action=executed_option.option_idx,
								  next_state=state_after_rollout,
								  option_transitions=option_transitions)

	def sample_qvalue(self, option):
		""" Sample Q-value from the given option's intra-option policy. """
		if len(option.solver.replay_buffer) > 500:
			sample_experiences = option.solver.replay_buffer.sample(batch_size=500)
			sample_states = torch.from_numpy(sample_experiences[0]).float().to(self.device)
			sample_actions = torch.from_numpy(sample_experiences[1]).float().to(self.device)
			sample_qvalues = option.solver.get_qvalues(sample_states, sample_actions)
			return sample_qvalues.mean().item()
		return 0.0

	@staticmethod
	def get_reward_from_experiences(experiences):
		total_reward = 0.
		for experience in experiences:
			reward = experience[2]
			total_reward += reward
		return total_reward

	def should_create_more_options(self):
		""" Continue chaining as long as any chain is still accepting new options. """
		return any([chain.should_continue_chaining() for chain in self.chains])

	def finish_option_initiation_phase(self, untrained_option, episode):
		"""

		Args:
			untrained_option (Option)
			episode (int)

		Returns:
			completed_option (Option)
		"""
		if untrained_option.get_training_phase() == "initiation_done":
			# Debug visualization
			plot_two_class_classifier(untrained_option, episode, self.experiment_name)

			# We fix the learned option's initiation set and remove it from the list of target events
			self.untrained_options.remove(untrained_option)

			return untrained_option

		return None

	def add_new_options_to_skill_chains(self, new_options):
		# Iterate through all the child options and add them to the skill tree 1 by 1
		for new_option in new_options:  # type: Option
			if new_option is not None:
				self.untrained_options.append(new_option)

				# If initialize_everywhere is True, also add to trained_options
				if new_option.initialize_everywhere:
					if self.pretrain_option_policies:
						new_option.initialize_with_global_solver()
					self.augment_agent_with_new_option(new_option, self.init_q)

	def _get_current_salient_events(self, state):
		""" Get all the salient events satisfied by the current `state`. """
		events = self.mdp.get_all_target_events_ever() + [self.mdp.get_start_state_salient_event()]
		current_events = [event for event in events if event(state)]
		return current_events

	def disable_triggering_options_targeting_init_event(self, state):
		""" Disable off-policy triggers for all options that target the current salient events. """
		current_salient_events = self._get_current_salient_events(state)

		for current_salient_event in current_salient_events:
			for option in self.trained_options:  # type: Option
				if option.chain_id is not None:
					option.started_at_target_salient_event = option.target_salient_event == current_salient_event

		# Additionally, disable off-policy triggering options whose termination set we are in
		for option in self.trained_options[1:]:
			option.started_at_target_salient_event = option.is_term_true(state)

	def manage_skill_chain_after_option_rollout(self, *, state_before_rollout, executed_option,
												created_options, episode_number, option_transitions):
		""" After a single option rollout, this method will perform the book-keeping necessary to maintain skill-chain. """

		# Update Q(s, o) for the `agent_over_options`
		self.update_policy_over_options_after_option_rollout(state_before_rollout=state_before_rollout,
															 executed_option=executed_option,
															 option_transitions=option_transitions)

		# Off-policy policy updates for other options
		self.make_off_policy_option_updates(executed_option=executed_option,
											pre_rollout_state=state_before_rollout,
											option_transitions=option_transitions)

		# In the skill-tree setting we could be learning many options at the current time
		# We must iterate through all such options and check if the current transition
		# triggered the termination condition of any such option
		for untrained_option in self.untrained_options:

			self.trigger_option_termination_conditions_off_policy(untrained_option=untrained_option,
																  executed_option=executed_option,
																  option_transitions=option_transitions,
																  episode_number=episode_number)

			completed_option = self.finish_option_initiation_phase(untrained_option=untrained_option,
																   episode=episode_number)

			should_create_children = self.should_create_children_options(completed_option)

			if should_create_children:
				new_options = self.create_children_options(completed_option)
				self.add_new_options_to_skill_chains(new_options)

			if completed_option is not None:
				created_options.append(completed_option)

		return created_options

	def perform_experience_replay(self, state, episodic_trajectory, option_trajectory, overall_goal=None):
		"""
		Perform experience replay for the global-option and the policy over options.

		Args:
			state (State): state at the end of the trajectory
			episodic_trajectory (list): sequence of (s, a, r, s') tuples
			option_trajectory (list): sequence of (option, trajectory) tuples
			overall_goal (np.ndarray): Goal DSC was pursuing for that episode

		"""
		# Experience replay for the global-option
		reached_goal = self.mdp.get_position(state.position)
		overall_goal = self.mdp.get_position(self.mdp.goal_state) if overall_goal is None else overall_goal
		episodic_trajectory = list(itertools.chain.from_iterable(episodic_trajectory))

		self.global_option_experience_replay(episodic_trajectory, goal_state=overall_goal)

		if self.use_her:
			reached_overall_goal = self.mdp.sparse_gc_reward_function(reached_goal, overall_goal, {})[1]
			if len(episodic_trajectory) > 0 and not reached_overall_goal:
				self.global_option_experience_replay(episodic_trajectory, goal_state=reached_goal)
			if len(option_trajectory) > 1:
				self.make_goal_conditioned_smdp_update(goal=overall_goal, final_state=state, option_trajectories=option_trajectory)

	def dsc_rollout(self, *, episode_number, step_number, interrupt_handle=lambda x: False, overall_goal=None):
		""" Single episode rollout of deep skill chaining from the current state in the MDP. """

		score = 0.

		state = deepcopy(self.mdp.cur_state)
		created_options = []
		episodic_trajectory = []
		option_trajectory = []

		goal = self.mdp.get_position(self.mdp.goal_state) if overall_goal is None else overall_goal

		while step_number < self.max_steps:

			# Pick an option
			selected_option = self.act(state=state, goal=goal, train_mode=True)

			# Disable options whose termination sets we are already in
			self.disable_triggering_options_targeting_init_event(state)

			# Execute it in the MDP
			option_transitions, reward = selected_option.execute_option_in_mdp(self.mdp,
																			   episode_number,
																			   step_number,
																			   poo_goal=goal,
																			   goal_salient_event=interrupt_handle)

			# Logging
			score += reward
			step_number += len(option_transitions)
			state = deepcopy(self.mdp.cur_state)
			episodic_trajectory.append(option_transitions)
			option_trajectory.append((selected_option, option_transitions))

			# DSC book keeping
			created_options = self.manage_skill_chain_after_option_rollout(state_before_rollout=state,
																		   executed_option=selected_option,
																		   option_transitions=option_transitions,
																		   episode_number=episode_number,
																		   created_options=created_options)

			if state.is_terminal() or interrupt_handle(state):
				break

		self.perform_experience_replay(state, episodic_trajectory, option_trajectory, overall_goal=overall_goal)

		return score, step_number, created_options

	def skill_chaining_run_loop(self, *, num_episodes, num_steps, to_reset, starting_episode=0, interrupt_handle=lambda x: False):

		# For logging purposes
		per_episode_scores = []
		per_episode_durations = []
		last_10_scores = deque(maxlen=10)
		last_10_durations = deque(maxlen=10)

		for episode in range(starting_episode, starting_episode + num_episodes):

			step_number = 0
			if to_reset: self.mdp.reset()

			score, step_number, newly_created_options = self.dsc_rollout(episode_number=episode,
																		 interrupt_handle=interrupt_handle,
																		 step_number=step_number)

			last_10_scores.append(score)
			last_10_durations.append(step_number)
			per_episode_scores.append(score)
			per_episode_durations.append(step_number)

			self._log_dqn_status(episode, last_10_scores, last_10_durations)

		return per_episode_scores, per_episode_durations

	def global_option_experience_replay(self, trajectory, goal_state):
		if self.global_option.solver_type == "mpc":
			return
		for state, action, _, next_state in trajectory:
			augmented_state = self.get_augmented_state(state, goal=goal_state)
			augmented_next_state = self.get_augmented_state(next_state, goal=goal_state)
			reward_func = self.mdp.dense_gc_reward_function if self.dense_reward else self.mdp.sparse_gc_reward_function
			reward, done = reward_func(next_state, goal_state, info={})
			self.global_option.solver.step(augmented_state, action, reward, augmented_next_state, done)

	def _log_dqn_status(self, episode, last_10_scores, last_10_durations):

		print("=" * 80)
		print('\rEpisode {}\tAverage Score: {:.2f}\tDuration: {:.2f} steps\tOP Eps: {:.2f}'.format(
			episode, np.mean(last_10_scores), np.mean(last_10_durations), self.agent_over_options.epsilon))
		print("=" * 80)

		self.num_options_history.append(len(self.trained_options))

		if self.writer is not None:
			self.writer.add_scalar("Episodic scores", last_10_scores[-1], episode)

		if episode > 0 and episode % 100 == 0:
			# eval_score, trajectory = self.trained_forward_pass(render=False)
			eval_score, trajectory = 0., []

			self.validation_scores.append(eval_score)
			print("\rEpisode {}\tValidation Score: {:.2f}".format(episode, eval_score))

		if self.generate_plots and episode % 10 == 0 and episode > 0:

			visualize_best_option_to_take(self.agent_over_options, episode, self.seed, self.experiment_name)

			for option in self.trained_options:
				make_chunked_value_function_plot(option.solver, episode, self.seed, self.experiment_name)
				visualize_ddpg_shaped_rewards(self.global_option, option, episode, self.seed, self.experiment_name)
				visualize_dqn_shaped_rewards(self.agent_over_options, option, episode, self.seed, self.experiment_name)

	def save_all_models(self):
		for option in self.trained_options: # type: Option
			save_model(option.solver, -1, best=False)

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
			visualize_dqn_replay_buffer(option.solver, self.experiment_name)

		for i, o in enumerate(self.trained_options):
			plt.subplot(1, len(self.trained_options), i + 1)
			plt.plot(self.option_qvalues[o.name])
			plt.title(o.name)
		plt.savefig("value_function_plots/{}/sampled_q_so_{}.png".format(self.experiment_name, self.seed))
		plt.close()

		for option in self.trained_options:
			visualize_next_state_reward_heat_map(option.solver, -1, self.experiment_name)

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

	def __getstate__(self):
		excluded_keys = ("mdp", "writer", "plotter")
		state_dictionary = {x: self.__dict__[x] for x in self.__dict__ if x not in excluded_keys}
		return state_dictionary

	def __setstate__(self, state_dictionary):
		excluded_keys = ("mdp", "writer", "plotter")
		for key in state_dictionary:
			if key not in excluded_keys:
				self.__dict__[key] = state_dictionary[key]


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
	parser.add_argument("--use_start_state_salience", action="store_true", default=False)
	parser.add_argument("--use_option_intersection_salience", action="store_true", default=False)
	parser.add_argument("--use_event_intersection_salience", action="store_true", default=False)
	parser.add_argument("--pretrain_option_policies", action="store_true", default=False)
	parser.add_argument("--create_backward_options", action="store_true", default=False)
	parser.add_argument("--use_her", action="store_true", help="Hindsight Experience Replay", default=False)
	parser.add_argument("--use_her_locally", action="store_true", help="HER for local options", default=False)
	parser.add_argument("--use_warmup_phase", action="store_true", default=False)

	args = parser.parse_args()

	if args.env == "point-reacher":
		from simple_rl.tasks.point_reacher.PointReacherMDPClass import PointReacherMDP
		overall_mdp = PointReacherMDP(seed=args.seed, dense_reward=args.dense_reward, render=args.render)
		state_dim = 6
		action_dim = 2
	elif args.env == "ant-reacher":
		from simple_rl.tasks.ant_reacher.AntReacherMDPClass import AntReacherMDP
		overall_mdp = AntReacherMDP(seed=args.seed, render=args.render, goal_state=np.array((8, 8)), task_agnostic=False)
		state_dim = overall_mdp.state_space_size()
		action_dim = overall_mdp.action_space_size()
	elif args.env == "d4rl-point-maze-easy":
		from simple_rl.tasks.d4rl_point_maze.D4RLPointMazeMDPClass import D4RLPointMazeMDP
		overall_mdp = D4RLPointMazeMDP(difficulty="easy", seed=args.seed, render=args.render, goal_directed=True)
		state_dim = 6
		action_dim = 2
	elif args.env == "d4rl-ant-maze":
		from simple_rl.tasks.d4rl_ant_maze.D4RLAntMazeMDPClass import D4RLAntMazeMDP
		overall_mdp = D4RLAntMazeMDP(maze_size="umaze", seed=args.seed,
									 render=args.render, goal_state=np.array((0., 8.)))
		state_dim = overall_mdp.state_space_size()
		action_dim = overall_mdp.action_space_size()
	elif "reacher" in args.env.lower():
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
							generate_plots=args.generate_plots, tensor_log=args.tensor_log, device=args.device,
							buffer_length=args.buffer_len,
							start_state_salience=args.use_start_state_salience,
							option_intersection_salience=args.use_option_intersection_salience,
							event_intersection_salience=args.use_event_intersection_salience,
							pretrain_option_policies=args.pretrain_option_policies,
							create_backward_options=args.create_backward_options,
							dense_reward=args.dense_reward,
							use_her=args.use_her,
							use_her_locally=args.use_her_locally,
							experiment_name=args.experiment_name, 
							use_warmup_phase=args.use_warmup_phase,
							use_hardcoded_init_sets=True, 
							use_hardcoded_goal_regions=True)
	episodic_scores, episodic_durations = chainer.skill_chaining_run_loop(num_episodes=args.episodes, num_steps=args.steps, to_reset=True)

	# Log performance metrics
	# chainer.save_all_models()
	# chainer.perform_experiments()
	chainer.save_all_scores(args.pretrained, episodic_scores, episodic_durations)
