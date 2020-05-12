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
import pandas as pd
from pathlib import Path
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.cm as cm
import matplotlib.patheffects as pe
import matplotlib.colors as mc

# Other imports.
from simple_rl.mdp.StateClass import State
from simple_rl.agents.func_approx.dsc.OptionClass import Option
from simple_rl.agents.func_approx.dsc.utils import *
from simple_rl.agents.func_approx.ddpg.utils import *
from simple_rl.agents.func_approx.dqn.DQNAgentClass import DQNAgent


class SkillChaining(object):
	def __init__(self, mdp, max_steps, lr_actor, lr_critic, lr_dqn, ddpg_batch_size, device, max_num_options=np.inf,
				 subgoal_reward=0., enable_option_timeout=True, buffer_length=20, num_subgoal_hits_required=3,
				 classifier_type="ocsvm", init_q=None, generate_plots=False, episodic_plots=False, use_full_smdp_update=False,
				 log_dir="", seed=0, tensor_log=False, opt_nu=0.5, pes_nu=0.5, experiment_name=None, num_run=0, discrete_actions=False,
				 use_old=False, episodic_saves=False, args=None, use_chain_fix=False):
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
			episodic_plots (bool): whether or not to produce plots for each episode 
			use_full_smdp_update (bool): sparse 0/1 reward or discounted SMDP reward for training policy over options
			log_dir (os.path): directory to store all the scores for this run
			seed (int): We are going to use the same random seed for all the DQN solvers
			tensor_log (bool): Tensorboard logging enable
			opt_nu (float): Nu parameter for optimitisic one-class classifier
			pes_nu (float): Nu parameter for pessimistic one-class classifier
			num_run (int): Number of the current run.
			discrete_actions (bool): Whether or not actions are discrete
			use_old (bool): Whether or not to use previous DSC methods
			episodic_saves (bool): Whether to save all data per episode
			args (argparse.ArgumentParser().parse_args())
			use_chain_fix (bool)
)
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

		# TODO: extra
		self.lr_actor = lr_actor
		self.lr_critic = lr_critic
		self.ddpg_batch_size = ddpg_batch_size
		self.opt_nu = opt_nu
		self.pes_nu = pes_nu
		self.experiment_name = experiment_name
		self.num_run = num_run
		self.discrete_actions = discrete_actions
		self.lr_dqn = lr_dqn
		self.use_old = use_old
		self.episodic_saves = episodic_saves
		self.args = args
		self.use_chain_fix = use_chain_fix
		self.episode = 0

		# TODO: changed log dir
		tensor_name = "logs/{}_{}".format(args.experiment_name, seed)
		self.writer = SummaryWriter(tensor_name) if tensor_log else None

		print("Initializing skill chaining with option_timeout={}, seed={}\n".format(self.enable_option_timeout, seed))

		random.seed(seed)
		np.random.seed(seed)

		self.validation_scores = []

		# This option has an initiation set that is true everywhere and is allowed to operate on atomic timescale only
		if self.discrete_actions:	# TODO discrete solver toggle
			self.global_option = Option(overall_mdp=self.mdp, name="global_option", global_solver=None,
										lr_actor=None, lr_critic=None, lr_dqn=self.lr_dqn, buffer_length=buffer_length,
										ddpg_batch_size=self.ddpg_batch_size, num_subgoal_hits_required=num_subgoal_hits_required,
										subgoal_reward=self.subgoal_reward, seed=self.seed, max_steps=self.max_steps,
										enable_timeout=self.enable_option_timeout, classifier_type=classifier_type,
										generate_plots=self.generate_plots, writer=self.writer, device=self.device,
										dense_reward=self.dense_reward, opt_nu=self.opt_nu, pes_nu=self.pes_nu, experiment_name=self.experiment_name,
										discrete_actions=self.discrete_actions, use_old=self.use_old,
										episode=self.episode)
		else:
			self.global_option = Option(overall_mdp=self.mdp, name="global_option", global_solver=None,
										lr_actor=self.lr_actor, lr_critic=self.lr_critic, lr_dqn=None, buffer_length=buffer_length,
										ddpg_batch_size=self.ddpg_batch_size, num_subgoal_hits_required=num_subgoal_hits_required,
										subgoal_reward=self.subgoal_reward, seed=self.seed, max_steps=self.max_steps,
										enable_timeout=self.enable_option_timeout, classifier_type=classifier_type,
										generate_plots=self.generate_plots, writer=self.writer, device=self.device,
										dense_reward=self.dense_reward, opt_nu=self.opt_nu, pes_nu=self.pes_nu, experiment_name=self.experiment_name,
										discrete_actions=self.discrete_actions, use_old=self.use_old,
										episode=self.episode)

		self.trained_options = [self.global_option]

		# This is our first untrained option - one that gets us to the goal state from nearby the goal
		# We pick this option when self.agent_over_options thinks that we should
		# Once we pick this option, we will use its internal DDPG solver to take primitive actions until termination
		# Once we hit its termination condition N times, we will start learning its initiation set
		# Once we have learned its initiation set, we will create its child option
		# TODO: changed how options are labeled due to chain fix options
		self.num_options = option_idx = 1
		name = 'option_{}'.format(option_idx)
		if self.discrete_actions:	# TODO: discrete solver toggle
			goal_option = Option(overall_mdp=self.mdp, name=name, global_solver=self.global_option.solver,
								lr_actor=None, lr_critic=None, lr_dqn=self.lr_dqn, buffer_length=self.buffer_length,
								ddpg_batch_size=ddpg_batch_size, num_subgoal_hits_required=self.num_subgoal_hits_required,
								subgoal_reward=self.subgoal_reward, seed=self.seed, max_steps=self.max_steps,
								enable_timeout=self.enable_option_timeout, classifier_type=self.classifier_type,
								generate_plots=self.generate_plots, writer=self.writer, device=self.device,
								dense_reward=self.dense_reward, opt_nu=self.opt_nu, pes_nu=self.pes_nu, experiment_name=self.experiment_name,
								discrete_actions=self.discrete_actions, use_old=self.use_old,
								episode=self.episode, option_idx=option_idx)
		else:
			goal_option = Option(overall_mdp=self.mdp, name=name, global_solver=self.global_option.solver,
								lr_actor=self.lr_actor, lr_critic=self.lr_critic, lr_dqn=None, buffer_length=self.buffer_length,
								ddpg_batch_size=ddpg_batch_size, num_subgoal_hits_required=self.num_subgoal_hits_required,
								subgoal_reward=self.subgoal_reward, seed=self.seed, max_steps=self.max_steps,
								enable_timeout=self.enable_option_timeout, classifier_type=self.classifier_type,
								generate_plots=self.generate_plots, writer=self.writer, device=self.device,
								dense_reward=self.dense_reward, opt_nu=self.opt_nu, pes_nu=self.pes_nu, experiment_name=self.experiment_name,
								discrete_actions=self.discrete_actions, use_old=self.use_old,
								episode=self.episode, option_idx=option_idx)

		# This is our policy over options
		# We use (double-deep) (intra-option) Q-learning to learn the Q-values of *options* at any queried state Q(s, o)
		# We start with this DQN Agent only predicting Q-values for taking the global_option, but as we learn new
		# options, this agent will predict Q-values for them as well
		self.agent_over_options = DQNAgent(self.mdp.state_space_size(), 1, trained_options=self.trained_options,
										   seed=seed, lr=1e-4, name="GlobalDQN", eps_start=1.0, tensor_log=tensor_log,
										   use_double_dqn=True, writer=self.writer, device=self.device)

		# Pointer to the current option:
		# 1. This option has the termination set which defines our current goal trigger
		# 2. This option has an untrained initialization set and policy, which we need to train from experience
		self.untrained_option = goal_option

		# TODO: don't need a list of initial states, just take it from the MDP directly
		# List of init states seen while running this algorithm
		# self.init_states = []
		self.start_state = np.array(self.mdp.init_state) 

		# Debug variables
		self.global_execution_states = []
		self.num_option_executions = defaultdict(lambda : [])
		self.option_rewards = defaultdict(lambda : [])
		self.option_qvalues = defaultdict(lambda : [])
		self.num_options_history = []
		self.temporal_chain_breaks = {}
		self.option_chain_breaks = {}
		self.goal_chain_breaks = {}
		self.final_skill_chain = None
		self.full_chain_breaks = {}
		self.temporal_full_chain_breaks = {}
		self.full_chain_fix_option = []

		# TODO: plotting/logging variables
		self.episodic_plots = episodic_plots
		self.x_mesh = None
		self.y_mesh = None
		self.x_mesh_hd = None
		self.y_mesh_hd = None
		self.option_data = {}
		self.goal_xy = np.array(self.mdp.goal_position)
		self.start_xy = np.array(self.start_state[:2])
		self.options_chain_breaks = {}

		# TODO: add environment image for plotting
		if "maze" in self.mdp.env_name:
			self.img_name = "images/point_maze_domain.png"
		elif "treasure" in self.mdp.env_name:
			self.img_name = "images/treasure_game_domain.png"
		else:
			self.img_name = None

	# TODO restructered for new naming convention
	def create_option(self, parent_option, option_idx, type=''):
		# Create new option whose termination is the initiation of the option we just trained
		name = "option_{}".format(option_idx)
		print("{}Creating {}".format(type, name))

		if parent_option is None:
			if self.discrete_actions:	# TODO: discrete solver toggle
				new_untrained_option = Option(overall_mdp=self.mdp, name=name, global_solver=self.global_option.solver,
											lr_actor=None, lr_critic=None, lr_dqn=self.lr_dqn, buffer_length=self.buffer_length,
											ddpg_batch_size=self.ddpg_batch_size, num_subgoal_hits_required=self.num_subgoal_hits_required,
											subgoal_reward=self.subgoal_reward, seed=self.seed, max_steps=self.max_steps,
											enable_timeout=self.enable_option_timeout, classifier_type=self.classifier_type,
											generate_plots=self.generate_plots, writer=self.writer, device=self.device,
											dense_reward=self.dense_reward, opt_nu=self.opt_nu, pes_nu=self.pes_nu, experiment_name=self.experiment_name,
											discrete_actions=self.discrete_actions, use_old=self.use_old,
											episode=self.episode, option_idx=option_idx)
			else:
				new_untrained_option = Option(overall_mdp=self.mdp, name=name, global_solver=self.global_option.solver,
											lr_actor=self.lr_actor, lr_critic=self.lr_critic, lr_dqn=None, buffer_length=self.buffer_length,
											ddpg_batch_size=self.ddpg_batch_size, num_subgoal_hits_required=self.num_subgoal_hits_required,
											subgoal_reward=self.subgoal_reward, seed=self.seed, max_steps=self.max_steps,
											enable_timeout=self.enable_option_timeout, classifier_type=self.classifier_type,
											generate_plots=self.generate_plots, writer=self.writer, device=self.device,
											dense_reward=self.dense_reward, opt_nu=self.opt_nu, pes_nu=self.pes_nu, experiment_name=self.experiment_name,
											discrete_actions=self.discrete_actions, use_old=self.use_old,
											episode=self.episode, option_idx=option_idx)
		else:
			if self.discrete_actions:	# TODO: discrete solver toggle
				new_untrained_option = Option(self.mdp, name=name, global_solver=self.global_option.solver,
										lr_actor=None,
										lr_critic=None,
										lr_dqn=parent_option.solver.learning_rate,
										ddpg_batch_size=parent_option.solver.batch_size,
										subgoal_reward=self.subgoal_reward,
										buffer_length=self.buffer_length,
										classifier_type=self.classifier_type,
										num_subgoal_hits_required=self.num_subgoal_hits_required,
										seed=self.seed, parent=parent_option,  max_steps=self.max_steps,
										enable_timeout=self.enable_option_timeout,
										writer=self.writer, device=self.device, dense_reward=self.dense_reward, opt_nu=self.opt_nu, pes_nu=self.pes_nu, experiment_name=self.experiment_name,
										discrete_actions=self.discrete_actions, use_old=self.use_old,
										episode=self.episode, option_idx=option_idx)
			else:
				new_untrained_option = Option(self.mdp, name=name, global_solver=self.global_option.solver,
											lr_actor=parent_option.solver.actor_learning_rate,
											lr_critic=parent_option.solver.critic_learning_rate,
											lr_dqn=None,
											ddpg_batch_size=parent_option.solver.batch_size,
											subgoal_reward=self.subgoal_reward,
											buffer_length=self.buffer_length,
											classifier_type=self.classifier_type,
											num_subgoal_hits_required=self.num_subgoal_hits_required,
											seed=self.seed, parent=parent_option,  max_steps=self.max_steps,
											enable_timeout=self.enable_option_timeout,
											writer=self.writer, device=self.device, dense_reward=self.dense_reward, opt_nu=self.opt_nu, pes_nu=self.pes_nu, experiment_name=self.experiment_name,
											discrete_actions=self.discrete_actions, use_old=self.use_old,
											episode=self.episode, option_idx=option_idx)
		
		old_untrained_option_id = id(parent_option)
		new_untrained_option_id = id(new_untrained_option)
		assert new_untrained_option_id != old_untrained_option_id, "Checking python references"
		assert id(new_untrained_option.parent) == old_untrained_option_id, "Checking python references"

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
		# assert self.subgoal_reward == 0, "This kind of SMDP update only makes sense when subgoal reward is 0"

		def get_reward(transitions):
			gamma = self.global_option.solver.gamma
			raw_rewards = [tt[2] for tt in transitions]
			return sum([(gamma ** idx) * rr for idx, rr in enumerate(raw_rewards)])

		# NOTE: Should we do intra-option learning only when the option was successful in reaching its subgoal?
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
		Initialize a new one option to target with the trained options.
		Add the newly_trained_option as a new node to the Q-function over options
		Args:
			newly_trained_option (Option)
			init_q_value (float): if given use this, else compute init_q optimistically
		"""
		# Add the trained option to the action set of the global solver
		if newly_trained_option not in self.trained_options:
			self.trained_options.append(newly_trained_option)

		# Augment the global DQN with the newly trained option
		num_actions = len(self.trained_options)
		new_global_agent = DQNAgent(self.agent_over_options.state_size, num_actions, self.trained_options,
									seed=self.seed, name=self.agent_over_options.name,
									eps_start=self.agent_over_options.epsilon,
									tensor_log=self.agent_over_options.tensor_log,
									use_double_dqn=self.agent_over_options.use_ddqn,
									lr=self.agent_over_options.learning_rate,
									writer=self.writer, device=self.device)
		new_global_agent.replay_buffer = self.agent_over_options.replay_buffer

		init_q = self.get_init_q_value_for_new_option(newly_trained_option) if init_q_value is None else init_q_value
		print("|-> Initializing new option node with q value {}".format(init_q))
		new_global_agent.policy_network.initialize_with_smaller_network(self.agent_over_options.policy_network, init_q)
		new_global_agent.target_network.initialize_with_smaller_network(self.agent_over_options.target_network, init_q)

		self.agent_over_options = new_global_agent

	def act(self, state):
		# Query the global Q-function to determine which option to take in the current state
		option_idx = self.agent_over_options.act(state.features(), train_mode=True)
		self.agent_over_options.update_epsilon()

		# Selected option
		selected_option = self.trained_options[option_idx]  # type: Option

		# Debug: If it was possible to take an option, did we take it?
		for option in self.trained_options:  # type: Option
			if option.is_init_true(state):
				option_taken = option.option_idx == selected_option.option_idx
				if option.writer is not None:
					option.writer.add_scalar("{}_taken".format(option.name), option_taken, option.n_taken_or_not)
					option.taken_or_not.append(option_taken)
					option.n_taken_or_not += 1

		return selected_option

	def take_action(self, state, step_number, episode_option_executions, episode=None):
		"""
		Either take a primitive action from `state` or execute a closed-loop option policy.
		Args:
			state (State)
			step_number (int): which iteration of the control loop we are on
			episode_option_executions (defaultdict)

		Returns:
			experiences (list): list of (s, a, r, s') tuples
			reward (float): sum of all rewards accumulated while executing chosen action
			next_state (State): state we landed in after executing chosen action
		"""
		selected_option = self.act(state)
		option_transitions, discounted_reward = selected_option.execute_option_in_mdp(
			self.mdp, step_number, episode)
		
		option_reward = self.get_reward_from_experiences(option_transitions)
		next_state = self.get_next_state_from_experiences(option_transitions)

		# If we triggered the untrained option's termination condition, add to its buffer of terminal transitions
		if self.untrained_option.is_term_true(next_state) and not self.untrained_option.is_term_true(state):
			self.untrained_option.final_transitions.append((state, selected_option.option_idx))

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
		if self.discrete_actions:	# TODO: discrete solver toggle
			# NOTE: different replay buffers for DQN
			if len(option.solver.replay_buffer) > 500:
				sample_experiences = option.solver.replay_buffer.sample(batch_size=500)
				sample_states = sample_experiences[0]
				sample_actions = sample_experiences[1]
				sample_qvalues = option.solver.get_qvalues(sample_states)
				return sample_qvalues.mean().item()
			return 0.0
		else:
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

	# TODO: utilities
	def get_all_global_states(self):
		return np.array([row[0] for row in list(self.global_option.solver.replay_buffer.memory)])

	# TODO: utilities
	def get_mesh_positions(self):
		return np.c_[self.x_mesh.ravel(), self.y_mesh.ravel()]

	# TODO: log chain breaks
	def log_full_chain_breaks(self, episode):
		# Time index with steps and episodes (assume can be 0)
		time_step = episode
		
		full_broken_options = self.check_full_chain_breaks()
		
		if full_broken_options == []:
			self.temporal_full_chain_breaks[time_step] = 0

		for full_broken_option in full_broken_options:
			# Update total temporal full chain breaks
			if time_step not in self.temporal_full_chain_breaks:
				self.temporal_full_chain_breaks[time_step] = 1
			else:
				self.temporal_full_chain_breaks[time_step] += 1

			# Update the options that broke the goal
			if full_broken_option.name not in self.full_chain_breaks:
				self.full_chain_breaks[full_broken_option.name] = {time_step : 1}
			else:
				if time_step not in self.full_chain_breaks[full_broken_option.name]:
					self.full_chain_breaks[full_broken_option.name][time_step] = 1
				else:
					self.full_chain_breaks[full_broken_option.name][time_step] += 1

	# TODO: detect chain breaks for logging
	def check_full_chain_breaks(self):
		local_options = self.trained_options[1:]
		full_chain_breaks = []
		for option in local_options:
			# NOTE: gestation will only happen with chain fix code
			if option.parent is not None and option.parent.get_training_phase() != 'gestation':
				# TODO: hack for treasure game
				if "treasure" in self.mdp.env_name:
					raise NotImplementedError
				else:
					states = option.parent.X

				current_pred = np.array(option.batched_is_init_true(states), dtype=int)
				parent_pred = np.array(option.parent.batched_is_init_true(states), dtype=int)
				
				if not 1 in (current_pred * parent_pred): 
					full_chain_breaks.append(option)
		return full_chain_breaks

	# TODO: detect chain breaks in main protocol
	def should_create_chain_fix_option(self):
		local_options = self.trained_options[1:]
		for option in local_options:
			if option.parent is not None and option.parent.get_training_phase() != 'gestation':
				# TODO: hack for treasure game
				if "treasure" in self.mdp.env_name:
					raise NotImplementedError
				else:
					states = option.parent.X
				
				current_pred = np.array(option.batched_is_init_true(states), dtype=int)
				parent_pred = np.array(option.parent.batched_is_init_true(states), dtype=int)
				
				if not 1 in (current_pred * parent_pred):
					print("    |-> FULL CHAIN BREAK: (parent) {}, (option) {}".format(option.parent.name, option.name))
					return True, option
		return False, None

	# TODO: main chain fix protocol
	def detect_and_fix_chain(self, episode, step_number):
		# Then detect if intermediate option needs to be fixed
		option_needs_fix, broken_option = self.should_create_chain_fix_option()
		if option_needs_fix:
			self.full_chain_fix_option.append(episode)

			# Create fix for intermediate option
			self.num_options += 1
			new_idx = self.num_options
			option_fix = self.create_option(broken_option.parent, new_idx, type='(option fix) ')

			# Link broken option to new chain fix option
			self.update_options_parent(broken_option, option_fix)
			return True, option_fix

		# Otherwise, no options need to be fixed
		return False, None

	# TODO: utilities
	def get_skill_chain(self):
		# NOTE: this only produces a skill chain of learned options (excludes global option)
		# Build parent to child option dictionary
		all_options = {}
		for option in self.trained_options[1:]:
			if option.parent is None:
				all_options['None'] = option
			else:
				all_options[option.parent.name] = option
		debug_trained_options = all_options

		# Check for empty list
		if not all_options:
			return []

		# Sort by lineage in chain
		skill_chain = [all_options['None']]
		for _ in range(len(all_options)-1):
			last_option = skill_chain[-1]
			try:
				skill_chain.append(all_options[last_option.name])
			except:
				print("DEBUG: waiting on untrained option to fix the chain")
				print("|-> DEBUG: self.trained_options[1:] = {}".format(self.trained_options[1:]))
				print("|-> DEBUG: debug_trained_options = {}".format(debug_trained_options))
				print("|-> DEBUG: self.untrained_option = {}".format(self.untrained_option))
				# Options could be in training
				break
		return skill_chain

	# TODO: assign new parent when chain fix option is created
	def update_options_parent(self, child_option, new_parent_option):
		for i, option in enumerate(self.trained_options):
			if option.name == child_option.name:
				child_option.update_parent(new_parent_option)
				self.trained_options[i] = child_option

	# TODO: restructured
	def should_create_child_options(self, verbose=False):
		local_options = self.trained_options[1:]
		
		# TODO: Wait for a trained option
		if local_options == []:
			return False
		
		# TODO: hack for treasure game
		if "treasure" in self.mdp.env_name:
			start_state = self.start_state
		else:
			start_state = self.start_xy

		# TODO: remove iterating over init states
		for option in local_options:  # type: Option
			if option.is_init_true(start_state):
				if verbose:
					print("Init state is in {}'s initiation set classifier".format(option.name))
				return False
		return True
	
	# TODO: utilities
	def make_meshgrid(self, x, y, h=.01):
		"""Create a mesh of points to plot in

		Args:
			x: data to base x-axis meshgrid on
			y: data to base y-axis meshgrid on
			h: stepsize for meshgrid, optional

		Returns:
			X and y mesh grid
		"""
		x_min, x_max = x.min(), x.max()
		y_min, y_max = y.min(), y.max()
		xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                       np.arange(y_min, y_max, h))
		return xx, yy

	# TODO: utilities
	def plot_contours(self, ax, clf, xx, yy, **params):
		"""Plot the decision boundaries for a classifier.

		Args:
			ax: matplotlib axes object
			clf: a classifier
			xx: meshgrid ndarray
			yy: meshgrid ndarray
			params: dictionary of params to pass to contourf, optional
		
		Returns:
			Contour of decision boundary
		"""
		Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
		Z = Z.reshape(xx.shape)
		out = ax.contourf(xx, yy, Z, **params)
		return out

	# TODO: utilities
	def adjust_lightness(self, color, amount=0.5):
		import colorsys
		try:
			c = mc.cnames[color]
		except:
			c = color
		c = colorsys.rgb_to_hls(*mc.to_rgb(c))
		return colorsys.hls_to_rgb(c[0], max(0, min(1, amount * c[1])), c[2])

	# TODO: utilities
	def plot_boundary(self, x_mesh, y_mesh, clfs, colors, option_name, episode, experiment_name, alpha, img_name=None, img_alpha=1, goal=None, start=None):
		# Create plotting dir (if not created)
		path = '{}/plots/clf_plots'.format(experiment_name)
		Path(path).mkdir(exist_ok=True)
	
		if img_name != None:
			back_img = plt.imread(img_name)
			plt.imshow(back_img, extent=[x_mesh.min(), x_mesh.max(), y_mesh.min(), y_mesh.max()], alpha=img_alpha)

		patches = []
		
		g = plt.scatter(goal[0], goal[1], marker="*", color="y", s=80, path_effects=[pe.Stroke(linewidth=1, foreground='k'), pe.Normal()], label="Goal")
		patches.append(g)
		
		s = plt.scatter(start[0], start[1], marker="X", color="k", label="Start")
		patches.append(s)

		# Plot classifier boundaries
		for (clf_name, clf), color in zip(clfs.items(), colors):
			z = clf.predict(np.c_[x_mesh.ravel(), y_mesh.ravel()])
			z = (z.reshape(x_mesh.shape) > 0).astype(int)
			z = np.ma.masked_where(z == 0, z)

			if 'pessimistic' in clf_name.lower():
				cf = plt.contourf(x_mesh, y_mesh, z, colors=color, alpha=alpha, hatches='xxx')
				patches.append(mpatches.Patch(color=color, label=clf_name, alpha=alpha, hatch='xxx'))
			else:
				cf = plt.contourf(x_mesh, y_mesh, z, colors=color, alpha=alpha)
				patches.append(mpatches.Patch(color=color, label=clf_name, alpha=alpha))

		plt.xticks(())
		plt.yticks(())
		plt.title("{}".format(option_name))
		plt.legend(handles=patches, 
				   bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
		
		# Save plots
		plt.savefig("{}/{}_{}.png".format(path, option_name, episode), bbox_inches='tight', edgecolor='black', format='png', quality=100)
		plt.close()

		# TODO: remove
		print("|-> Figure {}/{}_{}.png saved!".format(path, option_name, episode))

	def plot_multi_boundaries(self, x_mesh, y_mesh, option_data, rgb_color_palette, experiment_name, alpha, plot_eps, img_name=None, img_alpha=1, goal=None, start=None):
		# Create plotting dir (if not created)
		path = '{}/plots/clf_plots'.format(experiment_name)
		Path(path).mkdir(exist_ok=True)

		if img_name:
			back_img = plt.imread(img_name)
			plt.imshow(back_img, extent=[x_mesh.min(), x_mesh.max(), y_mesh.min(), y_mesh.max()], alpha=img_alpha)

		patches = []

		# Gray classifier key
		default_colors = ['lightgrey', 'grey']
		once = False
		for i, (option_name, option) in enumerate(option_data.items()):
			if not once:
				for episode, episode_data in option.items():
					once = True
					clfs = episode_data['clfs_bounds']
					if episode == plot_eps:
						saved_eps = episode
						for (clf_name, clf), color in zip(clfs.items(), default_colors):
							if 'pessimistic' in clf_name.lower():
								pass
								patches.append(mpatches.Patch(color=color, label='{}'.format(clf_name), alpha=alpha, hatch='xxx'))
							else:
								patches.append(mpatches.Patch(color=color, label='{}'.format(clf_name), alpha=alpha))

		# Plot all options
		saved_eps = None
		colors =[mc.to_hex(rgb) for rgb in rgb_color_palette]
		dark_colors = [mc.to_hex(self.adjust_lightness(color, amount=0.5)) for color in colors]

		for i, (option_name, option) in enumerate(option_data.items()):
			for episode, episode_data in option.items():
				clfs = episode_data['clfs_bounds']
				# plot boundaries of classifiers
				if episode == plot_eps:
					saved_eps = episode
					patches.append(mpatches.Patch(color=colors[i], label='{}'.format(option_name), alpha=alpha))

					# Plot classifier boundaries
					for (clf_name, clf) in clfs.items():
						z = clf.predict(np.c_[x_mesh.ravel(), y_mesh.ravel()])
						z = (z.reshape(x_mesh.shape) > 0).astype(int)
						z = np.ma.masked_where(z == 0, z)
						if 'pessimistic' in clf_name.lower():
							cf = plt.contourf(x_mesh, y_mesh, z, colors=dark_colors[i], alpha=alpha, hatches='xxx')
							for collection in cf.collections:
								collection.set_edgecolor(dark_colors[i])
						else:
							cf = plt.contourf(x_mesh, y_mesh, z, colors=colors[i] , alpha=alpha)

		g = plt.scatter(goal[0], goal[1], marker="*", color="y", s=80, path_effects=[pe.Stroke(linewidth=1, foreground='k'), pe.Normal()], label="Goal")
		patches.append(g)
			
		s = plt.scatter(start[0], start[1], marker="X", color="k", label="Start")
		patches.append(s)

		plt.xticks(())
		plt.yticks(())
		plt.title("All Options' Sets")
		plt.legend(handles=patches, 
					bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
		
		# Save plots
		plt.savefig("{}/all_options_{}.png".format(path, saved_eps),bbox_inches='tight', edgecolor='black', format='png', quality=100)
		plt.close()

		print("|-> Figure {}/all_options_{}.png saved!".format(path, saved_eps))

	# TODO: utilities
	def plot_learning_curves(self, experiment_name, data):
		# Create plotting dir (if not created)
		path = '{}/plots/learning_curves'.format(experiment_name)
		Path(path).mkdir(exist_ok=True)
		
		plt.plot(data, label=experiment_name)
			
		plt.title("{}".format(self.mdp.env_name))
		plt.ylabel("Rewards")
		plt.xlabel("Episodes")
		plt.legend(title="Tests", bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)

		# Save plots
		plt.savefig("{}/learning_curve.png".format(path), bbox_inches='tight', edgecolor='black', format='png', quality=100)
		plt.close()

		# TODO: remove
		print("|-> Figure {}/learning_curve.png saved!".format(path))

	# TODO: export option data
	def save_all_data(self, logdir, args, episodic_scores, episodic_durations):
		data_dir = logdir + '/run_{}_all_data'.format(self.seed)
		Path(data_dir).mkdir(exist_ok=True)
		
		all_data = {}
		all_data['args'] = args
		all_data['option_data'] = self.option_data
		all_data['x_mesh'] = self.x_mesh
		all_data['y_mesh'] = self.y_mesh
		all_data['x_mesh_hd'] = self.x_mesh_hd
		all_data['y_mesh_hd'] = self.y_mesh_hd
		all_data['experiment_name'] = self.experiment_name
		all_data['per_episode_scores'] = episodic_scores
		all_data['mdp_env_name'] = self.mdp.env_name
		all_data['episodic_durations'] = episodic_durations
		all_data['pretrained'] = args.pretrained
		all_data['validation_scores'] = self.validation_scores
		all_data['num_options_history'] = self.num_options_history
		all_data['temporal_chain_breaks'] = self.temporal_chain_breaks
		all_data['option_chain_breaks'] = self.option_chain_breaks
		all_data['goal_chain_breaks'] = self.goal_chain_breaks
		all_data['final_skill_chain'] = self.final_skill_chain
		all_data['full_chain_breaks'] = self.full_chain_breaks
		all_data['temporal_full_chain_breaks'] = self.temporal_full_chain_breaks
		all_data['goal_xy'] = self.goal_xy
		all_data['start_xy'] = self.start_xy

		for var_name, data in all_data.items():
			with open(data_dir + '/' + var_name + '.pkl', 'wb+') as f:
				pickle.dump(data, f)

	# TODO: intermediate processing for plots
	def run_plot_processing(self, episode):
		for option in self.trained_options:
			if option.optimistic_classifier or option.initiation_classifier:
				# Per option data
				if self.use_old:
					clfs_bounds = {'Initiation set': option.initiation_classifier}
				else:
					clfs_bounds = {'Optimistic initiation set':option.optimistic_classifier, 'Pessimistic initiation set':option.pessimistic_classifier}

				# Per episode data
				if option.name not in self.option_data:
					episode_data = {}
					episode_data[episode] = {'clfs_bounds' : clfs_bounds}
					self.option_data[option.name] = episode_data
				else:
					self.option_data[option.name][episode] = {'clfs_bounds' : clfs_bounds}

	# TODO: main plotting calss for episodic plots
	def plot_episodic_plots(self, episode, per_episode_scores):
		sns.set_style("white")
		rgb_color_palette = sns.color_palette('deep')

		# plot all options' boundaries
		self.plot_multi_boundaries(x_mesh=self.x_mesh,
									y_mesh=self.y_mesh,
									option_data=self.option_data,
									rgb_color_palette=rgb_color_palette,
									experiment_name=self.log_dir,
									alpha=0.7,
									plot_eps=episode,
									img_name='images/point_maze_domain.png',
									img_alpha=0.9,
									goal=self.goal_xy,
									start=self.start_xy)
		
		# plot learning curves
		self.plot_learning_curves(self.log_dir, per_episode_scores)

	# TODO: utilities
	def update_options_episode(self, episode):
		for option in self.trained_options:
			option.update_episode(episode)

	# TODO: restructured
	def skill_chaining(self, num_episodes, num_steps):
		print("\nSkillChaining::skill_chaining: call")	# TODO: remove

		# For logging purposes
		per_episode_scores = []
		per_episode_durations = []
		last_10_scores = deque(maxlen=10)
		last_10_durations = deque(maxlen=10)

		# TODO: create plotting directory
		if self.episodic_plots or self.generate_plots:
			print("|-> Generating {}/plots..".format(self.log_dir))
			Path('{}/plots'.format(self.log_dir)).mkdir(exist_ok=True)

		# TODO: create mesh of environment
		if "treasure" in self.mdp.env_name:
			raise NotImplementedError
		else:
			# TODO: hardcode for point maze
			width_coord = height_coord = np.array([-2,11])
			self.x_mesh, self.y_mesh = self.make_meshgrid(width_coord, height_coord, h=0.1)	# for predictions

		for episode in range(num_episodes):
			print("|-> episode: {}".format(episode))	# TODO: remove
			self.mdp.reset()
			score = 0.
			step_number = 0
			uo_episode_terminated = False
			state = deepcopy(self.mdp.init_state)
			experience_buffer = []
			state_buffer = []
			episode_option_executions = defaultdict(lambda : 0)

			while step_number < num_steps:
				if step_number % 500 == 0:
					print("  |-> step_number: {}".format(step_number))  # TODO: remove
				experiences, reward, state, steps = self.take_action(
					state, step_number, episode_option_executions, episode)
				score += reward
				step_number += steps
				for experience in experiences:
					experience_buffer.append(experience)
					state_buffer.append(experience[0])

				# Don't forget to add the last s' to the buffer_length
				if state.is_terminal() or (step_number == num_steps - 1):
					state_buffer.append(state)
				
				# NOTE: this still works with old DSC methods
				# TODO: can only do one thing per episode
				skip_child = False
				if (not uo_episode_terminated):
					# train terminating option
					if self.untrained_option.is_term_true(state) and\
						self.max_num_options > 0 and self.untrained_option.get_training_phase() == 'gestation':
						uo_episode_terminated = True

						if self.untrained_option.train(experience_buffer, state_buffer):
							self._augment_agent_with_new_option(self.untrained_option, init_q_value=self.init_q)
					
					# check on last step in episode, can only create options if one is not being trained
					elif self.untrained_option.get_training_phase() != 'gestation':
						# check and fix chain breaks
						if self.use_chain_fix:
							uo_episode_terminated = True	# check only once per episode
							should_fix, new_fix_option = self.detect_and_fix_chain(episode, step_number)
							if should_fix:
								skip_child = True
								self.untrained_option = new_fix_option
						
						# check if children need to be made from trained options
						if (not skip_child) and self.should_create_child_options(verbose=False):	
							self.num_options += 1
							
							# if untrained option was a chain fix, need to create child from the last in chain
							last_option_in_chain = self.get_skill_chain()[-1]
							new_idx = self.num_options

							child_option = self.create_option(last_option_in_chain, new_idx, type='(child) ')
							self.untrained_option = child_option

				if state.is_terminal():
					break

			last_10_scores.append(score)
			last_10_durations.append(step_number)
			per_episode_scores.append(score)
			per_episode_durations.append(step_number)

			# TODO: log full chain breaks
			self.log_full_chain_breaks(episode)

			# TODO: save data for plots
			self.run_plot_processing(episode)

			# TODO: call for making episodic plots
			if self.episodic_plots:
				self.plot_episodic_plots(episode, per_episode_scores)

			# TODO: log if child doesn't need to be created this episode
			self.should_create_child_options(verbose=True)

			self._log_dqn_status(episode, last_10_scores, episode_option_executions, last_10_durations)
			
			# TODO: print chain info
			print("Current Skill Chain: {}".format(self.get_skill_chain()))
			print("EPISODE FULL CHAIN BREAKS: {}\n".format(self.temporal_full_chain_breaks[episode]))

			# TODO: save data per episode
			if self.episodic_saves:
				self.save_all_data(self.log_dir, self.args, per_episode_scores, per_episode_durations)

			# TODO: update episode count of trained options
			self.update_options_episode(episode)
			self.episode += 1

		# TODO: post run assignments
		self.final_skill_chain = [str(option.name) for option in self.get_skill_chain()]
		self.x_mesh_hd, self.y_mesh_hd = self.make_meshgrid(width_coord, height_coord, h=0.01)	# save HD mesh for plotting

		print("Saving all data...")
		self.save_all_data(self.log_dir, self.args, per_episode_scores, per_episode_durations)

		return per_episode_scores, per_episode_durations

	def _log_dqn_status(self, episode, last_10_scores, episode_option_executions, last_10_durations):

		print('Episode {}\tScore: {:.2f}\tAverage Score: {:.2f}\tDuration: {:.2f} steps\tGO Eps: {:.2f}'.format(
			episode, last_10_scores[-1], np.mean(last_10_scores), np.mean(last_10_durations), self.global_option.solver.epsilon))

		self.num_options_history.append(len(self.trained_options))

		if self.writer is not None:
			self.writer.add_scalar("Episodic scores", last_10_scores[-1], episode)

		if episode > 0 and episode % 100 == 0:
			eval_score = self.trained_forward_pass(render=False)
			self.validation_scores.append(eval_score)
			# print("\rEpisode {}\tValidation Score: {:.2f}".format(episode, eval_score))

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

	# TODO: old
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

# TODO: restructured
def create_log_dir(path):
	Path(path).mkdir(exist_ok=True)
	return path

if __name__ == '__main__':
	# TODO: added more arguments
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
	parser.add_argument("--episodic_plots", type=bool, help="Whether or not to produce plots for each episode ", default=False)
	parser.add_argument("--tensor_log", type=bool, help="Enable tensorboard logging", default=False)
	parser.add_argument("--control_cost", type=bool, help="Penalize high actuation solutions", default=False)
	parser.add_argument("--dense_reward", type=bool, help="Use dense/sparse rewards", default=False)
	parser.add_argument("--max_num_options", type=int, help="Max number of options we can learn", default=np.inf)
	parser.add_argument("--num_subgoal_hits", type=int, help="Number of subgoal hits to learn an option", default=3)
	parser.add_argument("--buffer_len", type=int, help="buffer size used by option to create init sets", default=20)
	parser.add_argument("--classifier_type", type=str, help="ocsvm/elliptic for option initiation clf", default="ocsvm")
	parser.add_argument("--init_q", type=str, help="compute/zero", default="zero")
	parser.add_argument("--use_smdp_update", type=bool, help="sparse/SMDP update for option policy", default=False)
	parser.add_argument("--opt_nu", type=float, help="Conservative parameter for optimistic classifier", default=0.5)
	parser.add_argument("--pes_nu", type=float, help="Conservative parameter for pessimistic classifier", default=0.5)
	parser.add_argument("--num_run", type=int, help="The number of the current run.", default=0)
	parser.add_argument("--discrete_actions", type=bool, help="Whether or not actions are discrete", default=False)
	parser.add_argument("--lr_dqn", type=float, help="DQN learning rate", default=1e-4)
	parser.add_argument("--use_old", type=bool, help="Whether to use older DSC methods", default=False)
	parser.add_argument("--episodic_saves", type=bool, help="Save all data per episode", default=False)
	parser.add_argument("--use_chain_fix", type=bool, help="Whether or not to use chain fixing method", default=False)
	args = parser.parse_args()

	if "reacher" in args.env.lower():
		from simple_rl.tasks.fixed_reacher.FixedReacherMDPClass import FixedReacherMDP
		overall_mdp = FixedReacherMDP(seed=args.seed, dense_reward=args.dense_reward, render=args.render)
		state_dim = overall_mdp.init_state.features().shape[0]
		action_dim = overall_mdp.env.action_space.shape[0]
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
	elif "treasure" in args.env.lower():	# TODO: new treasure game domain
		from simple_rl.tasks.treasure_game.TreasureGameMDPClass import TreasureGameMDP
		overall_mdp = TreasureGameMDP(seed=args.seed, dense_reward=args.dense_reward, render=args.render)
		state_dim = overall_mdp.init_state.features().shape[0]
		action_dim = overall_mdp.env.action_space.n
		overall_mdp.env.seed(args.seed)
	else:
		from simple_rl.tasks.gym.GymMDPClass import GymMDP
		overall_mdp = GymMDP(args.env, render=args.render)
		state_dim = overall_mdp.env.observation_space.shape[0]
		action_dim = overall_mdp.env.action_space.n
		overall_mdp.env.seed(args.seed)

	# TODO: changed log dir
	# Create folders for saving various things
	logdir = create_log_dir('runs/' + args.experiment_name)
	create_log_dir("saved_runs")
	create_log_dir("value_function_plots")
	create_log_dir("initiation_set_plots")
	create_log_dir("value_function_plots/{}".format(args.experiment_name))
	create_log_dir("initiation_set_plots/{}".format(args.experiment_name))

	print("Training skill chaining agent from scratch with a subgoal reward = {} and buffer_len = {}".format(args.subgoal_reward, args.buffer_len))
	
	# TODO: added MDP info
	print("\nMDP: {}".format(overall_mdp.env_name))
	print("MDP InitState = ", overall_mdp.init_state)

	q0 = 0. if args.init_q == "zero" else None

	chainer = SkillChaining(overall_mdp, args.steps, args.lr_a, args.lr_c, args.lr_dqn, args.ddpg_batch_size,
							seed=args.seed, subgoal_reward=args.subgoal_reward,
							log_dir=logdir, num_subgoal_hits_required=args.num_subgoal_hits,
							enable_option_timeout=args.option_timeout, init_q=q0, use_full_smdp_update=args.use_smdp_update,
							generate_plots=args.generate_plots, episodic_plots=args.episodic_plots, tensor_log=args.tensor_log, device=args.device,
							opt_nu=args.opt_nu, pes_nu=args.pes_nu, experiment_name=args.experiment_name, num_run=args.num_run, discrete_actions=args.discrete_actions,
							use_old=args.use_old, episodic_saves=args.episodic_saves, args=args, use_chain_fix=args.use_chain_fix)
	episodic_scores, episodic_durations = chainer.skill_chaining(args.episodes, args.steps)

	# TODO: print final run info
	print("Scores: {}".format(episodic_scores))
	print("Final Skill Chain: {}".format(chainer.final_skill_chain))

	# TODO: old
	# Log performance metrics
	# chainer.save_all_models()
	# chainer.perform_experiments()
	# chainer.save_all_scores(args.pretrained, episodic_scores, episodic_durations)
