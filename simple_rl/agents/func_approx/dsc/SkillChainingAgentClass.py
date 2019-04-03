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

# Other imports.
from simple_rl.mdp.StateClass import State
from simple_rl.agents.func_approx.dqn.DQNAgentClass import DQNAgent
from simple_rl.agents.func_approx.dsc.OptionClass import Option
from simple_rl.agents.func_approx.dsc.utils import *
from simple_rl.agents.func_approx.ddpg.utils import *
from simple_rl.tasks.point_env.PointEnvMDPClass import PointEnvMDP


class SkillChaining(object):
	def __init__(self, mdp, max_steps, lr_dqn, lr_actor, lr_critic, ddpg_batch_size,
				 device, pretrained_options=[], buffer_length=20,
				 subgoal_reward=1.0, subgoal_hits=3, max_num_options=4,
				 enable_option_timeout=True, intra_option_learning=False,
				 generate_plots=False, log_dir="", seed=0, tensor_log=False):
		"""
		Args:
			mdp (MDP): Underlying domain we have to solve
			max_steps (int): Number of time steps allowed per episode of the MDP
			lr_dqn (float): Learning rate for agent over options
			lr_actor (float): Learning rate for DDPG Actor
			lr_critic (float): Learning rate for DDPG Critic
			ddpg_batch_size (int): Batch size for DDPG agents
			device (str): torch device {cpu/cuda:0/cuda:1}
			pretrained_options (list): options obtained from a previous run of the skill chaining algorithm
			buffer_length (int): size of the circular buffer used as experience buffer
			subgoal_reward (float): Hitting a subgoal must yield a supplementary reward to enable local policy
			subgoal_hits (int): number of times the RL agent has to hit the goal of an option o to learn its I_o, Beta_o
			max_num_options (int): Maximum number of options that the skill chaining agent can create
			enable_option_timeout (bool): whether or not the option times out after some number of steps
			intra_option_learning (bool): whether or not to use intra-option learning while making SMDP updates
			generate_plots (bool): whether or not to produce plots in this run
			log_dir (os.path): directory to store all the scores for this run
			seed (int): We are going to use the same random seed for all the DQN solvers
			tensor_log (bool): Tensorboard logging enable
		"""
		self.mdp = mdp
		self.original_actions = deepcopy(mdp.actions)
		self.max_steps = max_steps
		self.buffer_length = buffer_length
		self.subgoal_reward = subgoal_reward
		self.num_goal_hits_before_training = subgoal_hits
		self.max_num_options = max_num_options
		self.enable_option_timeout = enable_option_timeout
		self.enable_intra_option_learning = intra_option_learning
		self.generate_plots = generate_plots
		self.log_dir = log_dir
		self.seed = seed
		self.device = torch.device(device)

		tensor_name = "runs/{}_{}".format(args.experiment_name, seed)
		self.writer = SummaryWriter(tensor_name) if tensor_log else None

		print("Initializing skill chaining with option_timeout={}, seed={}".format(self.enable_option_timeout, seed))

		random.seed(seed)
		np.random.seed(seed)

		# If we are given pretrained options, we will just use them as trained options
		if len(pretrained_options) > 0:
			self.trained_options = pretrained_options

		self.validation_scores = []

		# This option has an initiation set that is true everywhere and is allowed to operate on atomic timescale only
		self.global_option = Option(overall_mdp=self.mdp, name="global_option", global_solver=None,
									buffer_length=self.buffer_length, lr_actor=lr_actor, lr_critic=lr_critic,
									ddpg_batch_size=ddpg_batch_size,
									num_subgoal_hits_required=self.num_goal_hits_before_training,
									subgoal_reward=self.subgoal_reward, seed=self.seed, max_steps=self.max_steps,
									classifier_type="ocsvm", enable_timeout=self.enable_option_timeout,
									generate_plots=self.generate_plots, writer=self.writer, device=self.device)

		self.trained_options = [self.global_option]

		# This is our policy over options
		# We use (double-deep) (intra-option) Q-learning to learn the Q-values of *options* at any queried state Q(s, o)
		# We start with this DQN Agent only predicting Q-values for taking the global_option, but as we learn new
		# options, this agent will predict Q-values for them as well
		self.agent_over_options = DQNAgent(self.mdp.state_space_size(), 1, trained_options=self.trained_options,
										   seed=seed, lr=lr_dqn, name="GlobalDQN", eps_start=1.0, tensor_log=True,
										   use_double_dqn=True, writer=self.writer, device=self.device)

		# This is our first untrained option - one that gets us to the goal state from nearby the goal
		# We pick this option when self.agent_over_options thinks that we should
		# Once we pick this option, we will use its internal DDPG solver to take primitive actions until termination
		# Once we hit its termination condition N times, we will start learning its initiation set
		# Once we have learned its initiation set, we will create its child option
		goal_option = Option(overall_mdp=self.mdp, name='overall_goal_policy', global_solver=self.global_option.solver,
							 buffer_length=self.buffer_length, lr_actor=lr_actor, lr_critic=lr_critic,
							 ddpg_batch_size=ddpg_batch_size, num_subgoal_hits_required=self.num_goal_hits_before_training,
							 subgoal_reward=self.subgoal_reward, seed=self.seed, max_steps=self.max_steps,
							 classifier_type="ocsvm", enable_timeout=self.enable_option_timeout,
							 generate_plots=self.generate_plots, writer=self.writer, device=self.device)

		# Pointer to the current option:
		# 1. This option has the termination set which defines our current goal trigger
		# 2. This option has an untrained initialization set and policy, which we need to train from experience
		self.untrained_option = goal_option

		# Debug variables
		self.global_execution_states = []
		self.num_option_executions = defaultdict(lambda : [])
		self.option_rewards = defaultdict(lambda : [])
		self.option_qvalues = defaultdict(lambda : [])

	def create_child_option(self):
		# Create new option whose termination is the initiation of the option we just trained
		name = "option_{}".format(str(len(self.trained_options) - 1))
		print("Creating {}".format(name))

		old_untrained_option_id = id(self.untrained_option)
		new_untrained_option = Option(self.mdp, name=name, global_solver=self.global_option.solver,
									  buffer_length=self.buffer_length,
									  lr_actor=self.untrained_option.solver.actor_learning_rate,
									  lr_critic=self.untrained_option.solver.critic_learning_rate,
									  ddpg_batch_size=self.untrained_option.solver.batch_size,
									  num_subgoal_hits_required=self.num_goal_hits_before_training,
									  subgoal_reward=self.subgoal_reward,
									  seed=self.seed, parent=self.untrained_option,
									  enable_timeout=self.enable_option_timeout,
									  writer=self.writer, device=self.device)

		self.untrained_option.children.append(new_untrained_option)
		self.untrained_option = new_untrained_option

		new_untrained_option_id = id(self.untrained_option)
		assert new_untrained_option_id != old_untrained_option_id, "Checking python references"
		assert id(self.untrained_option.parent) == old_untrained_option_id, "Checking python references"

	def get_init_q_value_for_new_option(self, newly_trained_option):
		global_solver = self.agent_over_options  # type: DQNAgent
		state_option_pairs = newly_trained_option.final_transitions
		q_values = []
		for state, option_idx in state_option_pairs:
			q_value = global_solver.get_qvalue(state.features(), option_idx)
			q_values.append(q_value)
		return np.max(q_values)

	def _augment_agent_with_new_option(self, newly_trained_option, init_q=None):
		"""
		Train the current untrained option and initialize a new one to target.
		Add the newly_trained_option as a new node to the Q-function over options
		Args:
			newly_trained_option (Option)
			init_q (float): if not None, use this value to initialize the q-value of the new option
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

		init_q_value = self.get_init_q_value_for_new_option(newly_trained_option) if init_q is None else init_q
		print("Initializing new option node with q value {}".format(init_q_value))
		new_global_agent.policy_network.initialize_with_smaller_network(self.agent_over_options.policy_network, init_q_value)
		new_global_agent.target_network.initialize_with_smaller_network(self.agent_over_options.target_network, init_q_value)

		self.agent_over_options = new_global_agent

		# TODO: I don't think options need to know about the new DQN agent
		# Update the global solver of all previously trained options
		# for trained_option in self.trained_options:
		#     trained_option.global_solver = new_global_agent

	def make_off_policy_updates_for_options(self, state, action, reward, next_state):
		for option in self.trained_options: # type: Option
			option.off_policy_update(state, action, reward, next_state)

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

		# TODO: Should we do intra-option learning only when the option was successful in reaching its subgoal?
		# TODO: (since the option could have "failed" simply because it timed out)
		selected_option = self.trained_options[action]  # type: Option
		if self.enable_intra_option_learning:
			for i, transition in enumerate(option_transitions):
				start_state = transition[0]
				if selected_option.is_init_true(start_state):
					option_reward = self.subgoal_reward if selected_option.is_term_true(next_state) else -1.
					self.agent_over_options.step(start_state.features(), action, option_reward, next_state.features(),
											next_state.is_terminal(), num_steps=1)
		else:
			# The policy over options gets a +0 reward if the lower level option is able to successfully able
			# to reach its subgoal and -1 reward if its not.
			# In this way, the (s, o, r, s') transitions can become stationary and since R_max in the MDP is 0,
			# we can be sure that the agent will not go around and around forever collecting subgoal rewards
			option_reward = self.subgoal_reward if selected_option.is_term_true(next_state) else -1.
			self.agent_over_options.step(state.features(), action, option_reward, next_state.features(),
									next_state.is_terminal(), num_steps=1)

	def take_action(self, state, step_number, episode_option_executions):
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
		# Query the global Q-function to determine which option to take in the current state
		option_idx = self.agent_over_options.act(state.features(), train_mode=True)
		self.agent_over_options.update_epsilon()

		# Selected option
		selected_option = self.trained_options[option_idx] # type: Option
		option_transitions, discounted_reward = selected_option.execute_option_in_mdp(self.mdp, step_number)

		option_reward = self.get_reward_from_experiences(option_transitions)
		next_state = self.get_next_state_from_experiences(option_transitions)

		# If we triggered the untrained option's termination condition, add to its buffer of terminal transitions
		if self.untrained_option.is_term_true(next_state) and not self.untrained_option.is_term_true(state):
			self.untrained_option.final_transitions.append((state, option_idx))

		# Add data to train Q(s, o)
		self.make_smdp_update(state, option_idx, discounted_reward, next_state, option_transitions)

		# Debug logging
		episode_option_executions[selected_option.name] += 1
		self.option_rewards[selected_option.name].append(discounted_reward)

		if not selected_option.pretrained:
			sampled_q_value = self.sample_qvalue(option_idx)
			self.option_qvalues[selected_option.name].append(sampled_q_value)
			if self.writer is not None:
				self.writer.add_scalar("{}_q_value".format(selected_option.name),
									   sampled_q_value, selected_option.num_executions)

		return option_transitions, option_reward, next_state, len(option_transitions)

	def sample_qvalue(self, option_idx):
		option = self.trained_options[option_idx]  # type: Option
		if len(option.solver.replay_buffer) > 500:
			sample_experiences = option.solver.replay_buffer.sample(batch_size=500)
			sample_states = torch.from_numpy(sample_experiences[0]).float().to(self.device)
			return self.agent_over_options.get_batched_qvalues(sample_states)[:, option_idx].mean()
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
		start_state = deepcopy(self.mdp.init_state)
		for option in self.trained_options: # type: Option
			if option.is_init_true(start_state):
				return False
		return len(self.trained_options) < self.max_num_options

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
			uo_episode_terminated = False
			state = deepcopy(self.mdp.init_state)
			experience_buffer = []
			state_buffer = []
			episode_option_executions = defaultdict(lambda : 0)

			while step_number < num_steps:
				experiences, reward, state, steps = self.take_action(state, step_number, episode_option_executions)
				score += reward
				step_number += steps
				for experience in experiences:
					experience_buffer.append(experience)
					state_buffer.append(experience[0])

				# Don't forget to add the last s' to the buffer
				if state.is_terminal() or (step_number == num_steps - 1):
					state_buffer.append(state)

				if self.untrained_option.is_term_true(state) and (not uo_episode_terminated) and \
						self.untrained_option.get_training_phase() == "gestation" and self.max_num_options > 0:

					uo_episode_terminated = True

					if self.untrained_option.train(experience_buffer, state_buffer):
						if self.generate_plots:
							render_sampled_value_function(self.agent_over_options, episode=episode-10000)
							render_sampled_initiation_classifier(self.untrained_option)
						self._augment_agent_with_new_option(self.untrained_option)
						if self.generate_plots:
							render_sampled_value_function(self.agent_over_options, episode=episode+10000)

				if self.untrained_option.get_training_phase() == "initiation_done" and self.should_create_more_options():
					self.create_child_option()

				if state.is_terminal():
					break

			last_10_scores.append(score)
			last_10_durations.append(step_number)
			per_episode_scores.append(score)
			per_episode_durations.append(step_number)

			self._log_dqn_status(episode, last_10_scores, episode_option_executions, last_10_durations)

		return per_episode_scores, per_episode_durations

	def _log_dqn_status(self, episode, last_10_scores, episode_option_executions, last_10_durations):

		print('\rEpisode {}\tAverage Score: {:.2f}\tDuration: {:.2f} steps\tOP Eps: {:.2f}\tGO Eps: {:.2f}'.format(
			episode, np.mean(last_10_scores), np.mean(last_10_durations),
			self.agent_over_options.epsilon, self.global_option.solver.epsilon), end="")

		if episode % 10 == 0:
			print('\rEpisode {}\tAverage Score: {:.2f}\tDuration: {:.2f} steps\tOP Eps: {:.2f}\tGO Eps: {:.2f}'.format(
				episode, np.mean(last_10_scores), np.mean(last_10_durations),
				self.agent_over_options.epsilon, self.global_option.solver.epsilon))

		if episode > 0 and episode % 100 == 0:
			eval_score = self.trained_forward_pass(render=False)
			self.validation_scores.append(eval_score)
			print("\rEpisode {}\tValidation Score: {:.2f}".format(episode, eval_score))

		if self.generate_plots and episode % 10 == 0:
			render_sampled_value_function(self.agent_over_options, episode=episode)

		for trained_option in self.trained_options:  # type: Option
			self.num_option_executions[trained_option.name].append(episode_option_executions[trained_option.name])
			if self.writer is not None:
				self.writer.add_scalar("{}_executions".format(trained_option.name),
									   episode_option_executions[trained_option.name], episode)

	def save_all_models(self):
		torch.save(self.agent_over_options.policy_network.state_dict(), 'saved_runs/global_policy_dqn.pth')
		torch.save(self.agent_over_options.target_network.state_dict(), 'saved_runs/global_target_dqn.pth')
		for option in self.trained_options: # type: Option
			save_model(option.solver, args.episodes, best=False)

	def save_all_initiation_classifiers(self):
		for option in self.trained_options:
			with open("saved_runs/{}_svm.pkl".format(option.name), "wb+") as _f:
				pickle.dump(option.initiation_classifier, _f)

	def save_all_scores(self, pretrained, scores, durations):
		print("\rSaving training and validation scores..")
		training_scores_file_name = "sc_pretrained_{}_training_scores_{}.pkl".format(pretrained, self.seed)
		training_durations_file_name = "sc_pretrained_{}_training_durations_{}.pkl".format(pretrained, self.seed)
		validation_scores_file_name = "sc_pretrained_{}_validation_scores_{}.pkl".format(pretrained, self.seed)

		if self.log_dir:
			training_scores_file_name = os.path.join(self.log_dir, training_scores_file_name)
			training_durations_file_name = os.path.join(self.log_dir, training_durations_file_name)
			validation_scores_file_name = os.path.join(self.log_dir, validation_scores_file_name)

		with open(training_scores_file_name, "wb+") as _f:
			pickle.dump(scores, _f)
		with open(training_durations_file_name, "wb+") as _f:
			pickle.dump(durations, _f)
		with open(validation_scores_file_name, "wb+") as _f:
			pickle.dump(self.validation_scores, _f)

	def perform_experiments(self):
		for option in self.trained_options:
			visualize_dqn_replay_buffer(option.solver, args.experiment_name)
		visualize_dqn_replay_buffer(self.agent_over_options, args.experiment_name)
		visualize_smdp_updates(self.agent_over_options, args.experiment_name)
		render_sampled_value_function(self.agent_over_options, episode=args.episodes, experiment_name=args.experiment_name)
		for i, o in enumerate(self.trained_options):
			plt.subplot(1, len(self.trained_options), i + 1)
			plt.plot(self.option_qvalues[o.name])
			plt.title(o.name)
		plt.savefig("value_function_plots/{}_sampled_q_so_{}.png".format(args.experiment_name, self.seed))
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

		while not state.is_terminal() and num_steps < self.max_steps:
			option_idx = self.agent_over_options.act(state.features(), train_mode=False)
			selected_option = self.trained_options[option_idx]  # type: Option

			option_reward, next_state, num_steps = selected_option.trained_option_execution(self.mdp, num_steps)
			overall_reward += option_reward

			state = next_state

		return overall_reward

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
	parser.add_argument("--intra_option_learning", type=bool, help="Whether or not to use i-o-l", default=False)
	parser.add_argument("--buffer_len", type=int, help="SkillChaining Buffer Length", default=100)
	parser.add_argument("--subgoal_reward", type=float, help="SkillChaining subgoal reward", default=0.)
	parser.add_argument("--lr_dqn", type=float, help="DQN learning rate", default=1e-4)
	parser.add_argument("--lr_a", type=float, help="DDPG Actor learning rate", default=1e-4)
	parser.add_argument("--lr_c", type=float, help="DDPG Critic learning rate", default=1e-3)
	parser.add_argument("--dqn_batch_size", type=int, help="DQN Batch Size", default=64)
	parser.add_argument("--ddpg_batch_size", type=int, help="DDPG Batch Size", default=64)
	parser.add_argument("--n_options", type=int, help="Max # options to learn", default=3)
	parser.add_argument("--render", type=bool, help="Render the mdp env", default=False)
	parser.add_argument("--hard_coded", type=bool, help="Whether to use hard coded goal option", default=False)
	parser.add_argument("--option_timeout", type=bool, help="Whether option times out at 200 steps", default=False)
	parser.add_argument("--difficulty", type=str, help="Difficulty level of the task", default="easy")
	parser.add_argument("--generate_plots", type=bool, help="Whether or not to generate plots", default=False)
	parser.add_argument("--tensor_log", type=bool, help="Enable tensorboard logging", default=False)
	parser.add_argument("--control_cost", type=bool, help="Penalize high actuation solutions", default=False)
	args = parser.parse_args()

	if "reacher" in args.env.lower():
		from simple_rl.tasks.dm_fixed_reacher.FixedReacherMDPClass import FixedReacherMDP
		overall_mdp = FixedReacherMDP(seed=args.seed, difficulty=args.difficulty, render=args.render)
		state_dim = overall_mdp.init_state.features().shape[0]
		action_dim = overall_mdp.env.action_spec().minimum.shape[0]
	elif "maze" in args.env.lower():
		from simple_rl.tasks.point_maze.PointMazeMDPClass import PointMazeMDP
		overall_mdp = PointMazeMDP(seed=args.seed, render=args.render)
		state_dim = 6
		action_dim = 2
	elif "point" in args.env.lower():
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

	print("Training skill chaining agent from scratch with a buffer length of {} and subgoal reward {}".format(
		args.buffer_len, args.subgoal_reward))
	print("MDP InitState = ", overall_mdp.init_state)

	chainer = SkillChaining(overall_mdp, args.steps, args.lr_dqn, args.lr_a, args.lr_c, args.ddpg_batch_size,
							buffer_length=args.buffer_len, seed=args.seed, subgoal_reward=args.subgoal_reward,
							max_num_options=args.n_options, log_dir=logdir,
							intra_option_learning=args.intra_option_learning,
							enable_option_timeout=args.option_timeout, generate_plots=args.generate_plots,
							tensor_log=args.tensor_log, device=args.device)
	episodic_scores, episodic_durations = chainer.skill_chaining(args.episodes, args.steps)

	# Log performance metrics
	chainer.save_all_models()
	chainer.save_all_initiation_classifiers()
	chainer.perform_experiments()
	chainer.save_all_scores(args.pretrained, episodic_scores, episodic_durations)
