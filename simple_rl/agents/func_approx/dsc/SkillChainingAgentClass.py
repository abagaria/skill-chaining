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
from simple_rl.agents.func_approx.dsc.SkillChainingPlotterClass import SkillChainingPlotter


class SkillChaining(object):
    def __init__(self, mdp, max_steps, lr_actor, lr_critic, ddpg_batch_size, device, max_num_options=5,
                 subgoal_reward=0., enable_option_timeout=True, buffer_length=20, num_subgoal_hits_required=3,
                 classifier_type="ocsvm", init_q=None, generate_plots=False, use_full_smdp_update=False,
                 start_state_salience=False, option_intersection_salience=False, event_intersection_salience=False,
                 pretrain_option_policies=False, create_backward_options=False, learn_backward_options_offline=False,
                 update_global_solver=False, use_warmup_phase=False, dense_reward=False,
                 seed=0, tensor_log=False, experiment_name="", plotter=None, fixed_epsilon=False):
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
            seed (int): We are going to use the same random seed for all the DQN solvers
            tensor_log (bool): Tensorboard logging enable
            experiment_name (str)
            plotter (SkillChainingPlotterClass): Plots any graphs of domain such as value function and initiation sets
            fixed_epsilon (bool): Use fixed epsilon for option DDPG if true, decreasing epsilon over time otherwise
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
        self.seed = seed
        self.device = torch.device(device)
        self.max_num_options = max_num_options
        self.classifier_type = classifier_type
        self.dense_reward = dense_reward
        self.start_state_salience = start_state_salience
        self.option_intersection_salience = option_intersection_salience
        self.event_intersection_salience = event_intersection_salience
        self.pretrain_option_policies = pretrain_option_policies
        self.create_backward_options = create_backward_options
        self.learn_backward_options_offline = learn_backward_options_offline
        self.update_global_solver = update_global_solver
        self.use_warmup_phase = use_warmup_phase
        self.experiment_name = experiment_name
        self.plotter = plotter
        Option.fixed_epsilon = fixed_epsilon
        ipdb.set_trace()

        tensor_name = "runs/{}_{}".format(self.experiment_name, seed)
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
                                    use_warmup_phase=self.use_warmup_phase, update_global_solver=self.update_global_solver,
                                    dense_reward=self.dense_reward, chain_id=None, is_backward_option=False, option_idx=0)

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
                                 dense_reward=self.dense_reward, chain_id=i + 1, max_num_children=1,
                                 use_warmup_phase=self.use_warmup_phase, update_global_solver=self.update_global_solver,
                                 target_salient_event=salient_event, option_idx=i + 1)
            self.untrained_options.append(goal_option)

        # This is our policy over options
        # We use (double-deep) (intra-option) Q-learning to learn the Q-values of *options* at any queried state Q(s, o)
        # We start with this DQN Agent only predicting Q-values for taking the global_option, but as we learn new
        # options, this agent will predict Q-values for them as well
        self.agent_over_options = DQNAgent(self.mdp.state_space_size(), 1, trained_options=self.trained_options,
                                           seed=seed, lr=1e-4, name="GlobalDQN", eps_start=0.1, tensor_log=tensor_log,
                                           use_double_dqn=True, writer=self.writer, device=self.device,
                                           exploration_strategy="shaping")

        # Keep track of which chain each created option belongs to
        start_state_salient_event = self.mdp.get_start_state_salient_event()
        self.s0 = start_state_salient_event.trigger_points
        self.chains = [SkillChain(start_states=self.s0, options=[], chain_id=(i + 1),
                                  intersecting_options=[], mdp_start_states=self.s0, is_backward_chain=False,
                                  target_salient_event=salient_event, init_salient_event=start_state_salient_event,
                                  option_intersection_salience=option_intersection_salience,
                                  event_intersection_salience=event_intersection_salience)
                       for i, salient_event in enumerate(self.mdp.get_original_target_events())]

        # List of init states seen while running this algorithm
        self.init_states = []

        self.current_option_idx = 1
        self.current_intersecting_chains = []

        # If initialize_everywhere is True, also add to trained_options
        for option in self.untrained_options:  # type: Option
            if option.initialize_everywhere:
                self.augment_agent_with_new_option(option, 0.)

        # Debug variables
        self.global_execution_states = []
        self.num_option_executions = defaultdict(lambda: [])
        self.option_rewards = defaultdict(lambda: [])
        self.option_qvalues = defaultdict(lambda: [])
        self.num_options_history = []

    def create_chain_targeting_new_salient_event(self, salient_event):
        """
        Given a new `salient_event`, create an option and the corresponding skill chain targeting it.
        Args:
            salient_event (SalientEvent)

        Returns:
            skill_chain (SkillChain)
        """
        chain_id = len(self.chains) + 1
        init_salient_event = self.mdp.get_start_state_salient_event()
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
                             is_backward_option=False,
                             init_salient_event=init_salient_event,
                             target_salient_event=salient_event)
        self.untrained_options.append(goal_option)

        new_chain = SkillChain(start_states=self.s0, options=[], chain_id=chain_id,
                               intersecting_options=[], mdp_start_states=self.s0, is_backward_chain=False,
                               target_salient_event=salient_event,
                               init_salient_event=init_salient_event,
                               option_intersection_salience=self.option_intersection_salience,
                               event_intersection_salience=self.event_intersection_salience)
        self.add_skill_chain(new_chain)

        if goal_option.initialize_everywhere:
            self.augment_agent_with_new_option(goal_option, 0.)

        return new_chain

    def add_skill_chain(self, new_chain):
        if new_chain not in self.chains:
            self.chains.append(new_chain)

    def add_negative_examples(self, new_option):
        """ When creating an option in the forward chain, add the positive examples of all the options
			in the forward chain as negative examples for the current option. Do the same when creating
			a new_option in the backward chain, i.e, only add positives in the backward chain as
			negative examples for the new_option.
		"""
        forward_option = not new_option.backward_option
        if forward_option:
            for option in self.trained_options[1:]:  # type: Option
                new_option.negative_examples += option.positive_examples
        else:
            for option in self.trained_options[1:]:  # type: Option
                if option.backward_option:
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
        start_states = self.chains[option.chain_id - 1].start_states
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
               and self.chains[parent_option.chain_id - 1].should_continue_chaining(self.chains)

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
        if self.init_state_in_option(parent_option) \
                or len(parent_option.children) >= parent_option.max_num_children:
            return None

        # Create new option whose termination is the initiation of the option we just trained
        if "goal_option" in parent_option.name:
            prefix = f"option_{parent_option.name.split('_')[-1]}"
        else:
            prefix = parent_option.name
        name = prefix + "_{}".format(len(parent_option.children))
        print("Creating {} with parent {}".format(name, parent_option.name))

        # Options in the backward chain have a pre-set init_salient_event defined for them.
        # This is generally the target event of the corresponding forward skill chain. Options
        # in the current backward chain have their init_salient_event parameter set, which
        # is only used during the gestation period of options
        is_backward_option = self.chains[parent_option.chain_id - 1].is_backward_chain
        gestation_init_salient_event = None
        if is_backward_option:
            gestation_init_salient_event = parent_option.init_salient_event  # type: SalientEvent

        old_untrained_option_id = id(parent_option)
        new_untrained_option = Option(self.mdp, name=name, global_solver=self.global_option.solver,
                                      lr_actor=parent_option.solver.actor_learning_rate,
                                      lr_critic=parent_option.solver.critic_learning_rate,
                                      ddpg_batch_size=parent_option.solver.batch_size,
                                      subgoal_reward=self.subgoal_reward,
                                      buffer_length=self.buffer_length,
                                      classifier_type=self.classifier_type,
                                      num_subgoal_hits_required=self.num_subgoal_hits_required,
                                      seed=self.seed, parent=parent_option, max_steps=self.max_steps,
                                      enable_timeout=self.enable_option_timeout, chain_id=parent_option.chain_id,
                                      writer=self.writer, device=self.device, dense_reward=self.dense_reward,
                                      initialize_everywhere=parent_option.initialize_everywhere,
                                      max_num_children=parent_option.max_num_children,
                                      is_backward_option=is_backward_option,
                                      init_salient_event=gestation_init_salient_event,
                                      use_warmup_phase=self.use_warmup_phase,
                                      update_global_solver=self.update_global_solver,
                                      initiation_period=parent_option.initiation_period)

        new_untrained_option_id = id(new_untrained_option)
        assert new_untrained_option_id != old_untrained_option_id, "Checking python references"
        assert id(new_untrained_option.parent) == old_untrained_option_id, "Checking python references"

        parent_option.children.append(new_untrained_option)

        return new_untrained_option

    def make_off_policy_updates_for_options(self, state, action, reward, next_state):
        for option in self.trained_options:  # type: Option
            option.off_policy_update(state, action, reward, next_state)

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
                                    exploration_strategy="shaping")
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
                                    exploration_strategy="shaping")
        self.agent_over_options = new_global_agent

    def _get_backward_options_in_gestation(self, state):
        for option in self.trained_options:
            if option.backward_option and option.get_training_phase() == "gestation" and \
                    option.is_init_true(state) and not option.is_term_true(state):
                return option
        return None

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

    def act(self, state, train_mode=True):
        # Query the global Q-function to determine which option to take in the current state
        option_idx = self.agent_over_options.act(state.features(), train_mode=train_mode)

        if train_mode:
            self.agent_over_options.update_epsilon()

        # Selected option
        selected_option = self.trained_options[option_idx]  # type: Option

        # TODO: Hack - If you satisfy the initiation condition for a backward option in gestation, take it
        backward_option = self._get_backward_options_in_gestation(state)
        if backward_option is not None:
            selected_option = backward_option

        # TODO: Hack - do not take option from a salient event targeting it
        if selected_option.is_term_true(state):
            selected_option = self.global_option

        return selected_option

    def update_policy_over_options_after_option_rollout(self, state_before_rollout, executed_option, option_transitions):
        """

		Args:
			state_before_rollout (State)
			executed_option (Option)
			option_transitions (list)

		Returns:

		"""

        state_after_rollout = self.get_next_state_from_experiences(option_transitions)

        # If we triggered the untrained option's termination condition, add to its buffer of terminal transitions
        for untrained_option in self.untrained_options:
            if untrained_option.is_term_true(state_after_rollout) and not untrained_option.is_term_true(state_before_rollout):
                untrained_option.final_transitions.append((state_before_rollout, executed_option.option_idx))
                untrained_option.terminal_states.append(state_after_rollout)

        # Add data to train Q(s, o)
        self.make_smdp_update(action=executed_option.option_idx, next_state=state_after_rollout, option_transitions=option_transitions)

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
        """ Iterate through all the skill chains and return all pairs of options that have intersecting initiation sets. """
        intersecting_pairs = []
        for chain in self.chains:  # type: SkillChain
            intersecting_options = chain.detect_intersection_with_other_chains(self.chains)
            if intersecting_options is not None:

                # Only care about intersections between forward chains
                if not intersecting_options[0].backward_option and not intersecting_options[1].backward_option:
                    intersecting_pairs.append(intersecting_options)

        return intersecting_pairs

    def get_current_salient_events(self):
        """ Get a list of states that satisfy final target events (i.e not init of an option) in the MDP so far. """
        return self.mdp.get_current_target_events()

    def create_all_backward_options_offline(self, forward_chain, back_chain):
        """
        Args:
            forward_chain (SkillChain)
            back_chain (SkillChain)

        Returns:
            created_options (list)
        """
        assert forward_chain.is_chain_completed(self.chains)
        assert not forward_chain.is_backward_chain

        parent = None

        # For each forward option, create the corresponding backward option
        for i, forward_option in enumerate(reversed(forward_chain.options)):  # type: int, Option

            if i == 0:
                back_option = Option(self.mdp, name=f"chain_{back_chain.chain_id}_backward_option",
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
                                     chain_id=len(self.chains),
                                     initialize_everywhere=True, max_num_children=1,
                                     writer=self.writer, device=self.device,
                                     dense_reward=self.dense_reward,
                                     use_warmup_phase=self.use_warmup_phase,
                                     update_global_solver=self.update_global_solver,
                                     is_backward_option=True,
                                     init_salient_event=back_chain.init_salient_event,
                                     target_salient_event=back_chain.target_salient_event)
            else:
                back_option = self.create_child_option(parent_option=parent)

            # Keep track of the current parent
            parent = back_option

            # Set the initiation classifier for this new option
            back_option.num_goal_hits = forward_option.num_goal_hits
            back_option.positive_examples = forward_option.positive_examples
            back_option.negative_examples = forward_option.negative_examples
            back_option.initiation_classifier = forward_option.initiation_classifier

            print(f"Set {back_option}'s training phase to {forward_option.get_training_phase()}")

            if self.pretrain_option_policies:
                back_option.initialize_with_global_solver()

            self.augment_agent_with_new_option(back_option, 0.)

            if back_chain.is_chain_completed(self.chains):
                break

        # Detect if the newly constructed backward chain is dangling
        # If it is, then we need to learn more options to complete that backward chain
        if not back_chain.is_chain_completed(self.chains):
            print(f"{back_chain} NOT completed - creating child options!")
            back_chain_leaf_node = back_chain.options[-1]  # type: Option
            new_untrained_back_options = self.create_children_options(option=back_chain_leaf_node)
            self.add_new_options_to_skill_chains(new_options=new_untrained_back_options)

        created_options = [option for option in back_chain.options if option.get_training_phase() == "initiation_done"]

        return created_options

    def manage_skill_chain_to_start_state(self, option):
        """
		When we complete creating a skill-chain targeting a salient event,
		we want to create a new skill chain that goes in the opposite direction.
		Args:
			option (Option)

		Returns:
			trained_back_options (list)

		"""
        chain = self.chains[option.chain_id - 1]  # type: SkillChain
        if chain.chained_till_start_state() and not chain.is_backward_chain \
                and not chain.has_backward_chain and self.create_backward_options:
            chain.has_backward_chain = True

            # 1. The new skill chain will chain back until it covers `start_states`
            # 2. The options in this new backward chain will use `start_predicate` as the
            #	 default initiation set during gestation
            start_states = chain.target_salient_event.trigger_points
            init_salient_event = chain.target_salient_event

            # These target predicates are used to determine the goal of the new skill chain
            target_salient_event = self.mdp.get_start_state_salient_event()

            # No need to check for intersections for backward chains
            new_chain = SkillChain(start_states=start_states, mdp_start_states=self.s0,
                                   target_salient_event=target_salient_event,
                                   init_salient_event=init_salient_event,
                                   options=[], chain_id=len(self.chains) + 1,
                                   intersecting_options=[], is_backward_chain=True,
                                   event_intersection_salience=False, option_intersection_salience=False)
            self.add_skill_chain(new_chain)

            if self.learn_backward_options_offline:
                trained_back_options = self.create_all_backward_options_offline(forward_chain=chain,
                                                                                back_chain=new_chain)
                return trained_back_options

            # Create a new option and equip the SkillChainingAgent with it
            new_option = Option(self.mdp, name=f"chain_{chain.chain_id}_backward_option",
                                global_solver=self.global_option.solver,
                                lr_actor=self.global_option.solver.actor_learning_rate,
                                lr_critic=self.global_option.solver.critic_learning_rate,
                                ddpg_batch_size=self.global_option.solver.batch_size,
                                subgoal_reward=self.subgoal_reward,
                                buffer_length=self.buffer_length,
                                classifier_type=self.classifier_type,
                                num_subgoal_hits_required=self.num_subgoal_hits_required,
                                seed=self.seed, parent=None, max_steps=self.max_steps,
                                enable_timeout=self.enable_option_timeout, chain_id=len(self.chains),
                                initialize_everywhere=True, max_num_children=1,
                                writer=self.writer, device=self.device, dense_reward=self.dense_reward,
                                use_warmup_phase=self.use_warmup_phase, update_global_solver=self.update_global_solver,
                                is_backward_option=True,
                                init_salient_event=init_salient_event,
                                target_salient_event=target_salient_event)

            if self.pretrain_option_policies:
                new_option.initialize_with_global_solver()
            self.augment_agent_with_new_option(new_option, 0.)
            self.untrained_options.append(new_option)

        # Since we did not learn any options offline, return the empty list
        return []

    def create_backward_skill_chain(self, start_event, target_event, create_root_option=True):
        """
		Create a skill chain that takes you from target_event to start_event
		Args:
			start_event (SalientEvent): Chain will continue until this is covered. This is also the
										default initiation set for backward options
			target_event (SalientEvent): Chain will try to hit this event
			create_root_option (bool)

		"""
        # No need to check for intersections if you are a backward chain
        back_chain = SkillChain(start_states=start_event.trigger_points,
                                mdp_start_states=self.s0,
                                target_salient_event=target_event,
                                init_salient_event=start_event,
                                options=[], chain_id=len(self.chains) + 1,
                                intersecting_options=[], is_backward_chain=True,
                                option_intersection_salience=False,
                                event_intersection_salience=False)
        self.add_skill_chain(back_chain)

        # Create a new option and equip the SkillChainingAgent with it
        if create_root_option:
            back_chain_root_option = Option(self.mdp, name=f"chain_{back_chain.chain_id}_backward_option",
                                            global_solver=self.global_option.solver,
                                            lr_actor=self.global_option.solver.actor_learning_rate,
                                            lr_critic=self.global_option.solver.critic_learning_rate,
                                            ddpg_batch_size=self.global_option.solver.batch_size,
                                            subgoal_reward=self.subgoal_reward,
                                            buffer_length=self.buffer_length,
                                            classifier_type=self.classifier_type,
                                            num_subgoal_hits_required=self.num_subgoal_hits_required,
                                            seed=self.seed, parent=None, max_steps=self.max_steps,
                                            enable_timeout=self.enable_option_timeout, chain_id=len(self.chains),
                                            initialize_everywhere=True, max_num_children=1,
                                            writer=self.writer, device=self.device, dense_reward=self.dense_reward,
                                            use_warmup_phase=self.use_warmup_phase,
                                            update_global_solver=self.update_global_solver,
                                            is_backward_option=True,
                                            init_salient_event=start_event,
                                            target_salient_event=target_event)

            if self.pretrain_option_policies:
                back_chain_root_option.initialize_with_global_solver()
            self.augment_agent_with_new_option(back_chain_root_option, 0.)
            self.untrained_options.append(back_chain_root_option)

        return back_chain

    def manage_intersecting_skill_chains(self):
        """
		When two forward chains intersect, we want to create a new chain that goes from
		their target salient events to the region of intersection.

		"""
        intersecting_options = self.get_intersecting_options()

        if intersecting_options:
            intersecting_options = intersecting_options[0]
            intersecting_chains = [self.chains[o.chain_id - 1] for o in intersecting_options if not o.backward_option]
            chain1, chain2 = intersecting_chains[0], intersecting_chains[1]

            if not chain1.has_backward_chain or not chain2.has_backward_chain:
                chain1.has_backward_chain = True
                chain2.has_backward_chain = True

                # Add this new salient event to the MDP - so that we can plan to and from it
                common_states = intersecting_options[1].effect_set
                all_salient_events = self.mdp.get_all_target_events_ever()
                all_salient_event_idx = [event.event_idx for event in all_salient_events]

                intersection_salient_event = LearnedSalientEvent(state_set=common_states,
                                                                 event_idx=max(all_salient_event_idx) + 1,
                                                                 intersection_event=True)

                self.mdp.add_new_target_event(intersection_salient_event)

                # Create the following skill-chains:
                # 1. Chain from beta1 to beta-intersection
                # 2. Chain from beta2 to beta-intersection
                # 3. Chain from beta-intersection to MDP start-state
                self.create_backward_skill_chain(start_event=chain1.target_salient_event,
                                                 target_event=intersection_salient_event)
                self.create_backward_skill_chain(start_event=chain2.target_salient_event,
                                                 target_event=intersection_salient_event)
                self.create_backward_skill_chain(start_event=intersection_salient_event,
                                                 target_event=self.mdp.get_start_state_salient_event())  # TODO: This has to be the init_salient_event of the long_chain

                self.rewire_intersecting_chains(intersecting_chains, intersecting_options, intersection_salient_event)

    def manage_intersection_between_option_and_events(self, option):
        """
		`option`'s initiation set completely contains the given `event`.
		Create an intersection learned salient event and then target it with a skill chain

		Args:
			option (Option)

		Returns:
			trained_backward_options (list)

		"""

        # Get chain corresponding to `option`
        option_chain = self.chains[option.chain_id - 1]  # type: SkillChain

        # Salient event corresponding to the current `option`
        option_target_event = option_chain.target_salient_event  # type: SalientEvent

        # Given just a freshly trained option, find all the events that the initiation set of the option completely engulfs
        intersecting_events = []

        for event in self.mdp.get_all_target_events_ever():  # type: SalientEvent
            event_chains = [chain for chain in self.chains if chain.target_salient_event == event and
                            chain.is_chain_completed(self.chains) and not chain.is_backward_chain]
            if event != option_target_event and \
                    len(event_chains) > 0 and \
                    SkillChain.detect_intersection_between_option_and_event(option, event):
                intersecting_events.append(event)

        trained_backward_options = []

        for event in intersecting_events:  # type: SalientEvent

            print(f"Found intersection between {option} and {event} - will create a backward chain and rewire {option_chain}")

            # Create a backward skill chain from the termination condition of the chain corresponding to `option` to the salient event
            if self.create_backward_options \
                    and not option_chain.has_backward_chain \
                    and not option_chain.is_backward_chain \
                    and option_chain.is_chain_completed(self.chains):

                back_chain = self.create_backward_skill_chain(start_event=option_chain.target_salient_event,
                                                              target_event=event,
                                                              create_root_option=(not self.learn_backward_options_offline))

                option_chain.has_backward_chain = True

                if self.learn_backward_options_offline:
                    learned_back_options = self.create_all_backward_options_offline(forward_chain=option_chain,
                                                                                    back_chain=back_chain)
                    trained_backward_options += learned_back_options

            # Set the `init_salient_event` of the chain corresponding to option to be the intersecting event
            option_chain.init_salient_event = event

        return trained_backward_options

    def rewire_intersecting_chains(self, intersecting_chains, intersecting_options, intersection_salient_event):
        """

		Args:
			intersecting_chains:
			intersecting_options:
			intersection_salient_event (SalientEvent)

		Returns:

		"""
        assert len(intersecting_options) == 2, intersecting_options
        assert len(intersecting_chains) == 2, intersecting_chains

        chain1, chain2 = intersecting_chains[0], intersecting_chains[1]  # type: SkillChain

        # Find the chain that chains until the start state of the MDP
        long_chain = chain1 if chain1.is_chain_completed(self.chains) else chain2

        # Find the chain that chains until the region of intersection
        short_chain = chain1 if long_chain == chain2 else chain2

        # Change the init_salient_event corresponding to the short_chain
        short_chain.init_salient_event = intersection_salient_event

        print(
            f"Set {short_chain}'s init event to {short_chain.init_salient_event}, while keeping {long_chain}'s init event as {long_chain.init_salient_event}")

    # Split the long chain into 2 parts: one that goes TO the intersection event and the other that goes to target_event
    # The split will occur at the intersection option that is part of the long chain
    # option1, option2 = intersecting_options[0], intersecting_options[1]  # type: Option

    # split_option = option1 if option1 in chain1.options else option2
    # split_index = get_split_idx(long_chain, split_option)
    # first_split = long_chain.options[:split_index]
    # second_split = short_chain.options[split_index:]
    #
    # # TODO: Create skills chains out of this...
    #
    # # Change the termination condition of options that target the `intersecting_options`
    # pass

    def conclude_option_initiation_phase(self, untrained_option, episode):
        """

		Args:
			untrained_option (Option)
			episode (int)

		Returns:
			completed_option (Option)
			should_create_children (bool)

		"""
        # In the skill-graph setting, we have to check if the current option's chain
        # is still accepting new options

        if self.should_create_more_options() and untrained_option.get_training_phase() == "initiation_done" \
                and self.chains[untrained_option.chain_id - 1].should_continue_chaining(self.chains):

            # We fix the learned option's initiation set and remove it from the list of target events
            self.untrained_options.remove(untrained_option)

            # Debug visualization
            plot_two_class_classifier(untrained_option, episode, self.experiment_name)

            return untrained_option, True

        elif untrained_option.get_training_phase() == "initiation_done":
            # Debug visualization
            plot_two_class_classifier(untrained_option, episode, self.experiment_name)

            # We fix the learned option's initiation set and remove it from the list of target events
            self.untrained_options.remove(untrained_option)

            return untrained_option, False

        return None, False

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
            # TODO: does this need to be run here?
            # plot_two_class_classifier(untrained_option, episode, self.experiment_name)

            # We fix the learned option's initiation set and remove it from the list of target events
            self.untrained_options.remove(untrained_option)

            return untrained_option

        return None

    def add_new_options_to_skill_chains(self, new_options):
        # Iterate through all the child options and add them to the skill tree 1 by 1
        for new_option in new_options:  # type: Option
            if new_option is not None:
                self.untrained_options.append(new_option)
                self.add_negative_examples(new_option)

                # If initialize_everywhere is True, also add to trained_options
                if new_option.initialize_everywhere:
                    if self.pretrain_option_policies:
                        new_option.initialize_with_global_solver()
                    self.augment_agent_with_new_option(new_option, self.init_q)

    def manage_skill_chain_after_option_rollout(self, *, state_before_rollout, executed_option,
                                                created_options, episode_number, option_transitions):
        """ After a single option rollout, this method will perform the book-keeping necessary to maintain skill-chain."""

        # Update Q(s, o) for the `agent_over_options`
        self.update_policy_over_options_after_option_rollout(state_before_rollout=state_before_rollout,
                                                             executed_option=executed_option,
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

            # if completed_option is not None:
            # 	self.mdp.satisfy_target_event(self.chains)

            if completed_option is not None and self.start_state_salience:
                trained_back_options = self.manage_skill_chain_to_start_state(completed_option)
                created_options += trained_back_options

            if completed_option is not None and self.event_intersection_salience:
                trained_back_options = self.manage_intersection_between_option_and_events(completed_option)
                created_options += trained_back_options

            should_create_children = self.should_create_children_options(completed_option)

            if should_create_children:
                new_options = self.create_children_options(completed_option)
                self.add_new_options_to_skill_chains(new_options)

            if completed_option is not None:
                created_options.append(completed_option)

        if self.option_intersection_salience:
            self.manage_intersecting_skill_chains()

        return created_options

    def dsc_rollout(self, *, episode_number, step_number, interrupt_handle=lambda x: False):
        """ Single episode rollout of deep skill chaining from the current state in the MDP. """

        score = 0.

        state = deepcopy(self.mdp.cur_state)
        created_options = []

        while step_number < self.max_steps:

            # Pick an option
            selected_option = self.act(state)

            # Execute it in the MDP
            option_transitions, reward = selected_option.execute_option_in_mdp(self.mdp,
                                                                               episode_number,
                                                                               step_number)

            # Logging
            score += reward
            step_number += len(option_transitions)
            state = self.get_next_state_from_experiences(option_transitions)

            # DSC book keeping
            created_options = self.manage_skill_chain_after_option_rollout(state_before_rollout=state,
                                                                           executed_option=selected_option,
                                                                           option_transitions=option_transitions,
                                                                           episode_number=episode_number,
                                                                           created_options=created_options)

            if state.is_terminal() or interrupt_handle(state):
                break

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

    def _log_dqn_status(self, episode, last_10_scores, last_10_durations):

        print('\rEpisode {}\tAverage Score: {:.2f}\tDuration: {:.2f} steps\tOP Eps: {:.2f}'.format(
            episode, np.mean(last_10_scores), np.mean(last_10_durations), self.agent_over_options.epsilon))

        self.num_options_history.append(len(self.trained_options))

        if self.writer is not None:
            self.writer.add_scalar("Episodic scores", last_10_scores[-1], episode)

        if episode % 10 == 0:
            print('\rEpisode {}\tAverage Score: {:.2f}\tDuration: {:.2f} steps\tOP Eps: {:.2f}'.format(
                episode, np.mean(last_10_scores), np.mean(last_10_durations), self.agent_over_options.epsilon))

        if episode > 0 and episode % 100 == 0:
            # eval_score, trajectory = self.trained_forward_pass(render=False)
            eval_score, trajectory = 0., []

            self.validation_scores.append(eval_score)
            print("\rEpisode {}\tValidation Score: {:.2f}".format(episode, eval_score))

        if self.plotter is not None and self.generate_plots and episode % 10 == 0 and episode > 0:
            self.plotter.generate_episode_plots(self, episode)

    def save_all_models(self):
        for option in self.trained_options:  # type: Option
            save_model(option.solver, -1, self.log_dir, best=False)

    def save_all_scores(self, pretrained, scores, durations):
        print("\rSaving training and validation scores..")
        training_scores_file_name = "sc_pretrained_{}_training_scores_{}.pkl".format(pretrained, self.seed)
        training_durations_file_name = "sc_pretrained_{}_training_durations_{}.pkl".format(pretrained, self.seed)
        validation_scores_file_name = "sc_pretrained_{}_validation_scores_{}.pkl".format(pretrained, self.seed)
        num_option_history_file_name = "sc_pretrained_{}_num_options_per_epsiode_{}.pkl".format(pretrained, self.seed)

        # TODO: Fix this
        # if self.log_dir:
        #     training_scores_file_name = os.path.join(self.log_dir, training_scores_file_name)
        #     training_durations_file_name = os.path.join(self.log_dir, training_durations_file_name)
        #     validation_scores_file_name = os.path.join(self.log_dir, validation_scores_file_name)
        #     num_option_history_file_name = os.path.join(self.log_dir, num_option_history_file_name)

        with open(training_scores_file_name, "wb+") as _f:
            pickle.dump(scores, _f)
        with open(training_durations_file_name, "wb+") as _f:
            pickle.dump(durations, _f)
        with open(validation_scores_file_name, "wb+") as _f:
            pickle.dump(self.validation_scores, _f)
        with open(num_option_history_file_name, "wb+") as _f:
            pickle.dump(self.num_options_history, _f)

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


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment_name", type=str, help="Experiment Name")
    parser.add_argument("--device", type=str, help="cpu/cuda:0/cuda:1")
    parser.add_argument("--env", type=str, help="name of gym environment", default="Pendulum-v0")
    parser.add_argument("--pretrained", action="store_true", help="whether or not to load pretrained options", default=False)
    parser.add_argument("--seed", type=int, help="Random seed for this run (default=0)", default=0)
    parser.add_argument("--episodes", type=int, help="# episodes", default=200)
    parser.add_argument("--steps", type=int, help="# steps", default=1000)
    parser.add_argument("--subgoal_reward", type=float, help="SkillChaining subgoal reward", default=0.)
    parser.add_argument("--lr_a", type=float, help="DDPG Actor learning rate", default=1e-4)
    parser.add_argument("--lr_c", type=float, help="DDPG Critic learning rate", default=1e-3)
    parser.add_argument("--ddpg_batch_size", type=int, help="DDPG Batch Size", default=64)
    parser.add_argument("--render", action="store_true", help="Render the mdp env", default=False)
    parser.add_argument("--option_timeout", action="store_true", help="Whether option times out at 200 steps", default=False)
    parser.add_argument("--generate_plots", action="store_true", help="Whether or not to generate plots", default=False)
    parser.add_argument("--tensor_log", action="store_true", help="Enable tensorboard logging", default=False)
    parser.add_argument("--control_cost", action="store_true", help="Penalize high actuation solutions", default=False)
    parser.add_argument("--dense_reward", action="store_true", help="Use dense/sparse rewards", default=False)
    parser.add_argument("--max_num_options", type=int, help="Max number of options we can learn", default=5)
    parser.add_argument("--num_subgoal_hits", type=int, help="Number of subgoal hits to learn an option", default=3)
    parser.add_argument("--buffer_len", type=int, help="buffer size used by option to create init sets", default=20)
    parser.add_argument("--classifier_type", type=str, help="ocsvm/elliptic for option initiation clf", default="ocsvm")
    parser.add_argument("--init_q", type=str, help="compute/zero", default="zero")
    parser.add_argument("--use_smdp_update", action="store_true", help="sparse/SMDP update for option policy", default=False)
    parser.add_argument("--use_start_state_salience", action="store_true", default=False)
    parser.add_argument("--use_option_intersection_salience", action="store_true", default=False)
    parser.add_argument("--use_event_intersection_salience", action="store_true", default=False)
    parser.add_argument("--pretrain_option_policies", action="store_true", default=False)
    parser.add_argument("--create_backward_options", action="store_true", default=False)
    parser.add_argument("--fixed_epsilon", action="store_true", help="Use fixed epsilon or decreasing epsilon for DDPG", default=False)
    args = parser.parse_args()

    if args.env == "point-reacher":
        from simple_rl.tasks.point_reacher.PointReacherMDPClass import PointReacherMDP

        overall_mdp = PointReacherMDP(seed=args.seed, dense_reward=args.dense_reward, render=args.render)
        state_dim = 6
        action_dim = 2
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
    elif "sawyer" in args.env.lower():
        from simple_rl.tasks.leap_wrapper.LeapWrapperMDPClass import LeapWrapperMDP
        from simple_rl.tasks.leap_wrapper.LeapWrapperPlotter import LeapWrapperPlotter
        overall_mdp = LeapWrapperMDP(dense_reward=args.dense_reward, render=args.render)
        mdp_plotter = LeapWrapperPlotter("sawyer", args.experiment_name, overall_mdp)
        overall_mdp.env.seed(args.seed)
    else:
        from simple_rl.tasks.gym.GymMDPClass import GymMDP

        overall_mdp = GymMDP(args.env, render=args.render)
        state_dim = overall_mdp.env.observation_space.shape[0]
        action_dim = overall_mdp.env.action_space.shape[0]
        overall_mdp.env.seed(args.seed)

    print("Training skill chaining agent from scratch with a subgoal reward {}".format(args.subgoal_reward))
    print("MDP InitState = ", overall_mdp.init_state)

    q0 = 0. if args.init_q == "zero" else None

    chainer = SkillChaining(overall_mdp, args.steps, args.lr_a, args.lr_c, args.ddpg_batch_size,
                            seed=args.seed, subgoal_reward=args.subgoal_reward,
                            num_subgoal_hits_required=args.num_subgoal_hits,
                            enable_option_timeout=args.option_timeout, init_q=q0, use_full_smdp_update=args.use_smdp_update,
                            generate_plots=args.generate_plots, tensor_log=args.tensor_log, device=args.device,
                            buffer_length=args.buffer_len,
                            start_state_salience=args.use_start_state_salience,
                            option_intersection_salience=args.use_option_intersection_salience,
                            event_intersection_salience=args.use_event_intersection_salience,
                            pretrain_option_policies=args.pretrain_option_policies,
                            create_backward_options=args.create_backward_options,
                            dense_reward=args.dense_reward,
                            experiment_name=args.experiment_name,
                            plotter=mdp_plotter,
                            fixed_epsilon=args.fixed_epsilon)
    episodic_scores, episodic_durations = chainer.skill_chaining_run_loop(num_episodes=args.episodes, num_steps=args.steps, to_reset=True)

    # Log performance metrics
    # chainer.save_all_models()
    # Used to be chainer.perform_experiments() --Kiran
    # chainer.plotter.generate_experiment_plots(chainer)
    chainer.save_all_scores(args.pretrained, episodic_scores, episodic_durations)
