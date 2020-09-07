# Python imports.
from __future__ import print_function

import ipdb
import itertools
import random
from copy import deepcopy
import numpy as np
import torch
from scipy.spatial import distance
from sklearn import svm
from tqdm import tqdm

from simple_rl.agents.func_approx.ddpg.DDPGAgentClass import DDPGAgent
from simple_rl.agents.func_approx.dsc.utils import Experience
from simple_rl.agents.func_approx.td3.TD3AgentClass import TD3
from simple_rl.mdp.GoalDirectedMDPClass import GoalDirectedMDP
from simple_rl.mdp.StateClass import State


class Option(object):
    fixed_epsilon = False
    constant_noise = False

    def __init__(self, overall_mdp, name, global_solver, lr_actor, lr_critic, ddpg_batch_size, classifier_type="ocsvm", subgoal_reward=0.,
                 max_steps=20000, seed=0, parent=None, num_subgoal_hits_required=3, buffer_length=20, dense_reward=False,
                 enable_timeout=True, timeout=200, initiation_period=5, option_idx=None, chain_id=None, initialize_everywhere=True,
                 max_num_children=3, init_salient_event=None, target_salient_event=None, update_global_solver=False, use_warmup_phase=True,
                 gestation_init_predicates=None, is_backward_option=False, solver_type="ddpg", device=torch.device("cpu"), writer=None):
        """
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
            device (torch.device)
            writer (SummaryWriter)
        """
        if gestation_init_predicates is None:
            gestation_init_predicates = []
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

        self.state_size = self.overall_mdp.state_space_size()
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

        self.global_solver = global_solver
        self.solver = None

        if self.name == "global_option":
            self.reset_global_solver()
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

        self.final_transitions = []
        self.terminal_states = []
        self.children = []

        # Debug member variables
        self.num_executions = 0
        self.num_on_policy_goal_hits = 0
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

    def _get_epsilon_greedy_epsilon(self):
        if self.constant_noise:
            if "point" in self.overall_mdp.env_name:
                return 0.1
            elif "ant" in self.overall_mdp.env_name:
                return 0.25
            elif "sawyer" in self.overall_mdp.env_name:
                return 0.25
            else:
                raise NotImplementedError(f"Epsilon not defined for {self.overall_mdp.env_name}")
        return 0.

    def act(self, state, eval_mode, warmup_phase):
        """ Epsilon greedy action selection when in training mode. """

        if warmup_phase and self.use_warmup_phase:
            return self.overall_mdp.sample_random_action()

        if random.random() < self._get_epsilon_greedy_epsilon() and not eval_mode:
            return self.overall_mdp.sample_random_action()
        return self.solver.act(state.features(), evaluation_mode=eval_mode)

    def reset_global_solver(self):
        if self.solver_type == "ddpg":
            self.global_solver = DDPGAgent(self.state_size, self.action_size, self.seed, self.device, self.lr_actor, self.lr_critic,
                                           self.ddpg_batch_size, tensor_log=(self.writer is not None), writer=self.writer,
                                           name="global_option", fixed_epsilon=self.fixed_epsilon)
        elif self.solver_type == "td3":
            self.global_solver = TD3(state_dim=self.state_size, action_dim=self.action_size,
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
            exploration = "shaping" if self.name == "global_option" else ""
            self.solver = DDPGAgent(self.state_size, self.action_size, seed, device, lr_actor, lr_critic, ddpg_batch_size,
                                    tensor_log=(writer is not None), writer=writer, name=solver_name, exploration=exploration,
                                    fixed_epsilon=self.fixed_epsilon)
        elif _type == "td3":
            exploration_method = "shaping" if self.name == "global_option" else ""
            self.solver = TD3(state_dim=self.state_size, action_dim=self.action_size,
                              max_action=self.overall_mdp.env.action_space.high[0],
                              device=self.device, exploration_method=exploration_method)

        else:
            raise NotImplementedError(self.solver_type)

    def sample_state(self):
        """ Return a state from the option's initiation set. """
        if self.get_training_phase() != "initiation_done":
            return None

        sampled_state = None
        num_tries = 0

        while sampled_state is None and num_tries < 50:
            if isinstance(self.solver, DDPGAgent):
                if len(self.solver.replay_buffer) > 0:
                    sampled_experience = random.choice(self.solver.replay_buffer.memory)
                elif len(self.experience_buffer) > 0:
                    sampled_experience = random.choice(self.experience_buffer)[0].serialize()
                else:
                    continue
                sampled_state = sampled_experience[0] if self.is_init_true(sampled_experience[0]) else None
            elif isinstance(self.solver, TD3):
                sampled_idx = random.randint(0, len(self.solver.replay_buffer) - 1)
                sampled_experience = self.solver.replay_buffer[sampled_idx]
                sampled_state = sampled_experience[0] if self.is_init_true(sampled_experience[0]) else None
            num_tries += 1
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

    def initialize_with_global_solver(self, reset_option_buffer=True):
        """" Initialize option policy - unrestricted support because not checking if I_o(s) is true. """
        # Not using off_policy_update() because we have numpy arrays not state objects here
        if not self.pretrained_option_policy:  # Make sure that we only do this once
            for state, action, reward, next_state, done in tqdm(self.global_solver.replay_buffer,
                                                                desc=f"Initializing {self.name} policy"):
                if self.is_term_true(state):
                    continue
                if self.is_term_true(next_state):
                    # TODO: probably can delete this case but will check later
                    self.solver.step(state, action, self.subgoal_reward, next_state, True)
                else:
                    subgoal_reward = self.get_subgoal_reward(next_state)
                    self.solver.step(state, action, subgoal_reward, next_state, done)

            # To keep the support of the option policy restricted to the initiation region
            # of the option, we can clear the option's replay buffer after we have pre-trained its policy
            if reset_option_buffer:
                self.solver.replay_buffer.clear()

            self.pretrained_option_policy = True

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
                # TODO: Kshitij deleted
                # state_matrix = state_matrix[:, :2]

                # When we treat the start state as a salient event, we pass in the
                # init predicate that we are going to use during gestation
                if self.init_salient_event is not None:
                    return self.init_salient_event(state_matrix)

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

        # print('using svm for predicting if in initiation set.')
        features = ground_state.features() if isinstance(ground_state, State) else ground_state
        return self.initiation_classifier.predict([features])[0] == 1

    def is_term_true(self, ground_state):
        if self.parent is not None:
            return self.parent.is_init_true(ground_state)

        if self.target_salient_event is not None:
            return self.target_salient_event(ground_state)

        # If option does not have a parent, it must be the goal option or the global option
        if self.name == "global_option" or self.name == "exploration_option":
            return self.overall_mdp.is_goal_state(ground_state)

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
        ipdb.set_trace()

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
        # TODO: Kshitij needs to fix this
        position_vector = state.features() if isinstance(state, State) else state

        # For global and parent option, we use the negative distance to the goal state
        if self.parent is None:
            distance_to_goal = self.target_salient_event.distance_from_goal(position_vector)
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

            if self.is_term_true(state) or state.is_terminal():
                if step_number == 0 or num_steps == 0:
                    # failed to reset the option after reaching a terminal state
                    ipdb.set_trace()

            while not self.is_term_true(state) and not state.is_terminal() and \
                    step_number < self.max_steps and num_steps < self.timeout:

                warmup_phase = self.get_training_phase() == "gestation"
                action = self.act(state, eval_mode=eval_mode, warmup_phase=warmup_phase)
                reward, next_state = mdp.execute_agent_action(action, option_idx=self.option_idx)

                if not warmup_phase or (self.name == "global_option" and self.update_global_solver):
                    self.update_option_solver(state, action, reward, next_state)
                    self.solver.update_epsilon()

                if self.name != "global_option" and self.update_global_solver:
                    self.global_solver.step(state.features(), action, reward, next_state.features(), next_state.is_terminal())
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

            # TODO: When initialize_everywhere is true, self.train() needs to be called from the OptionClass and not from SkillChainingClass

            if self.is_term_true(state) and self.last_episode_term_triggered != episode:  # and self.is_valid_init_data(visited_states):
                self.num_goal_hits += 1
                self.num_on_policy_goal_hits += 1
                self.last_episode_term_triggered = episode
                self.effect_set.append(state)

            if self.parent is None and self.is_term_true(state) and self.target_salient_event is not None:
                print(f"[{self}] Adding {state.position} to {self.target_salient_event}'s trigger points")
                self.target_salient_event.trigger_points.append(state)

            if self.name != "global_option" and self.get_training_phase() != "initiation_done":
                self.refine_initiation_set_classifier(visited_states, start_state, state, num_steps, step_number)

            return option_transitions, total_reward

        raise Warning("Wanted to execute {}, but initiation condition not met".format(self))

    def is_eligible_for_off_policy_triggers(self):
        if "point" in self.overall_mdp.env_name:
            eligible_phase = self.get_training_phase() == "gestation"
        elif "ant" in self.overall_mdp.env_name:
            eligible_phase = self.get_training_phase() != "initiation_done"
        elif "sawyer" in self.overall_mdp.env_name:
            eligible_phase = self.get_training_phase() != "initiation_done"
        else:
            raise NotImplementedError(self.overall_mdp)

        return eligible_phase and not self.backward_option

    def trigger_termination_condition_off_policy(self, trajectory, episode):
        trajectory_so_far = []
        states_so_far = []

        for state, action, reward, next_state in trajectory:  # type: State, np.ndarray, float, State
            trajectory_so_far.append((state, action, reward, next_state))
            states_so_far.append(state)

            if self.is_term_true(next_state) and self.is_valid_init_data(states_so_far) and \
                    episode != self.last_episode_term_triggered and self.is_eligible_for_off_policy_triggers():

                print(f"Triggered termination condition for {self}")

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

        while not self.is_term_true(state) and not state.is_terminal() \
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
            return self.num_on_policy_goal_hits / self.num_executions
        return 1.
