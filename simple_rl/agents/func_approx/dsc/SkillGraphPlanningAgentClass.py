import ipdb
import numpy as np
from tqdm import tqdm
from copy import deepcopy
from simple_rl.mdp.StateClass import State
from simple_rl.tasks.point_reacher.PointReacherMDPClass import PointReacherMDP
from simple_rl.agents.func_approx.dsc.SkillChainingAgentClass import SkillChaining
from simple_rl.agents.func_approx.dsc.ChainClass import SkillChain
from simple_rl.agents.func_approx.dsc.OptionClass import Option
from simple_rl.agents.func_approx.dsc.GraphSearchClass import GraphSearch
from simple_rl.agents.func_approx.dsc.SalientEventClass import SalientEvent


class SkillGraphPlanningAgent(object):
    def __init__(self, mdp, chainer, experiment_name, seed):
        """
        The Skill Graph Planning Agent uses graph search to get to target states given
        during test time. If the goal state is in a salient event already known to the agent,
        it can get there without any learning involved. If it is not a known salient event,
        it will create new skill chains to reliably target it.
        Args:
            mdp (PointReacherMDP)
            chainer (SkillChaining)
            experiment_name (str)
            seed (int)
        """
        self.mdp = mdp
        self.chainer = chainer
        self.experiment_name = experiment_name
        self.seed = seed

        self.plan_graph = self.construct_plan_graph()

    def construct_plan_graph(self):
        # Perform test rollouts to determine edge weights in the plan graph
        self.perform_test_rollouts(num_rollouts=500)

        # Construct the plan graph
        plan_graph = GraphSearch()
        plan_graph.construct_graph(self.chainer.chains)

        # Visualize the graph
        file_name = f"value_function_plots/{self.experiment_name}/plan_graph_seed_{self.seed}.png"
        plan_graph.visualize_plan_graph(file_name)

        return plan_graph

    def act(self, start_state, goal_state):
        """
        Given a start state `s` and a goal state `sg` pick an option to execute.
        Use planning if possible, else revert to the trained policy over options.
        Args:
            start_state (State)
            goal_state (State or np.ndarray)

        Returns:
            option (Option): Option to execute in the ground MDP
        """
        if self.plan_graph.does_path_exist(start_state, goal_state):
            plan = self.plan_graph.get_path_to_execute(start_state, goal_state)
            if len(plan) > 0:
                admissible_options = [o for o in plan if o.is_init_true(start_state) and not o.is_term_true(start_state)]
                if len(admissible_options) > 0:
                    print(f"Got plan {plan} and chose to execute {admissible_options[0]}")
                    return admissible_options[0]  # type: Option
        return self.chainer.act(start_state, train_mode=False)

    def get_salient_event_for_state(self, state):
        """
        If `state` is inside a known salient event, return that salient event. Else, return None.
        Args:
            state (State)

        Returns:
            salient_event (SalientEvent or None)
        """
        all_salient_events = self.chainer.get_chained_salient_events()
        satisfied_existing_salient_events = [event for event in all_salient_events if event(state)]

        # If the goal state falls within a known salient event, use that one,
        # else create a new salient event and add it to the list of events in the MDP
        if len(satisfied_existing_salient_events) > 0:
            target_salient_event = satisfied_existing_salient_events[0]
            return target_salient_event

        return None

    def is_state_in_known_salient_event(self, state):
        return self.get_salient_event_for_state(state) is not None

    def get_goal_salient_event(self, goal_state):
        """ If there is no salient event already corresponding to the given goal_state, create one. """

        all_salient_events = self.chainer.get_chained_salient_events()
        target_salient_event = self.get_salient_event_for_state(goal_state)

        if target_salient_event is not None:
            created_new_event = False
        else:
            event_indices = [event.event_idx for event in all_salient_events]
            target_salient_event = SalientEvent(goal_state, event_idx=max(event_indices) + 1)
            self.mdp.add_new_target_event(target_salient_event)
            created_new_event = True

        return target_salient_event, created_new_event

    def should_planning_run_loop_terminate(self, state, goal_state, goal_salient_event, target_option, created_test_option):
        """
        There are 3 possible cases:
        1. When the goal_state is a salient event we already know how to get to:
           In this case, we only want to terminate when we reach the goal_state
        2. When the goal_state is not a salient event we know how to get to, but is inside the initiation
           set of some option that we have already learned:
           In this case, we want to terminate when we reach the option that contains that goal_state, but
           not execute the option itself. Later, we will learn a new option that will drive us to that goal state.
        3. When the goal_state is not a salient event we know about AND it is not inside the initiation set of
           any of the options that we have learned:
           In this case, we have to find the option whose initiation set is "closest" to the goal state.
           We can either use a pre-specified measure (eg euclidean distance) or use the space of option value functions
           to find the "closest" option.
        Args:
            state (State)
            goal_state (State)
            goal_salient_event (SalientEvent)
            target_option (Option)
            created_test_option (bool): once we have already created the test-time option, we no longer
                                        have to terminate when we reach its initiation set. We can take
                                        the test-time option to reach the goal-salient-event

        Returns:
            should_terminate (bool)
        """

        # Case 1: goal_state satisfies one of the salient events we already know about
        if self.is_state_in_known_salient_event(goal_state):
            return goal_salient_event(state)

        # Case 2: goal_state is inside the initiation set classifier of some learned option
        if target_option is not None and not created_test_option:
            return target_option.is_init_true(state)

        # Case 3: goal_state is not in salient event or inside some option's initiation set classifier
        # Will have to grab the nearest option - poll each option's value function for that
        raise NotImplementedError("Case 3")

    def planning_run_loop(self, *, goal_state, start_state=None):
        """
        This is a test-time run loop that uses planning to select options when it can.
        Args:
            goal_state (State)
            start_state (State)

        Returns:
            accumulated_reward (float)
            executed_options (list)
            reached_goal (bool)
        """

        self.reset(start_state)

        state = deepcopy(self.mdp.cur_state)
        overall_reward = 0.
        num_steps = 0
        option_trajectories = []
        should_terminate_run_loop = False

        goal_salient_event, created_new_salient_event = self.get_goal_salient_event(goal_state)

        # When going in the forward direction, we will target this option if the goal is not in a known salient event
        forward_target_option = self.get_target_option(goal_state, backward_direction=False)

        created_test_time_option = False

        while not should_terminate_run_loop and num_steps < self.chainer.max_steps:

            selected_option = self.act(state, goal_state)

            print(f"Executing {selected_option}")

            option_reward, next_state, num_steps, option_trajectory = selected_option.trained_option_execution(self.mdp, num_steps)
            overall_reward += option_reward

            if selected_option.is_term_true(next_state):
                print(f"Successfully executed {selected_option}")

            # Modify the edge weight associated with the executed option
            if selected_option in self.plan_graph.option_nodes:  # type: Option
                for child_option in selected_option.children:  # type: Option or SalientEvent
                    if child_option is not None:
                        # TODO: We want to do a moving average filter so as to not wipe off the optimism

                        if isinstance(child_option, Option):
                            child_option_success_rate = child_option.get_option_success_rate()
                            self.plan_graph.set_edge_weight(child_option, selected_option, weight=1./child_option_success_rate)
                        elif isinstance(child_option, SalientEvent):
                            self.plan_graph.set_edge_weight(child_option, selected_option, weight=0.)
                        else:
                            raise ValueError(child_option, type(child_option))

            # option_state_trajectory is a list of (o, s) tuples
            option_trajectories.append(option_trajectory)

            state = next_state

            should_terminate_run_loop = self.should_planning_run_loop_terminate(state, goal_state, goal_salient_event,
                                                                                forward_target_option, created_test_time_option)

            if should_terminate_run_loop and not goal_salient_event(state):  # Case II or Case III

                if forward_target_option is not None and created_new_salient_event and not created_test_time_option:  # Case II
                    self.create_test_time_forward_goal_option(forward_target_option, goal_salient_event)
                    created_test_time_option = True

                    # Visualize the graphs
                    file_name = f"value_function_plots/{self.experiment_name}/plan_graph_seed_{self.seed}_2.png"
                    self.plan_graph.visualize_plan_graph(file_name)

        return overall_reward, option_trajectories

    def get_target_option(self, goal_state, backward_direction):  # TODO: This needs to be different when searching for a forward or backward option

        # Grab all the completed skill chains that contain the goal_state
        chains_with_goal_state = [chain for chain in self.chainer.chains if
                                  chain.state_in_chain(goal_state) and
                                  chain.chained_till_start_state() and
                                  chain.is_backward_chain == backward_direction]

        # Select the option (options created earlier have a higher priority)
        options_with_goal_state = [chain.get_option_for_state(goal_state) for chain in chains_with_goal_state]
        if len(options_with_goal_state) > 0:  # TODO: How to pick an option from this list of feasible choices?
            target_option = options_with_goal_state[0]
            return target_option

        return None

    def create_test_time_forward_goal_option(self, train_goal_option, target_salient_event):
        """
        1. We are going to create a new option whose initiation set is the same as the `train_goal_option`.
        2. We need to augment the skill-chaining agent with this new option.
        3. We also need to add this new option to the plan-graph, where it will have the same incoming edges
           as the `train_goal_option`, but an outgoing edge to the `target_salient_event`.
           This outgoing edge should be optimistically initialized to have a success-rate of 1.
        4. We somehow need to initialize this option to already be in the `initiation_done` phase of learning.
        5. Finally, we need to pre-train this option's policy using the data in `train_goal_option`'s
           replay buffer (but using the new option's option-reward function).
        Args:
            train_goal_option (Option)
            target_salient_event (SalientEvent)

        Returns:
            new_option (Option)
        """

        # Create a new chain
        init_salient_event = train_goal_option.is_init_true  # TODO: Create a Salient Event out of this
        start_state = train_goal_option.terminal_states[0]
        chain_id = max([chain.chain_id for chain in self.chainer.chains]) + 1
        new_forward_chain = SkillChain(start_states=[start_state], mdp_start_states=[self.mdp.init_state],
                                       init_salient_event=init_salient_event,
                                       target_salient_event=target_salient_event, is_backward_chain=False,
                                       chain_until_intersection=False, chain_id=chain_id, options=[])

        if new_forward_chain not in self.chainer.chains:
            assert new_forward_chain not in self.chainer.chains
            self.chainer.add_skill_chain(new_forward_chain)

        # Create a new option
        name = f"goal_option_{target_salient_event.event_idx}"
        new_untrained_option = Option(self.mdp, name=name, global_solver=self.chainer.global_option.solver,
                                      lr_actor=train_goal_option.solver.actor_learning_rate,
                                      lr_critic=train_goal_option.solver.critic_learning_rate,
                                      ddpg_batch_size=train_goal_option.solver.batch_size,
                                      subgoal_reward=self.chainer.subgoal_reward,
                                      buffer_length=self.chainer.buffer_length,
                                      classifier_type=self.chainer.classifier_type,
                                      num_subgoal_hits_required=self.chainer.num_subgoal_hits_required,
                                      seed=self.seed, parent=None, max_steps=self.chainer.max_steps,
                                      enable_timeout=self.chainer.enable_option_timeout,
                                      chain_id=chain_id,
                                      writer=self.chainer.writer, device=self.chainer.device,
                                      dense_reward=self.chainer.dense_reward,
                                      initialize_everywhere=train_goal_option.initialize_everywhere,
                                      max_num_children=train_goal_option.max_num_children,
                                      is_backward_option=False,
                                      init_salient_event=train_goal_option.init_salient_event,
                                      target_salient_event=target_salient_event)

        print(f"Created {new_untrained_option} targeting {target_salient_event}")

        # Set the initiation classifier for this new option
        new_untrained_option.num_goal_hits = train_goal_option.num_goal_hits
        new_untrained_option.positive_examples = train_goal_option.positive_examples
        new_untrained_option.negative_examples = train_goal_option.negative_examples
        new_untrained_option.initiation_classifier = train_goal_option.initiation_classifier

        print(f"Set {new_untrained_option}'s training phase to {new_untrained_option.get_training_phase()}")

        # Augment the DSC agent with the new option and pre-train it's policy
        new_untrained_option.initialize_option_ddpg_with_restricted_support()
        new_untrained_option.solver.epsilon = train_goal_option.solver.epsilon
        self.chainer.augment_agent_with_new_option(new_untrained_option, 0.)

        # Add to the plan graph
        print(f"Adding node {new_untrained_option} to the plan-graph")
        self.plan_graph.add_node(new_untrained_option)
        self.plan_graph.add_node(target_salient_event)

        # Grab all the children of the train_goal_option and make those the children of the new option as well
        in_edges = list(self.plan_graph.plan_graph.in_edges(train_goal_option))
        child_options = [pair[0] for pair in in_edges]
        for child_option in child_options:  # type: Option
            print(f"Adding edge from {child_option} to {new_untrained_option} in plan-graph")
            self.plan_graph.add_edge(child_option, new_untrained_option)
        new_untrained_option.children = child_options

        # Add an outgoing edge from the new option to the target salient event
        print(f"Adding edge from {new_untrained_option} to {target_salient_event} in plan-graph")
        self.plan_graph.add_edge(new_untrained_option, target_salient_event, edge_weight=1./new_untrained_option.get_option_success_rate())

    def perform_test_rollouts(self, num_rollouts):
        """ Perform test rollouts to collect statistics on option success rates.
            We can then use those success rates to set edge weights in our
            abstract skill graph.
        """
        summed_reward = 0.
        for _ in tqdm(range(num_rollouts), desc="Performing DSC Test Rollouts"):
            r, _ = self.chainer.trained_forward_pass(render=False)
            summed_reward += r
        return summed_reward

    def reset(self, start_state=None):
        """ Reset the MDP to either the default start state or to the specified one. """
        self.mdp.reset()

        if start_state is not None:
            start_position = start_state.position if isinstance(start_state, State) else start_state[:2]
            self.mdp.set_xy(start_position)

    def measure_success(self, goal_state, start_state=None, goal_tol=0.6):
        successes = 0
        for _ in range(100):
            self.planning_run_loop(goal_state=goal_state, start_state=start_state)
            goal_pos = goal_state
            is_terminal = np.linalg.norm(self.mdp.cur_state.position - goal_pos) <= goal_tol
            if is_terminal: successes += 1
        return successes
