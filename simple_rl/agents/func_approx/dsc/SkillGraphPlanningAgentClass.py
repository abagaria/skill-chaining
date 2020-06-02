import ipdb
import itertools
import numpy as np
from tqdm import tqdm
from copy import deepcopy
from simple_rl.mdp.StateClass import State
from simple_rl.tasks.point_reacher.PointReacherMDPClass import PointReacherMDP
from simple_rl.agents.func_approx.dsc.SkillChainingAgentClass import SkillChaining
from simple_rl.agents.func_approx.dsc.ChainClass import SkillChain
from simple_rl.agents.func_approx.dsc.OptionClass import Option
from simple_rl.agents.func_approx.dsc.GraphSearchClass import GraphSearch
from simple_rl.agents.func_approx.dsc.SalientEventClass import SalientEvent, DSCOptionSalientEvent
from simple_rl.agents.func_approx.exploration.UCBActionSelectionAgentClass import UCBActionSelectionAgent
from simple_rl.agents.func_approx.dsc.utils import make_chunked_value_function_plot, visualize_graph


class SkillGraphPlanningAgent(object):
    def __init__(self, *, mdp, chainer, pretrain_option_policies, initialize_graph, use_ucb, experiment_name, seed):
        """
        The Skill Graph Planning Agent uses graph search to get to target states given
        during test time. If the goal state is in a salient event already known to the agent,
        it can get there without any learning involved. If it is not a known salient event,
        it will create new skill chains to reliably target it.
        Args:
            mdp (PointReacherMDP)
            chainer (SkillChaining)
            pretrain_option_policies (bool)
            initialize_graph (bool)
            use_ucb (bool)
            experiment_name (str)
            seed (int)
        """
        self.mdp = mdp
        self.chainer = chainer
        self.pretrain_option_policies = pretrain_option_policies
        self.initialize_graph = initialize_graph
        self.use_ucb = use_ucb
        self.experiment_name = experiment_name
        self.seed = seed

        self.bandit_agents = {}

        self.plan_graph = self.construct_plan_graph() if initialize_graph else GraphSearch()

    def construct_plan_graph(self):
        # Perform test rollouts to determine edge weights in the plan graph
        self.perform_test_rollouts(num_rollouts=100)

        # Construct the plan graph
        plan_graph = GraphSearch()
        plan_graph.construct_graph(self.chainer.chains)

        # Visualize the graph
        file_name = f"value_function_plots/{self.experiment_name}/plan_graph_seed_{self.seed}.png"
        plan_graph.visualize_plan_graph(file_name)

        return plan_graph

    def act(self, start_state, goal_state, target_option, inside_graph):
        """
        Given a start state `s` and a goal state `sg` pick an option to execute.
        Use planning if possible, else revert to the trained policy over options.
        Args:
            start_state (State)
            goal_state (State or np.ndarray)
            target_option (Option)
            inside_graph (bool)

        Returns:
            option (Option): Option to execute in the ground MDP
        """
        def _pick_option_to_execute_from_plan(selected_plan):
            admissible_options = [o for o in selected_plan if o.is_init_true(start_state) and not o.is_term_true(start_state)]
            if len(admissible_options) > 0:
                print(f"Got plan {selected_plan} and chose to execute {admissible_options[0]}")
                return admissible_options[0]  # type: Option
            return None

        def _get_revised_goal_state(inside_target):
            if isinstance(inside_target, Option):
                return inside_target.sample_state()
            if isinstance(inside_target, SalientEvent):
                return inside_target.target_state
            return None

        if self.plan_graph.does_path_exist(start_state, goal_state):
            plan = self.plan_graph.get_path_to_execute(start_state, goal_state)
            if len(plan) > 0:
                selected_option = _pick_option_to_execute_from_plan(plan)  # type: Option
                if selected_option is not None:
                    return selected_option

        # If the goal state is outside the graph, we want to use the planner to get to the `target_option`
        revised_goal_state = _get_revised_goal_state(target_option)
        if revised_goal_state is not None and not inside_graph:
            if self.plan_graph.does_path_exist(start_state, revised_goal_state):
                plan = self.plan_graph.get_path_to_execute(start_state, revised_goal_state)
                if len(plan) > 0:
                    selected_option = _pick_option_to_execute_from_plan(plan)  # type: Option
                    if selected_option is not None:
                        return selected_option

        print(f"Reverting to the DSC policy over options")
        return self.chainer.act(start_state, train_mode=True)

    def chain_for_target_event(self, goal_state):
        for chain in self.chainer.chains:  # type: SkillChain
            if chain.target_salient_event(goal_state):
                return chain
        return None

    def does_exist_chain_targeting_state(self, goal_state):
        return self.chain_for_target_event(goal_state) is not None

    def get_salient_event_for_state(self, state):
        """
        If `state` is inside a known salient event, return that salient event. Else, return None.
        Args:
            state (State)

        Returns:
            salient_event (SalientEvent or None)
        """

        all_salient_events = self.plan_graph.plan_graph.nodes
        all_salient_events = [event for event in all_salient_events if isinstance(event, SalientEvent)]
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

        def _create_new_salient_event():
            all_salient_events = self.mdp.get_all_target_events_ever()
            event_indices = [event.event_idx for event in all_salient_events]
            new_salient_event = SalientEvent(goal_state, event_idx=max(event_indices) + 1)
            self.mdp.add_new_target_event(new_salient_event) # TODO: Shouldn't you add the salient event to the graph as well??
            return new_salient_event

        target_salient_event = self.get_salient_event_for_state(goal_state)
        if target_salient_event is None:
            target_salient_event = _create_new_salient_event()
        return target_salient_event

    def should_planning_run_loop_terminate(self, state, goal_state, goal_salient_event, inside_target, salient_event_inside_graph):
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
            inside_target (Option or SalientEvent): Target event inside the graph (can be option or salient-event)
            salient_event_inside_graph (bool): Whether the salient event is inside the known part of the graph

        Returns:
            should_terminate (bool)
        """

        # When the target event is inside the graph, we want to terminate in 2 situations
        # 1) If we haven't yet constructed an option targeting the specific goal-state, then
        #    we terminate when we reach the target-option
        # 2) If we have already added the test-time-option to the graph, then we terminate
        #    when we reach the goal-state.
        if salient_event_inside_graph:
            if self.is_state_in_known_salient_event(goal_state):
                return goal_salient_event(state)
            return inside_target.is_init_true(state)

        # -- When the target event is outside the graph, we want to terminate when we reach the
        # -- option which we have chosen to fall off the graph from.
        # -- The small caveat is that random actions could have taken us to the test-goal, in
        # -- which case we should also terminate.
        else:
            reached_jumping_off_point = inside_target.is_init_true(state)
            magically_reached_goal = goal_salient_event(state)
            return reached_jumping_off_point or magically_reached_goal

    def create_goal_option(self, goal_salient_event, target_option, goal_event_inside_graph, episode):
        """

        Args:
            goal_salient_event (SalientEvent): The salient event induced by the goal_state
            target_option (Option): The option to get to in the known part of the graph
            target_option (Option): The option to get to in the known part of the graph
            goal_event_inside_graph (bool): Whether the `goal_salient_event` is inside the graph
            episode (int): The current episode number

        Returns:
            new_option (Option): Option created to take us to the `goal_salient_event`. When the
                                 `goal_state` is inside the graph, this option takes us from the
                                  `target_option`'s initiation set to the `goal_salient_event`.
                                  When it is outside the graph, the DSC agent can execute this
                                  option from anywhere.
        """
        if goal_event_inside_graph:
            if not isinstance(target_option, Option):
                ipdb.set_trace()
            new_option = self.create_goal_option_inside_graph(target_option, goal_salient_event)
        else:
            new_option = self.create_goal_option_outside_graph(goal_salient_event)

        # Visualize the new graph with the new option node
        file_name = f"value_function_plots/{self.experiment_name}/plan_graph_seed_{self.seed}_episode_{episode}.png"
        self.plan_graph.visualize_plan_graph(file_name)

        return new_option

    def execute_actions_to_reach_goal(self, episode, num_steps, inside_graph,
                                      new_option=None, goal_salient_event=None, eval_mode=False):
        """

        Args:
            episode (int)
            num_steps (int)
            inside_graph (bool)
            new_option (Option): If `inside_graph`, execute this option to get to the goal
            goal_salient_event (SalientEvent)
            eval_mode (bool)

        Returns:
            next_state (State)
            step_number (int)
        """

        # If the DSC agent has completed creating a forward chain targeting a salient event, but
        # is still in the process of creating the backward chain, `inside_graph` will be true,
        # but no `new_option` would have been created
        if inside_graph and new_option is not None:
            step_number = self.perform_option_rollout(option=new_option,
                                                      episode=episode,
                                                      step=num_steps,
                                                      goal_salient_event=goal_salient_event,
                                                      eval_mode=eval_mode)

        else:
            dsc_run_loop_budget = self.chainer.max_steps - num_steps
            pre_rollout_state = deepcopy(self.mdp.cur_state)

            print(f"About to rollout DSC for {dsc_run_loop_budget} steps with interrupt handle {goal_salient_event}")

            # Deliberately did not add `eval_mode` to the DSC rollout b/c we usually do this when we
            # want to fall off the graph, and it is always good to do some exploration when outside the graph
            score, step_number, newly_created_options = self.chainer.dsc_rollout(episode_number=episode,
                                                                                 step_number=num_steps,
                                                                                 interrupt_handle=goal_salient_event)

            print(f"Returned from DSC runloop having learned {newly_created_options}")

            self.manage_options_learned_during_rollout(newly_created_options, pre_rollout_state,
                                                       goal_salient_event, jumped_off_the_graph=True)

        return deepcopy(self.mdp.cur_state), step_number

    def perform_option_rollout(self, option, episode, step, eval_mode, goal_salient_event):

        state_before_rollout = deepcopy(self.mdp.cur_state)

        option_transitions, option_reward = option.execute_option_in_mdp(self.mdp,
                                                                         episode=episode,
                                                                         step_number=step,
                                                                         eval_mode=eval_mode)

        new_options = self.chainer.manage_skill_chain_after_option_rollout(state_before_rollout=state_before_rollout,
                                                                           executed_option=option,
                                                                           created_options=[],
                                                                           episode_number=episode,
                                                                           option_transitions=option_transitions)

        # With off-policy triggers, you can trigger the termination condition of an option w/o rolling it out
        self.manage_options_learned_during_rollout(newly_created_options=new_options,
                                                   pre_rollout_state=state_before_rollout,
                                                   goal_salient_event=goal_salient_event,
                                                   jumped_off_the_graph=False)

        # Modify the edge weight associated with the executed option
        self.modify_executed_option_edge_weight(option)

        return step + len(option_transitions)

    def manage_options_learned_during_rollout(self, newly_created_options, pre_rollout_state,
                                              goal_salient_event, jumped_off_the_graph):

        # TODO: Adding new options is not enough - we also have to create their children
        for newly_created_option in newly_created_options:  # type: Option
            self.add_newly_created_option_to_plan_graph(newly_created_option)

        if self.use_ucb:
            self.manage_bandits_after_dsc_rollout(pre_rollout_state, newly_created_options,
                                                  goal_salient_event, jumped_off_the_graph=jumped_off_the_graph)

    def manage_bandits_after_dsc_rollout(self, pre_rollout_state, newly_created_options,
                                         goal_salient_event, jumped_off_the_graph):
        """
        If new options were created during the DSC rollout, we have to inform all the bandits about that.
        Furthermore, we need to update the value of the option from which we ended up exiting the graph.

        Args:
            pre_rollout_state (State)
            newly_created_options (list)
            goal_salient_event (SalientEvent)
            jumped_off_the_graph (bool)

        Returns:

        """
        assert self.use_ucb

        for newly_created_option in newly_created_options:  # type: Option

            # If the newly created option is part of a completed chain, then we should tell all
            # the bandit agents about this option so that they can compute the value of falling off the graph from it
            self.update_bandits_with_new_option(newly_created_option)

        if goal_salient_event in self.bandit_agents.keys() and jumped_off_the_graph:
            used_bandit = self.bandit_agents[goal_salient_event]  # type: UCBActionSelectionAgent

            jumping_off_options = [option for option in self.get_options_in_known_part_of_the_graph()
                                    if option.is_init_true(pre_rollout_state)]

            for option in jumping_off_options:  # type: Option
                if option in used_bandit.options:
                    used_bandit.update(option, success=goal_salient_event(self.mdp.cur_state))

    def update_bandits_with_new_option(self, new_option):
        """
        When the DSC agent learns a new option, not only do we want to add it to the skill-graph,
        but we also want to add this option to the candidate set of options available to the
        UCB option selection module.

        Args:
            new_option (Option)

        """
        assert new_option.get_training_phase() == "initiation_done"
        assert self.use_ucb
        for salient_event in self.bandit_agents:  # type: SalientEvent
            if not new_option.backward_option:  # TODO: Adding condition for implementation ease for now
                bandit = self.bandit_agents[salient_event]
                bandit.add_candidate_option(new_option)

    def add_newly_created_option_to_plan_graph(self, newly_created_option):

        assert newly_created_option.get_training_phase() == "initiation_done",\
               f"{newly_created_option} in {newly_created_option.get_training_phase()}"

        # If the new option isn't already in the graph, add it
        self.plan_graph.add_node(newly_created_option)

        # Skill Chain corresponding to the newly created option
        chain = self.chainer.chains[newly_created_option.chain_id - 1]  # type: SkillChain
        is_leaf_node = newly_created_option in chain.get_leaf_nodes_from_skill_chain()
        init_salient_event = chain.init_salient_event
        target_salient_event = chain.target_salient_event

        # If the target_salient_event is not in the plan-graph, add it
        if target_salient_event not in self.plan_graph.plan_graph:
            self.plan_graph.add_node(target_salient_event)

        # If the init salient event is not in the plan-graph, add it
        if is_leaf_node and (init_salient_event not in self.plan_graph.plan_graph):
            self.plan_graph.add_node(init_salient_event)

        # For adding edges, there are 3 possible cases:
        # 1. Nominal case - you are neither a root nor a leaf - just connect to your parent option
        # 2. Leaf Option - if you are a leaf option, you should add a 0 weight edge from the init
        #                  salient event to the leaf option
        # 3. Root option - if you are a root option, you should add an optimistic node from the
        #                  root option to the target salient event

        # Case 3: Add edge from the first salient event to the first backward option
        if newly_created_option.parent is None:
            print(f"Case 3: Adding edge from {newly_created_option} to {target_salient_event}")
            self.plan_graph.add_edge(newly_created_option, target_salient_event, edge_weight=1.)

        # Case 2: Leaf option # TODO: Need to check intersection with the init_salient_event as well
        if chain.is_chain_completed(self.chainer.chains) \
            and is_leaf_node \
                and chain.detect_intersection_between_option_and_event(newly_created_option, init_salient_event):

            print(f"Case 2: Adding edge from {init_salient_event} to {newly_created_option}")
            self.plan_graph.add_edge(init_salient_event, newly_created_option, edge_weight=0.)

        # Case 1: Nominal case
        if newly_created_option.parent is not None:
            print(f"Case 1: Adding edge from {newly_created_option} to {newly_created_option.parent}")
            self.plan_graph.add_edge(newly_created_option, newly_created_option.parent, edge_weight=1.)

        # Case 4: I did not consider this case before. But, you could also intersect with any other
        # event in the MDP -- you might not rewire your chain because that event is not "completed/chained-to",
        # but that still means that there should be an edge from it to the newly learned option
        if self.chainer.event_intersection_salience or self.chainer.option_intersection_salience:
            # 1. Get intersecting events
            # 2. Add an edge from each intersecting event to the newly_created_option
            events = [event for event in self.mdp.get_all_target_events_ever() if event != chain.target_salient_event]
            intersecting_events = [event for event in events
                                   if chain.detect_intersection_between_option_and_event(newly_created_option, event)]
            for event in intersecting_events:  # type: SalientEvent
                assert isinstance(event, SalientEvent)
                print(f"Adding edge from {event} to {newly_created_option}")
                self.plan_graph.add_edge(event, newly_created_option, edge_weight=0.)

        elif self.chainer.option_intersection_salience:
            # 1. Get intersecting options
            # 2. Add an each from each intersecting option to the newly_created_option
            raise NotImplementedError("Option intersections")

        if chain.is_chain_completed(self.chainer.chains):
            visualize_graph(self.chainer.chains, self.chainer.experiment_name, True)

    def planner_rollout(self, *, state, goal_state, target_option, inside_graph,
                                 goal_salient_event, episode_number, step_number, eval_mode):

        should_terminate_run_loop = False

        while not should_terminate_run_loop and step_number < self.chainer.max_steps:

            selected_option = self.act(state, goal_state, target_option, inside_graph)  # type: Option

            step_number = self.perform_option_rollout(option=selected_option,
                                                      episode=episode_number,
                                                      step=step_number,
                                                      eval_mode=eval_mode,
                                                      goal_salient_event=goal_salient_event)

            state = deepcopy(self.mdp.cur_state)

            should_terminate_run_loop = self.should_planning_run_loop_terminate(state,
                                                                                goal_state,
                                                                                goal_salient_event,
                                                                                target_option,
                                                                                inside_graph)

            if should_terminate_run_loop and not goal_salient_event(state):  # Case II or Case III

                new_option = None
                if not self.does_exist_chain_targeting_state(goal_state):
                    assert target_option is not None
                    new_option = self.create_goal_option(goal_salient_event, target_option, inside_graph, episode_number)  # type: Option

                print("[SkillGraphPlanning] Handing control to DSC inside the planning run loop")
                state, step_number = self.execute_actions_to_reach_goal(episode_number,
                                                                        step_number,
                                                                        inside_graph,
                                                                        new_option,
                                                                        goal_salient_event,
                                                                        eval_mode=eval_mode)

        return step_number, goal_salient_event(state)


    def planning_run_loop(self, *, start_episode, goal_state, to_reset,
                          step_number=0, start_state=None, goal_salient_event=None, eval_mode=False):
        """
        This is a test-time run loop that uses planning to select options when it can.
        Args:
            start_episode (int)
            goal_state (State)
            to_reset (bool)
            step_number (int)
            start_state (State)
            goal_salient_event (SalientEvent)
            eval_mode (bool)

        Returns:
            executed_options (list)
            reached_goal (bool)
        """

        if to_reset:
            self.reset(start_state)

        state = deepcopy(self.mdp.cur_state)

        episode = start_episode

        goal_salient_event = self.get_goal_salient_event(goal_state) if goal_salient_event is None else goal_salient_event

        # When going in the forward direction, we will target this option if the goal is not in a known salient event
        target_option, inside_graph = self.get_target_option(goal_state, goal_salient_event, backward_direction=False)

        # If your goal is outside the graph and you are already at the target option, run skill chaining.
        already_at_target = self.is_at_target_inside_graph(target_option, state) and not inside_graph
        no_planning_possible = self.is_plan_graph_empty() or already_at_target

        if no_planning_possible:
            print(f"Handing control to DSC - already_at_target_option {already_at_target} \t empty_graph: {self.is_plan_graph_empty()}")
            state, step_number = self.execute_actions_to_reach_goal(episode, step_number,
                                                                    inside_graph, new_option=None,
                                                                    goal_salient_event=goal_salient_event,
                                                                    eval_mode=eval_mode)

            return step_number, goal_salient_event(state)

        step_number, success = self.planner_rollout(state=state, goal_state=goal_state, target_option=target_option,
                                                    inside_graph=inside_graph, goal_salient_event=goal_salient_event,
                                                    episode_number=episode, step_number=step_number, eval_mode=eval_mode)

        return step_number, success

    def get_target_option_for_event_inside_graph(self, goal_state, backward_direction):
        """
        What does it mean for a state to be "inside the graph"? It means that the state is
        inside a skill-chain and that that skill chain is "completed". To be a "complete" chain
        means that you are no longer adding skills to your chain AND that all your skills are
        already part of the plan-graph.

        Args:
            goal_state (State)
            backward_direction (bool)

        Returns:
            target_option: Option in the graph if it exists, None otherwise
        """

        def _chain_is_completed(c):
            options_in_graph = all([o in self.plan_graph.option_nodes for o in c.options])
            return options_in_graph and c.is_chain_completed(self.chainer.chains)

        # Grab all the completed skill chains that contain the goal_state
        chains_with_goal_state = [chain for chain in self.chainer.chains if
                                  chain.state_in_chain(goal_state) and
                                  _chain_is_completed(chain) and
                                  chain.is_backward_chain == backward_direction]

        # Select the option (options created earlier have a higher priority)
        options_with_goal_state = [chain.get_option_for_state(goal_state) for chain in chains_with_goal_state]
        options_with_goal_state = [option for option in options_with_goal_state if option is not None]

        if len(options_with_goal_state) > 0:  # TODO: How to pick an option from this list of feasible choices?
            target_option = options_with_goal_state[0]
            return target_option

        return None

    def get_target_option(self, goal_state, goal_salient_event, backward_direction):
        """
        If the `goal_salient_event` is inside the known part of the graph, then we want the
        planning run-loop to get us to the option whose initiation set contains that event.
        If the `goal_salient_event` is outside the graph, then we use the planning run-loop
        to get to a target-option selected by our bandit algorithm.
        Args:
            goal_state (State)
            goal_salient_event (SalientEvent)
            backward_direction (bool)

        Returns:
            target_option (Option): The option inside the graph that we want to get to
            inside_graph (bool): Whether the `goal_salient_event` is inside the graph or not
        """
        inside_target_option = self.get_target_option_for_event_inside_graph(goal_state, backward_direction)

        if inside_target_option is not None:
            return inside_target_option, True

        if self.use_ucb and goal_salient_event not in self.bandit_agents:
            candidate_options = self.get_options_in_known_part_of_the_graph()
            candidate_options = [option for option in candidate_options if not option.backward_option]  # TODO: This only makes sense if you start at s0
            self.bandit_agents[goal_salient_event] = UCBActionSelectionAgent(goal_state=goal_state,
                                                                             options=candidate_options,
                                                                             chains=self.chainer.chains,
                                                                             use_option_vf=False)  # TODO: Set to true in the future

        outside_target_option = self.choose_option_to_fall_off_graph_from(goal_salient_event)
        return outside_target_option, False

    def get_options_in_known_part_of_the_graph(self):
        completed_chains = [chain for chain in self.chainer.chains if chain.is_chain_completed(self.chainer.chains)]

        # Grab the learned options from the completed chains
        known_options = []
        for chain in completed_chains:  # type: SkillChain
            chain_options = [option for option in chain.options if option.get_training_phase() == "initiation_done"
                                and option in self.plan_graph.option_nodes]
            known_options.append(chain_options)
        known_options = list(itertools.chain.from_iterable(known_options))
        return known_options

    def get_events_in_known_part_of_the_graph(self):
        def _chain_contains_event(c, e):
            return c.target_salient_event == e or c.init_salient_event == e

        completed_chains = [chain for chain in self.chainer.chains if chain.is_chain_completed(self.chainer.chains)]
        known_events = self.plan_graph.salient_nodes
        known_events_inside_graph = []
        for event in known_events:  # type: SalientEvent
            if any([_chain_contains_event(chain, event) for chain in completed_chains]):
                known_events_inside_graph.append(event)
        return list(set(known_events_inside_graph))

    def is_plan_graph_empty(self):
        return len(self.get_options_in_known_part_of_the_graph()) == 0

    def is_at_target_inside_graph(self, target, state):
        """

        Args:
            target (SalientEvent or Option)
            state (State)

        Returns:
            at_target (bool)
        """
        if target is not None:
            assert isinstance(target, (Option, SalientEvent))
            return target.is_init_true(state)
        return False

    def create_goal_option_inside_graph(self, train_goal_option, target_salient_event):
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

        ipdb.set_trace()

        def _get_start_state_for_new_chain(in_graph_option):
            for s, a, r, sp, done in in_graph_option.solver.replay_buffer:
                if done:
                    return sp
            return np.array((0, 0))

        # Create a new salient event out of
        if len(train_goal_option.children) > 0:
            init_salient_option = train_goal_option.children[0]  # type: Option
            event_idx = len(self.mdp.get_all_target_events_ever())
            init_salient_event = DSCOptionSalientEvent(option=init_salient_option,
                                                       event_idx=event_idx)
            start_states = init_salient_option.effect_set
        else:  # Leaf node
            init_salient_event = self.chainer.chains[train_goal_option.chain_id-1].init_salient_event
            start_states = [_get_start_state_for_new_chain(train_goal_option)]  # Why not use the effect set of the option?

        chain_id = max([chain.chain_id for chain in self.chainer.chains]) + 1
        new_forward_chain = SkillChain(start_states=start_states, mdp_start_states=[self.mdp.init_state],
                                       init_salient_event=init_salient_event,
                                       target_salient_event=target_salient_event, is_backward_chain=False,
                                       chain_id=chain_id, options=[], option_intersection_salience=False,
                                       event_intersection_salience=False)

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
                                      use_warmup_phase=self.chainer.use_warmup_phase,
                                      update_global_solver=self.chainer.update_global_solver,
                                      is_backward_option=False,
                                      init_salient_event=init_salient_event,
                                      target_salient_event=target_salient_event)

        print(f"Created {new_untrained_option} targeting {target_salient_event}")

        # Set the initiation classifier for this new option
        new_untrained_option.num_goal_hits = train_goal_option.num_goal_hits
        new_untrained_option.positive_examples = train_goal_option.positive_examples
        new_untrained_option.negative_examples = train_goal_option.negative_examples
        new_untrained_option.initiation_classifier = train_goal_option.initiation_classifier

        print(f"Set {new_untrained_option}'s training phase to {new_untrained_option.get_training_phase()}")

        # Augment the DSC agent with the new option and pre-train it's policy
        if self.pretrain_option_policies:
            new_untrained_option.initialize_with_global_solver()
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

        return new_untrained_option

    def create_goal_option_outside_graph(self, target_salient_event):
        chain_id = max([chain.chain_id for chain in self.chainer.chains]) + 1

        # The init salient event of this new chain is initialized to be the start state of the MDP
        # However, as DSC adds skills to this chain, it will eventually stop when it finds a chain intersection
        # At that point it will rewire the chain to set its init_salient_event to be the region of intersection
        init_salient_event = self.mdp.get_start_state_salient_event()
        new_skill_chain = SkillChain(start_states=[self.mdp.init_state],
                                     mdp_start_states=[self.mdp.init_state],
                                     init_salient_event=init_salient_event,
                                     target_salient_event=target_salient_event,
                                     is_backward_chain=False,
                                     option_intersection_salience=self.chainer.option_intersection_salience,
                                     event_intersection_salience=self.chainer.event_intersection_salience,
                                     chain_id=chain_id,
                                     options=[])

        if new_skill_chain not in self.chainer.chains:
            self.chainer.add_skill_chain(new_skill_chain)

        # Create the root option for this new skill-chain
        name = f"goal_option_{target_salient_event.event_idx}"
        new_untrained_option = Option(self.mdp, name=name, global_solver=self.chainer.global_option.solver,
                                      lr_actor=self.chainer.global_option.solver.actor_learning_rate,
                                      lr_critic=self.chainer.global_option.solver.critic_learning_rate,
                                      ddpg_batch_size=self.chainer.global_option.solver.batch_size,
                                      subgoal_reward=self.chainer.subgoal_reward,
                                      buffer_length=self.chainer.buffer_length,
                                      classifier_type=self.chainer.classifier_type,
                                      num_subgoal_hits_required=self.chainer.num_subgoal_hits_required,
                                      seed=self.seed, parent=None, max_steps=self.chainer.max_steps,
                                      enable_timeout=self.chainer.enable_option_timeout,
                                      chain_id=chain_id,
                                      writer=self.chainer.writer, device=self.chainer.device,
                                      dense_reward=self.chainer.dense_reward,
                                      initialize_everywhere=self.chainer.trained_options[1].initialize_everywhere,
                                      max_num_children=self.chainer.trained_options[1].max_num_children,
                                      use_warmup_phase=self.chainer.use_warmup_phase,
                                      update_global_solver=self.chainer.update_global_solver,
                                      is_backward_option=False,
                                      init_salient_event=init_salient_event,
                                      target_salient_event=target_salient_event,
                                      initiation_period=0)  # TODO: This might not be wise in Ant

        print(f"Created {new_untrained_option} targeting {target_salient_event}")

        # Augment the DSC agent with the new option and pre-train it's policy
        if self.pretrain_option_policies:
            new_untrained_option.initialize_with_global_solver()

            # Check if the value function diverges - if it does, then retry. If it still does,
            # then initialize to random weights and use that
            max_q_value = make_chunked_value_function_plot(new_untrained_option.solver, -1, self.seed,
                                                           self.experiment_name,
                                                           replay_buffer=self.chainer.global_option.solver.replay_buffer)
            if max_q_value > 500:
                print("=" * 80)
                print(f"{new_untrained_option} VF diverged")
                print("=" * 80)
                new_untrained_option.reset_option_solver()
                new_untrained_option.initialize_with_global_solver()
                max_q_value = make_chunked_value_function_plot(new_untrained_option.solver, -1, self.seed,
                                                               self.experiment_name,
                                                               replay_buffer=self.chainer.global_option.solver.replay_buffer)
                if max_q_value > 500:
                    print("=" * 80)
                    print(f"{new_untrained_option} VF diverged AGAIN")
                    print("Just initializing to random weights")
                    print("=" * 80)
                    new_untrained_option.reset_option_solver()

        self.chainer.untrained_options.append(new_untrained_option)
        self.chainer.augment_agent_with_new_option(new_untrained_option, 0.)

        # Note: we are not adding the new option to the plan-graph

        return new_untrained_option

    def modify_executed_option_edge_weight(self, selected_option):

        def _modify_edge_weight_of_root_option(root_option):
            assert root_option.parent is None
            target_event = self.chainer.chains[root_option.chain_id - 1].target_salient_event
            success_rate = root_option.get_option_success_rate()
            weight = 1. / success_rate if success_rate > 0 else 0.
            self.plan_graph.set_edge_weight(root_option, target_event, weight=weight)

        def _modify_edge_weight_of_child_option(child_option):
            assert child_option is not None
            if isinstance(child_option, Option):
                child_option_success_rate = child_option.get_option_success_rate()
                weight = 1. / child_option_success_rate if child_option_success_rate > 0 else 0.
                self.plan_graph.set_edge_weight(child_option, selected_option,
                                                weight=1. / weight)
            elif isinstance(child_option, SalientEvent):
                self.plan_graph.set_edge_weight(child_option, selected_option, weight=0.)
            else:
                raise ValueError(child_option, type(child_option))

        if selected_option in self.plan_graph.option_nodes:  # type: Option
            for child in selected_option.children:  # type: Option or SalientEvent
                if child is not None:
                    # TODO: We want to do a moving average filter so as to not wipe off the optimism
                    _modify_edge_weight_of_child_option(child)

            if selected_option.parent is None:
                _modify_edge_weight_of_root_option(selected_option)

    def choose_option_to_fall_off_graph_from(self, goal_salient_event):
        def _ucb_choose_option_to_fall_off_graph_from():
            bandit_agent = self.bandit_agents[goal_salient_event]  # type: UCBActionSelectionAgent
            selected_option = bandit_agent.act()  # type: Option
            print(f"[UCB-Node-Selection] Chose {selected_option} to fall off the graph from")
            return selected_option

        def _distance_choose_event_too_fall_off_graph_from():
            candidate_events = self.get_events_in_known_part_of_the_graph()
            candidate_events = [event for event in candidate_events if event != goal_salient_event]
            distances = [np.linalg.norm(event.target_state - goal_salient_event.target_state) for event in candidate_events]
            selected_idx = np.argmin(distances)
            selected_idx = distances[0] if isinstance(distances, np.ndarray) else selected_idx
            selected_event = candidate_events[selected_idx]
            print(f"[Distance-Node-Selection] Chose {selected_event} to fall off the graph from")
            return selected_event

        if not self.is_plan_graph_empty():
            if self.use_ucb:
                return _ucb_choose_option_to_fall_off_graph_from()
            return _distance_choose_event_too_fall_off_graph_from()

        return None

    def is_state_inside_graph(self, state):
        completed_chains = [chain for chain in self.chainer.chains if chain.is_chain_completed(self.chainer.chains)]
        for chain in completed_chains:  # type: SkillChain
            if chain.state_in_chain(state):
                return True
        return False

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

    def measure_success(self, goal_state, start_state=None, starting_episode=0, num_episodes=100):
        successes = 0
        self.chainer.create_backward_options = False
        self.chainer.learn_backward_options_offline = False  # TODO: Need a better way to say that we should not create back options at test time
        for episode in range(num_episodes):
            step_number, reached_goal = self.planning_run_loop(start_episode=starting_episode+episode,
                                                               goal_state=goal_state,
                                                               start_state=start_state,
                                                               to_reset=True,
                                                               eval_mode=True)
            if reached_goal:
                successes += 1
        return successes
