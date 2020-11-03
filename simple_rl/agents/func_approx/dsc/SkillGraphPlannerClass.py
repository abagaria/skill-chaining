import ipdb
import random
import itertools
import numpy as np
from copy import deepcopy
from simple_rl.mdp.StateClass import State
from simple_rl.agents.func_approx.dsc.SkillChainingAgentClass import SkillChaining
from simple_rl.agents.func_approx.dsc.ChainClass import SkillChain
from simple_rl.agents.func_approx.dsc.OptionClass import Option
from simple_rl.agents.func_approx.dsc.SalientEventClass import SalientEvent, DSCOptionSalientEvent
from simple_rl.agents.func_approx.dsc.utils import make_chunked_value_function_plot, visualize_graph
from simple_rl.agents.func_approx.dsc.PlanGraphClass import PlanGraph


class SkillGraphPlanner(object):
    def __init__(self, *, mdp, chainer, use_her, pretrain_option_policies, experiment_name, seed):
        """
        The Skill Graph Planning Agent uses graph search to get to target states given
        during test time. If the goal state is in a salient event already known to the agent,
        it can get there without any learning involved. If it is not a known salient event,
        it will create new skill chains to reliably target it.
        Args:
            mdp (PointReacherMDP)
            chainer (SkillChaining)
            use_her (bool)
            pretrain_option_policies (bool)
            experiment_name (str)
            seed (int)
        """
        self.mdp = mdp
        self.chainer = chainer
        self.use_her = use_her
        self.pretrain_option_policies = pretrain_option_policies
        self.experiment_name = experiment_name
        self.seed = seed

        self.plan_graph = PlanGraph()

    # -----------------------------–––––––--------------
    # Control loop methods
    # -----------------------------–––––––--------------

    def choose_closest_source_target_vertex_pair(self, state, goal_salient_event, choose_among_events):
        """
        Given a salient event outside the graph, find its nearest neighbor in the graph. We will
        use our planning based control loop to reach that nearest neighbor and then use deep
        skill chaining the rest of the way.

        Args:
            state (State)
            goal_salient_event (SalientEvent)
            choose_among_events (bool)

        Returns:
            closest_source (SalientEvent or Option)
            closest_target (SalientEvent or Option)
        """
        candidate_vertices_to_fall_from = self.plan_graph.get_reachable_nodes_from_source_state(state)
        candidate_vertices_to_jump_to = self.plan_graph.get_nodes_that_reach_target_node(goal_salient_event)
        candidate_vertices_to_jump_to = list(candidate_vertices_to_jump_to) + [goal_salient_event]

        if choose_among_events:
            candidate_vertices_to_fall_from = [v for v in candidate_vertices_to_fall_from if isinstance(v, SalientEvent)]
            candidate_vertices_to_jump_to = [v for v in candidate_vertices_to_jump_to if isinstance(v, SalientEvent)]

        return self.get_closest_pair_of_vertices(candidate_vertices_to_fall_from, candidate_vertices_to_jump_to)

    def act(self, state, goal_vertex, sampled_goal):
        assert isinstance(state, (State, np.ndarray)), f"{type(state)}"
        assert isinstance(goal_vertex, (SalientEvent, Option)), f"{type(goal_vertex)}"

        def _pick_option_to_execute_from_plan(selected_plan):
            admissible_options = [o for o in selected_plan if o.is_init_true(state) and not o.is_term_true(state)]
            if len(admissible_options) > 0:
                print(f"Got plan {selected_plan} and chose to execute {admissible_options[0]}")
                return admissible_options[0]  # type: Option
            return None

        if self.plan_graph.does_path_exist(state, goal_vertex):
            plan = self.plan_graph.get_path_to_execute(state, goal_vertex)
            if len(plan) > 0:
                selected_option = _pick_option_to_execute_from_plan(plan)  # type: Option
                if selected_option is not None:
                    return selected_option

        print(f"Reverting to the DSC policy over options targeting {sampled_goal}")
        return self.chainer.act(state, goal=sampled_goal, train_mode=True, goal_vertex=goal_vertex)

    def planner_rollout_inside_graph(self, *, state, goal_vertex, goal_salient_event,
                                     episode_number, step_number, eval_mode):
        """ Control loop that drives the agent from `state` to `goal_vertex` or times out. """
        assert isinstance(state, State)
        assert isinstance(goal_vertex, (SalientEvent, Option))
        assert isinstance(goal_salient_event, SalientEvent)
        assert isinstance(episode_number, int)
        assert isinstance(step_number, int)
        assert isinstance(eval_mode, bool)

        episodic_trajectory = []
        option_trajectory = []

        poo_goal = self.sample_from_vertex(goal_vertex) if self.use_her else None
        should_terminate = lambda s, o, inside, outside, step: inside(s, o) or outside(s, o) or step >= self.chainer.max_steps
        inside_predicate = lambda s, o: goal_vertex(s) if isinstance(goal_vertex, SalientEvent) else goal_vertex.is_term_true(s)
        outside_predicate = lambda s, o: goal_salient_event(s)

        # We target DSCOptionSalientEvents because we want to improve specific options.
        # So we don't terminate unless it was executing that option that took us to its effect set.
        if isinstance(goal_vertex, DSCOptionSalientEvent):
            inside_predicate = lambda s, o: goal_vertex(s) and o == goal_vertex.option
        if isinstance(goal_salient_event, DSCOptionSalientEvent):
            outside_predicate = lambda s, o: goal_salient_event(s) and o == goal_salient_event.option

        option = None

        while not should_terminate(state, option, inside_predicate, outside_predicate, step_number):
            option = self.act(state, goal_vertex, sampled_goal=poo_goal)
            step_number, option_transitions = self.perform_option_rollout(option=option,
                                                                          episode=episode_number,
                                                                          step=step_number,
                                                                          eval_mode=eval_mode,
                                                                          policy_over_options_goal=poo_goal,
                                                                          goal_salient_event=goal_salient_event)

            # Book keeping
            state = deepcopy(self.mdp.cur_state)
            episodic_trajectory.append(option_transitions)
            option_trajectory.append((option, option_transitions))

            # TODO: Hack - Breaking is not enough -- you have to sample a new goal too
            if option.name == "global_option" and self.mdp.dense_gc_reward_function(state, poo_goal, {})[1]:
                break

        if self.use_her:
            self.chainer.perform_experience_replay(state, episodic_trajectory, option_trajectory, overall_goal=poo_goal)

        return step_number

    def run_sub_loop(self, state, goal_vertex, step, goal_salient_event, episode, eval_mode):
        should_terminate = lambda s, step_number: goal_salient_event(s) or step_number >= self.chainer.max_steps
        planner_condition = lambda s, g: self.plan_graph.does_path_exist(s, g) and not self.is_state_inside_vertex(s, g)

        if planner_condition(state, goal_vertex) and not should_terminate(state, step):
            print(f"[Planner] Rolling out from {state.position} targeting {goal_vertex}")
            step = self.planner_rollout_inside_graph(state=state,
                                                     goal_vertex=goal_vertex,
                                                     goal_salient_event=goal_salient_event,
                                                     episode_number=episode,
                                                     step_number=step,
                                                     eval_mode=eval_mode)
            state = deepcopy(self.mdp.cur_state)

        elif not should_terminate(state, step):
            state, step = self.dsc_outside_graph(episode=episode,
                                                 num_steps=step,
                                                 goal_salient_event=goal_salient_event,
                                                 dsc_goal_vertex=goal_vertex)

        return state, step

    def run_loop(self, *, state, goal_salient_event, episode, step, eval_mode):
        assert isinstance(state, State)
        assert isinstance(goal_salient_event, SalientEvent)
        assert isinstance(episode, int)
        assert isinstance(step, int)
        assert isinstance(eval_mode, bool)

        planner_goal_vertex, dsc_goal_vertex = self._get_goal_vertices_for_rollout(state, goal_salient_event)
        print(f"Planner goal: {planner_goal_vertex}, DSC goal: {dsc_goal_vertex} and Goal: {goal_salient_event}")

        if isinstance(planner_goal_vertex, DSCOptionSalientEvent) and planner_goal_vertex == dsc_goal_vertex == goal_salient_event:
            print(f"First targeting {planner_goal_vertex.option.init_salient_event} and then {planner_goal_vertex}")
            state, step = self.run_sub_loop(state, planner_goal_vertex.option.init_salient_event, step, goal_salient_event, episode, eval_mode)
            state, step = self.run_sub_loop(state, planner_goal_vertex, step, goal_salient_event, episode, eval_mode)
            return step, goal_salient_event(state)

        state, step = self.run_sub_loop(state, planner_goal_vertex, step, goal_salient_event, episode, eval_mode)
        state, step = self.run_sub_loop(state, dsc_goal_vertex, step, goal_salient_event, episode, eval_mode)
        state, step = self.run_sub_loop(state, goal_salient_event, step, goal_salient_event, episode, eval_mode)

        return step, goal_salient_event(state)

    def get_test_time_subgoals(self, start_state_event, goal_salient_event):
        """
        When the start event has no descendants, we can do better than relying on the policy over options
        targeting the goal-salient-event. More specifically, we first find the best vertex that will take
        us to the goal-salient-event, and *then* find the best vertex in the graph that we can achieve from
        the start-state-salient-event.

        Args:
            start_state_event (SalientEvent)
            goal_salient_event (SalientEvent)

        Returns:
            beta1 (SalientEvent)
            beta2 (Salient Event)
        """
        # Choose sub-goals from among salient events because Skill-Chains from arbitrary vertices not supported yet
        filtering_condition = lambda v: isinstance(v, SalientEvent) and not isinstance(v, DSCOptionSalientEvent)

        # Pick the target node that is closest to the goal-salient-event
        num_ancestors = len(self.plan_graph.get_nodes_that_reach_target_node(goal_salient_event))
        src_vertices = self.plan_graph.get_nodes_that_reach_target_node(goal_salient_event) if num_ancestors > 0 \
                         else self.plan_graph.plan_graph.nodes
        src_vertices = [v for v in src_vertices if filtering_condition(v)]
        pair2 = self.get_closest_pair_of_vertices(src_vertices, [goal_salient_event])
        beta2 = pair2[0] if pair2 is not None else goal_salient_event

        # Pick an ancestor of beta2 that is closest to the start-state-salient-event
        dest_vertices = self.plan_graph.get_nodes_that_reach_target_node(beta2)
        dest_vertices = [v for v in dest_vertices if filtering_condition(v) and v != start_state_event]
        pair1 = self.get_closest_pair_of_vertices([start_state_event], dest_vertices)
        beta1 = pair1[1] if pair1 is not None else start_state_event

        return beta1, beta2

    def test_loop(self, *, state, start_state_event, goal_salient_event, episode, step):
        num_descendants = len(self.plan_graph.get_reachable_nodes_from_source_state(state))

        if num_descendants == 0:
            beta1, beta2 = self.get_test_time_subgoals(start_state_event, goal_salient_event)
            state, step = self.run_sub_loop(state, beta1, step, goal_salient_event, episode, eval_mode=True)
            state, step = self.run_sub_loop(state, beta2, step, goal_salient_event, episode, eval_mode=True)
            state, step = self.run_sub_loop(state, goal_salient_event, step, goal_salient_event, episode, eval_mode=True)
            return step, state

        step, success = self.run_loop(state=state,
                                      goal_salient_event=goal_salient_event,
                                      episode=episode, step=step, eval_mode=True)

        return step, self.mdp.cur_state

    # -----------------------------–––––––--------------
    # Managing DSC Control Loops
    # -----------------------------–––––––--------------

    def dsc_outside_graph(self, episode, num_steps, goal_salient_event, dsc_goal_vertex):
        state = deepcopy(self.mdp.cur_state)
        dsc_goal_state = self.sample_from_vertex(dsc_goal_vertex)
        dsc_run_loop_budget = self.chainer.max_steps - num_steps
        dsc_interrupt_handle = lambda s: self.is_state_inside_vertex(s, dsc_goal_vertex) or goal_salient_event(s)

        if dsc_interrupt_handle(state):
            print(f"Not rolling out DSC because {state.position} triggered interrupt handle")
            return state, num_steps

        print(f"Rolling out DSC for {dsc_run_loop_budget} steps with goal vertex {dsc_goal_vertex} and goal state {dsc_goal_state}")

        # Deliberately did not add `eval_mode` to the DSC rollout b/c we usually do this when we
        # want to fall off the graph, and it is always good to do some exploration when outside the graph
        score, step_number, newly_created_options = self.chainer.dsc_rollout(episode_number=episode,
                                                                             step_number=num_steps,
                                                                             interrupt_handle=dsc_interrupt_handle,
                                                                             overall_goal=dsc_goal_state,
                                                                             goal_vertex=dsc_goal_vertex)

        print(f"Returned from DSC runloop having learned {newly_created_options}")

        self.manage_options_learned_during_rollout(newly_created_options, episode)

        return deepcopy(self.mdp.cur_state), step_number

    def perform_option_rollout(self, option, episode, step, eval_mode, policy_over_options_goal, goal_salient_event):

        state_before_rollout = deepcopy(self.mdp.cur_state)

        # Don't (off-policy) trigger the termination condition of options targeting the event we are currently leaving
        self.chainer.disable_triggering_options_targeting_init_event(state=state_before_rollout)

        # Execute the option: if the option is the global-option, it doesn't have a default
        # goal in the task-agnostic setting. As a result, we ask it to target the same goal
        # as the policy-over-options.
        option_goal = policy_over_options_goal if option.name == "global_option" else None
        option_transitions, option_reward = option.execute_option_in_mdp(self.mdp,
                                                                         episode=episode,
                                                                         step_number=step,
                                                                         eval_mode=eval_mode,
                                                                         poo_goal=option_goal,
                                                                         goal_salient_event=goal_salient_event)

        new_options = self.chainer.manage_skill_chain_after_option_rollout(state_before_rollout=state_before_rollout,
                                                                           executed_option=option,
                                                                           created_options=[],
                                                                           episode_number=episode,
                                                                           option_transitions=option_transitions)

        state_after_rollout = deepcopy(self.mdp.cur_state)

        # With off-policy triggers, you can trigger the termination condition of an option w/o rolling it out
        self.manage_options_learned_during_rollout(newly_created_options=new_options, episode=episode)

        # Modify the edge weight associated with the executed option
        self.modify_edge_weight(executed_option=option, final_state=state_after_rollout)

        return step + len(option_transitions), option_transitions

    def manage_options_learned_during_rollout(self, newly_created_options, episode):

        # TODO: Adding new options is not enough - we also have to create their children
        for newly_created_option in newly_created_options:  # type: Option
            self.add_newly_created_option_to_plan_graph(newly_created_option, episode)

    def _get_goal_vertices_for_rollout(self, state, goal_salient_event):
        # Revise the goal_salient_event if it cannot be reached from the current state
        if not self.plan_graph.does_path_exist(state, goal_salient_event):
            closest_vertex_pair = self.choose_closest_source_target_vertex_pair(state, goal_salient_event, False)
            if closest_vertex_pair is not None:
                planner_goal_vertex, dsc_goal_vertex = closest_vertex_pair
                print(f"Revised planner goal vertex to {planner_goal_vertex} and dsc goal vertex to {dsc_goal_vertex}")
                return planner_goal_vertex, dsc_goal_vertex
        return goal_salient_event, goal_salient_event

    # -----------------------------–––––––--------------
    # Maintaining the graph
    # -----------------------------–––––––--------------

    def modify_edge_weight(self, executed_option, final_state):
        """
        The edge weight in the graph represents the cost of executing an option policy.
        Remember that each option could be connected to a bunch of other options (if the effect
        set of the first option is inside the inside the initiation set of the other options).
        So, we must check if option execution successfully landed up in any of those initiation sets,
        and decrement the cost if that is the case. If we didn't reach the initiation set of some
        option that we are connected to, we must increment the edge cost.

        Args:
            executed_option (Option)
            final_state (State)

        """
        def modify(node, success):
            """ When successful, the cost of the edge is halved. When unsuccessful, the cost is doubled. """
            edge_weight = self.plan_graph.plan_graph[executed_option][node]["weight"]
            new_weight = (0.95 ** success) * edge_weight
            self.plan_graph.set_edge_weight(executed_option, node, new_weight)

        outgoing_edges = [edge for edge in self.plan_graph.plan_graph.edges if edge[0] == executed_option]
        neighboring_vertices = [edge[1] for edge in outgoing_edges]
        successfully_reached_vertices = [vertex for vertex in neighboring_vertices if vertex.is_init_true(final_state)]
        failed_reaching_vertices = [vertex for vertex in neighboring_vertices if not vertex.is_init_true(final_state)]

        for vertex in successfully_reached_vertices:
            modify(vertex, +1)

        for vertex in failed_reaching_vertices:
            modify(vertex, -1)

    def add_newly_created_option_to_plan_graph(self, newly_created_option, episode):

        assert newly_created_option.get_training_phase() == "initiation_done", \
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
        if chain.is_chain_completed() \
                and is_leaf_node \
                and chain.should_exist_edge_from_event_to_option(init_salient_event, newly_created_option):

            print(f"Case 2: Adding edge from {init_salient_event} to {newly_created_option}")
            self.plan_graph.add_edge(init_salient_event, newly_created_option, edge_weight=0.)

        # Case 1: Nominal case
        if newly_created_option.parent is not None:
            print(f"Case 1: Adding edge from {newly_created_option} to {newly_created_option.parent}")
            self.plan_graph.add_edge(newly_created_option, newly_created_option.parent, edge_weight=1.)

        # Case 4: I did not consider this case before. But, you could also intersect with any other
        # event in the MDP -- you might not rewire your chain because that event is not "completed/chained-to",
        # but that still means that there should be an edge from it to the newly learned option
        if self.chainer.event_intersection_salience:
            # 1. Get intersecting events
            # 2. Add an edge from each intersecting event to the newly_created_option
            events = [event for event in self.mdp.get_all_target_events_ever() if event != chain.target_salient_event]
            intersecting_events = [event for event in events
                                   if chain.should_exist_edge_from_event_to_option(event, newly_created_option)]
            for event in intersecting_events:  # type: SalientEvent
                assert isinstance(event, SalientEvent)
                print(f"Adding edge from {event} to {newly_created_option}")
                self.plan_graph.add_edge(event, newly_created_option, edge_weight=0.)

        elif self.chainer.option_intersection_salience:
            # 1. Get intersecting options
            # 2. Add an each from each intersecting option to the newly_created_option
            other_options = [o for o in self.get_options_in_known_part_of_the_graph() if o != newly_created_option]
            for other_option in other_options:  # type: Option
                if SkillChain.should_exist_edge_between_options(other_option, newly_created_option):
                    print(f"Adding edge from {other_option} to {newly_created_option}")
                    self.plan_graph.add_edge(other_option, newly_created_option)
                if SkillChain.should_exist_edge_between_options(newly_created_option, other_option):
                    print(f"Adding edge from {newly_created_option} to {other_option}")
                    self.plan_graph.add_edge(newly_created_option, other_option)

        self.update_chain_init_descendants()

        if chain.should_expand_initiation_classifier(newly_created_option):
            newly_created_option.expand_initiation_classifier(chain.init_salient_event)

        if chain.should_complete_chain(newly_created_option):
            chain.set_chain_completed()

        if chain.is_chain_completed():
            visualize_graph(self, episode, self.chainer.experiment_name, self.chainer.seed, True)

    # -----------------------------–––––––--------------
    # Utility Functions
    # -----------------------------–––––––--------------

    @staticmethod
    def get_closest_pair_of_vertices(src_vertices, dest_vertices):
        closest_pair = None
        min_distance = np.inf

        for source in src_vertices:  # type: SalientEvent or Option
            for target in dest_vertices:  # type: SalientEvent or Option
                assert source != target, f"{source, target}"
                distance = SkillGraphPlanner.distance_between_vertices(source, target)
                if distance < min_distance:
                    min_distance = distance
                    closest_pair = source, target

        return closest_pair

    @staticmethod
    def distance_between_vertices(v1, v2):
        if isinstance(v1, SalientEvent) and isinstance(v2, SalientEvent):
            return v1.distance_to_other_event(v2)
        if isinstance(v1, SalientEvent) and isinstance(v2, Option):
            return v1.distance_to_effect_set(v2.effect_set)
        if isinstance(v1, Option) and isinstance(v2, SalientEvent):
            return v2.distance_to_effect_set(v1.effect_set)  # Assuming a symmetric distance function here
        if isinstance(v1, Option) and isinstance(v2, Option):
            return SalientEvent.set_to_set_distance(v1.effect_set, v2.effect_set)
        raise NotImplementedError(f"{type(v1), type(v2)}")

    @staticmethod
    def is_state_inside_vertex(state, vertex):
        assert isinstance(state, (State, np.ndarray)), f"{type(state)}"
        assert isinstance(vertex, (Option, SalientEvent)), f"{type(vertex)}"

        # State satisfies the salient event
        if isinstance(vertex, SalientEvent):
            return vertex(state)

        # State is in the effect set of the option
        return vertex.is_term_true(state)

    def is_state_inside_any_vertex(self, state):
        assert isinstance(state, (State, np.ndarray)), f"{type(state)}"

        if any([event(state) for event in self.plan_graph.salient_nodes]):
            return True

        return any([option.is_in_effect_set(state) for option in self.plan_graph.option_nodes])

    def does_exist_chain_for_event(self, event):
        assert isinstance(event, SalientEvent)

        for chain in self.chainer.chains:  # type: SkillChain
            if chain.target_salient_event == event:
                return True
        return False

    def update_chain_init_descendants(self):
        for chain in self.chainer.chains:  # type: SkillChain
            descendants = self.plan_graph.get_reachable_nodes_from_source_node(chain.init_salient_event)
            ancestors = self.plan_graph.get_nodes_that_reach_target_node(chain.init_salient_event)
            chain.set_init_descendants(descendants)
            chain.set_init_ancestors(ancestors)

    def get_options_in_known_part_of_the_graph(self):
        completed_chains = [chain for chain in self.chainer.chains if chain.is_chain_completed()]

        # Grab the learned options from the completed chains
        known_options = []
        for chain in completed_chains:  # type: SkillChain
            chain_options = [option for option in chain.options if option.get_training_phase() == "initiation_done"
                             and option in self.plan_graph.option_nodes]
            known_options.append(chain_options)
        known_options = list(itertools.chain.from_iterable(known_options))
        return known_options

    @staticmethod
    def sample_from_vertex(vertex):
        assert isinstance(vertex, (Option, SalientEvent)), f"{type(vertex)}"

        if isinstance(vertex, Option):
            return vertex.get_goal_for_option_rollout(method="use_effect_set")
        if vertex.get_target_position() is not None:
            return vertex.get_target_position()
        return random.sample(vertex.trigger_points, k=1)[0]
