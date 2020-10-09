import ipdb
import random
import logging
import itertools
import numpy as np
from copy import deepcopy
from simple_rl.mdp.StateClass import State
from simple_rl.agents.func_approx.dsc.SkillChainingAgentClass import SkillChaining
from simple_rl.agents.func_approx.dsc.ChainClass import SkillChain
from simple_rl.agents.func_approx.dsc.OptionClass import Option
from simple_rl.agents.func_approx.dsc.SalientEventClass import SalientEvent, DSCOptionSalientEvent
from simple_rl.agents.func_approx.dsc.utils import make_chunked_value_function_plot, visualize_graph, visualize_mpc_rollout_and_graph_result
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

        self.logger = logging
        for handler in self.logger.root.handlers[:]:
            self.logger.root.removeHandler(handler)
        self.logger.basicConfig(filename=f"value_function_plots/{self.chainer.experiment_name}/SkillGraphPlanner.log", filemode='w', format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p', level=logging.DEBUG)

    # -----------------------------–––––––--------------
    # Control loop methods
    # -----------------------------–––––––--------------

    def choose_closest_source_target_vertex_pair(self, state, goal_salient_event):
        """
        Given a salient event outside the graph, find its nearest neighbor in the graph. We will
        use our planning based control loop to reach that nearest neighbor and then use deep
        skill chaining the rest of the way.

        Args:
            state (State)
            goal_salient_event (SalientEvent)

        Returns:
            closest_source (SalientEvent or Option)
            closest_target (SalientEvent or Option)
        """
        candidate_vertices_to_fall_from = self.plan_graph.get_reachable_nodes_from_source_state(state)
        candidate_vertices_to_jump_to = self.plan_graph.get_nodes_that_reach_target_node(goal_salient_event)
        candidate_vertices_to_jump_to = list(candidate_vertices_to_jump_to) + [goal_salient_event]

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
        return self.chainer.act(state, goal=sampled_goal, train_mode=True)

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
        should_terminate = lambda s, inside, outside, step: inside(s) or outside(s) or step >= self.chainer.max_steps
        inside_predicate = goal_vertex if isinstance(goal_vertex, SalientEvent) else goal_vertex.is_term_true

        while not should_terminate(state, inside_predicate, goal_salient_event, step_number):
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

    def mpc_revise_salient_event(self, mpc, state, goal_salient_event, step, episode, max_steps):
        if step + max_steps > self.chainer.max_steps:
            self.logger.debug(f'Episode {episode}: Not enough budget to run MPC, terminating early')
            return state, self.chainer.max_steps, False # ends episode
        
        self.logger.debug(f'Episode {episode}: MPC starting at {(state[0], state[1])}')
        mpc_result_state, steps_taken = mpc.rollout(self.mdp, 14000, 7, goal_salient_event.get_target_position(), max_steps=max_steps)
        self.logger.debug(f'Episode {episode}: MPC ended at {(mpc_result_state[0], mpc_result_state[1])}')

        reject_mpc = self.should_reject_mpc_rollout(mpc_result_state, goal_salient_event)
        visualize_mpc_rollout_and_graph_result(self, self.mdp, self.chainer.chains, goal_salient_event.get_target_position(), state, mpc_result_state, steps_taken, episode, reject_mpc, self.chainer.experiment_name, True)

        if reject_mpc:
            self.logger.debug(f'Episode {episode}: MPC rollout rejected')
            self.mdp.all_salient_events_ever.remove(goal_salient_event)
            return mpc_result_state, self.chainer.max_steps, True
        
        goal_salient_event.target_state = mpc_result_state.position
        goal_salient_event._initialize_trigger_points()
        goal_salient_event.revised_by_mpc = True
        self.logger.debug(f'Episode {episode}: MPC rollout succeeded, revised goal_salient_event!')

        return mpc_result_state, step + steps_taken, False
            

    def run_loop(self, *, mpc, state, goal_salient_event, episode, step, eval_mode):
        # TODO add assert statement here for mpc?
        assert isinstance(state, State)
        assert isinstance(goal_salient_event, SalientEvent)
        assert isinstance(episode, int)
        assert isinstance(step, int)
        assert isinstance(eval_mode, bool)

        if episode != 0 and episode % 250 == 0:
            for option in self.chainer.trained_options[1:]:
                if option.parent is None and option.get_training_phase() == "initiation_done":
                    self.logger.debug(f'Episode {episode}: option success rates {option}, {option.get_option_success_rate()}')

        planner_goal_vertex, dsc_goal_vertex = self._get_goal_vertices_for_rollout(state, goal_salient_event)
        print(f"Planner goal: {planner_goal_vertex}, DSC goal: {dsc_goal_vertex} and Goal: {goal_salient_event}")
        self.logger.debug(f'Episode {episode}: goal_salient_event at {(goal_salient_event.target_state[0], goal_salient_event.target_state[1])}')
        if isinstance(planner_goal_vertex, SalientEvent):
            self.logger.debug(f'Episode {episode}: planner_goal_vertex at {(planner_goal_vertex.target_state[0], planner_goal_vertex.target_state[1])}')
        else:
            self.logger.debug(f'Episode {episode}: planner_goal_vertex is option {planner_goal_vertex}')
        if isinstance(dsc_goal_vertex, SalientEvent):
            self.logger.debug(f'Episode {episode}: dsc_goal_vertex at {(dsc_goal_vertex.target_state[0], dsc_goal_vertex.target_state[1])}')
        else:
            self.logger.debug(f'Episode {episode}: dsc_goal_vertex is option {dsc_goal_vertex}')

        if eval_mode is False and not goal_salient_event.revised_by_mpc and self.mdp.get_start_state_salient_event()(state):
            print("running mpc")
            if not goal_salient_event == planner_goal_vertex:
                state, step = self.run_sub_loop(state, planner_goal_vertex, step, goal_salient_event, episode, eval_mode)
            state, step, rejected = self.mpc_revise_salient_event(mpc, state, goal_salient_event, step, episode, 100) # TODO change back for ant-maze
            return self.chainer.max_steps, not rejected, rejected
        else:
            state, step = self.run_sub_loop(state, planner_goal_vertex, step, goal_salient_event, episode, eval_mode)
            state, step = self.run_sub_loop(state, dsc_goal_vertex, step, goal_salient_event, episode, eval_mode)
            state, step = self.run_sub_loop(state, goal_salient_event, step, goal_salient_event, episode, eval_mode)
        return step, goal_salient_event(state), False

    def test_loop(self, *, state, goal_salient_event, test_option, episode, step, eval_mode):
        assert isinstance(state, State)
        assert isinstance(goal_salient_event, SalientEvent)
        assert isinstance(test_option, Option)
        assert isinstance(episode, int)
        assert isinstance(step, int)
        assert isinstance(eval_mode, bool)

        def find_closest_node_in_graph(s):
            point = self.mdp.get_position(s)
            min_distance, best_node = np.inf, None
            for node in self.plan_graph.plan_graph.nodes:
                assert isinstance(node, (SalientEvent, Option))
                states = node.trigger_points if isinstance(node, SalientEvent) else node.effect_set
                points = [self.mdp.get_position(_state) for _state in states]
                distance = SalientEvent.point_to_set_distance(point, points)
                if distance < min_distance:
                    min_distance = distance
                    best_node = node
            return best_node, min_distance

        def find_closest_point_in_graph(s):
            point = self.mdp.get_position(s)
            closest_node, _ = find_closest_node_in_graph(s)
            if closest_node is None: ipdb.set_trace()
            states = closest_node.trigger_points if isinstance(closest_node, SalientEvent) else closest_node.effect_set
            positions = np.array([self.mdp.get_position(_state) for _state in states])
            closest_idx = np.argmin(np.sum((point - positions) ** 2, axis=1))
            closest_point = positions[closest_idx]
            return closest_point

        # Set the training-phase of the goal option to be gestation always
        test_option.num_goal_hits = 0

        nearest_point_in_graph = find_closest_point_in_graph(state)
        print(f"Targeting {nearest_point_in_graph} with the global-option")

        while len(self.plan_graph.get_reachable_nodes_from_source_state(state)) == 0 \
            and not goal_salient_event(state) and not step < self.chainer.max_steps:
            step, transitions = self.perform_option_rollout(self.chainer.global_option,
                                                            episode, step, eval_mode,
                                                            nearest_point_in_graph, goal_salient_event)
            state = deepcopy(self.mdp.cur_state)
        print(f"Reachable nodes from {self.mdp.get_position(state)}: {self.plan_graph.get_reachable_nodes_from_source_state(state)}")

        planner_goal_vertex, dsc_goal_vertex = self._get_goal_vertices_for_rollout(state, goal_salient_event)
        print(f"Planner goal: {planner_goal_vertex}, DSC goal: {dsc_goal_vertex} and Goal: {goal_salient_event}")
        state, step = self.run_sub_loop(state, planner_goal_vertex, step, goal_salient_event, episode, eval_mode)

        print(f"Targeting {goal_salient_event} with {test_option}")
        while not goal_salient_event(state) and step < self.chainer.max_steps and not test_option.is_term_true(state):
            step, transitions = self.perform_option_rollout(test_option, episode, step, eval_mode,
                                                            goal_salient_event.get_target_position(),
                                                            goal_salient_event)
            state = deepcopy(self.mdp.cur_state)

        return step, goal_salient_event(state)

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
                                                                             overall_goal=dsc_goal_state)

        print(f"Returned from DSC runloop having learned {newly_created_options}")

        self.manage_options_learned_during_rollout(newly_created_options, episode)

        return deepcopy(self.mdp.cur_state), step_number

    def perform_option_rollout(self, option, episode, step, eval_mode, policy_over_options_goal, goal_salient_event):

        state_before_rollout = deepcopy(self.mdp.cur_state)

        # Don't (off-policy) trigger the termination condition of options targeting the event we are currently leaving
        self.chainer.disable_triggering_options_targeting_init_event(state=state_before_rollout)

        # TODO: Always running ant with some exploration
        modified_eval_mode = eval_mode if "point" in self.mdp.env_name else False

        # Execute the option: if the option is the global-option, it doesn't have a default
        # goal in the task-agnostic setting. As a result, we ask it to target the same goal
        # as the policy-over-options.
        option_goal = policy_over_options_goal if option.name == "global_option" else None
        option_transitions, option_reward = option.execute_option_in_mdp(self.mdp,
                                                                         episode=episode,
                                                                         step_number=step,
                                                                         eval_mode=modified_eval_mode,
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
        dsc_goal_vertex = deepcopy(goal_salient_event)
        planner_goal_vertex = deepcopy(goal_salient_event)

        # Revise the goal_salient_event if it cannot be reached from the current state
        if not self.plan_graph.does_path_exist(state, goal_salient_event):
            closest_vertex_pair = self.choose_closest_source_target_vertex_pair(state, goal_salient_event)
            if closest_vertex_pair is not None:
                planner_goal_vertex, dsc_goal_vertex = closest_vertex_pair
                print(f"Revised planner goal vertex to {planner_goal_vertex} and dsc goal vertex to {dsc_goal_vertex}")

        return planner_goal_vertex, dsc_goal_vertex


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

    def should_reject_mpc_rollout(self, target_state, goal_salient_event):
        """
        Reject the discovered salient event if it is the same as an old one or inside the initiation set of an option.
        Args:
            salient_event (SalientEvent)
        Returns:
            should_reject (bool)
        """
        existing_target_events = self.mdp.get_all_target_events_ever() + [self.mdp.get_start_state_salient_event()] # start state isn't revised by MPC
        if any([event(target_state) for event in existing_target_events if event != goal_salient_event and event.revised_by_mpc]): # TODO debug
            self.logger.debug('mpc rollout rejected because similar salient event already exists')
            return True

        if self.chainer.state_in_any_completed_option(target_state):
            self.logger.debug('mpc rollout rejected because state is in a completed option')
            return True

        return False

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
