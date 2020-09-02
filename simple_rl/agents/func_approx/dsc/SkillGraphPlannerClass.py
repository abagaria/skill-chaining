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
from simple_rl.agents.func_approx.dsc.utils import make_chunked_value_function_plot, visualize_graph, visualize_mpc_rollout_result
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

    def choose_vertex_to_fall_off_graph_from(self, state, goal_salient_event, choose_among_events=False):
        """
        Given a salient event outside the graph, find its nearest neighbor in the graph. We will
        use our planning based control loop to reach that nearest neighbor and then use deep
        skill chaining the rest of the way.

        Args:
            state (State)
            goal_salient_event (SalientEvent)
            choose_among_events (bool)

        Returns:
            graph_vertex (SalientEvent or Option)

        """

        def _single_argmin(array):
            """ Convenience function - return a single min and argmin corresponding to `array`. """
            array = np.array(array) if isinstance(array, list) else array
            selected_idx = np.argmin(array)
            selected_idx = selected_idx[0] if isinstance(selected_idx, np.ndarray) else selected_idx
            return selected_idx, array[selected_idx]

        def _distance_choose_event_to_fall_off_graph_from(candidate_vertices):
            """ If we were restricted to jump off the graph only from salient events, which event should that be? """
            candidate_events = [vertex for vertex in candidate_vertices if isinstance(vertex, SalientEvent)]
            candidate_events = [event for event in candidate_events if event != goal_salient_event]
            if len(candidate_events) > 0:
                distances = [goal_salient_event.distance_to_other_event(event) for event in candidate_events]
                selected_idx, min_distance = _single_argmin(distances)
                selected_event = candidate_events[selected_idx]
                print(f"[Distance-Event-Selection] Chose {selected_event} to fall off the graph from")
                return selected_event, min_distance
            return None, None

        def _distance_choose_vertex_to_fall_off_graph_from(candidate_vertices):
            """ If we were allowed to jump off the graph from any vertex, which vertex should that be? """
            candidate_options = [vertex for vertex in candidate_vertices if isinstance(vertex, Option)]
            candidate_options = [option for option in candidate_options if len(option.effect_set) > 0]
            if len(candidate_options) > 0:
                distances = [goal_salient_event.distance_to_effect_set(option.effect_set) for option in candidate_options]
                closest_event, closest_event_distance = _distance_choose_event_to_fall_off_graph_from(candidate_vertices)
                selected_idx, closest_option_distance = _single_argmin(distances)
                selected_option = candidate_options[selected_idx]
                selected_vertex = selected_option if closest_option_distance < closest_event_distance else closest_event
                selected_min_distance = closest_option_distance if closest_option_distance < closest_event_distance else closest_event_distance
                print(f"[Distance-Vertex-Selection] Chose {selected_vertex} to fall off the graph from")
                return selected_vertex, selected_min_distance
            return None, None

        vertices_to_choose_from = self.plan_graph.get_reachable_nodes_from_source_state(state)

        if len(vertices_to_choose_from) > 0:
            if choose_among_events:
                chosen_event, _ = _distance_choose_event_to_fall_off_graph_from(vertices_to_choose_from)
                return chosen_event
            chosen_vertex, _ = _distance_choose_vertex_to_fall_off_graph_from(vertices_to_choose_from)
            return chosen_vertex

        return None

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

        print(f"Reverting to the DSC policy over options")
        return self.chainer.act(state, goal=sampled_goal, train_mode=True)

    def planner_rollout_inside_graph(self, *, state, goal_vertex, goal_salient_event,
                                     episode_number, step_number, eval_mode):
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
                                                                          policy_over_options_goal=poo_goal)

            # Book keeping
            state = deepcopy(self.mdp.cur_state)
            episodic_trajectory.append(option_transitions)
            option_trajectory.append((option, option_transitions))

        if self.use_her:
            self.chainer.perform_experience_replay(state, episodic_trajectory, option_trajectory, overall_goal=poo_goal)

        return step_number

    def run_loop(self, *, mpc, state, goal_salient_event, episode, step, eval_mode, to_reset):
        assert isinstance(state, State)
        assert isinstance(goal_salient_event, SalientEvent)
        assert isinstance(episode, int)
        assert isinstance(step, int)
        assert isinstance(eval_mode, bool)
        assert isinstance(to_reset, bool)

        if to_reset:
            self.reset(state)

        goal_vertex = deepcopy(goal_salient_event)

        # Revise the goal_salient_event if it cannot be reached from the current state
        if not self.plan_graph.does_path_exist(state, goal_salient_event):
            revised_goal = self.choose_vertex_to_fall_off_graph_from(state, goal_salient_event, choose_among_events=False)
            if revised_goal is not None:
                print(f"Revised goal vertex to {revised_goal}")
                goal_vertex = revised_goal
            if not self.does_exist_chain_for_event(goal_salient_event):
                self.create_goal_option_outside_graph(goal_salient_event)

        # Use the planner if there is a path to the revised goal node and we are not already at it
        ran_planner = False
        if self.plan_graph.does_path_exist(state, goal_vertex) and not self.is_state_inside_vertex(state, goal_vertex):
            ran_planner = True
            print(f"[Planner] Rolling out from {state.position} targeting {goal_vertex}")
            step = self.planner_rollout_inside_graph(state=state,
                                                     goal_vertex=goal_vertex,
                                                     goal_salient_event=goal_salient_event,
                                                     episode_number=episode,
                                                     step_number=step,
                                                     eval_mode=eval_mode)
            
           
            # 1) reach vertex <- only use this one for now
            # 2) reach other vertex (not intended) <- could use this, if closes node isn't connected well to goal
            # 3) completely outside of graph
            # Question: run MPC for 2, 3?

            state = deepcopy(self.mdp.cur_state)

        orig_goal_salient_event = deepcopy(goal_salient_event)
        # TODO investigate why MPC is being run when goal_vertex is not closest to goal_salient_event
        mpc_constraint = (ran_planner and goal_vertex.is_init_true(state) and not goal_salient_event(state) and not goal_salient_event.revised_by_mpc) or (not ran_planner and not goal_salient_event(state) and not goal_salient_event.revised_by_mpc) and (self.chainer.max_steps - step > 200) and (mpc.is_trained)
        if mpc_constraint:
            # TODO record train times for MPC
            steps_remaining = 200
            print("running mpc")
            new_state, steps_taken = mpc.mpc_rollout(self.mdp, 14000, 7, goal_salient_event.get_target_position(), max_steps=steps_remaining)
            step += steps_taken
            print("ending mpc at state {} in {}".format(new_state, steps_taken))
            visualize_mpc_rollout_result(self.chainer.chains, goal_salient_event.get_target_position(), state, new_state, steps_taken, episode, self.chainer.experiment_name)
            state = new_state # point state to new_state variable
            goal_salient_event.target_state = state
            goal_salient_event._initialize_trigger_points()
            goal_salient_event.revised_by_mpc = True

        if not orig_goal_salient_event(state) and step < self.chainer.max_steps:
            state, step = self.dsc_outside_graph(episode=episode,
                                                 num_steps=step,
                                                 goal_salient_event=orig_goal_salient_event)

        return step, orig_goal_salient_event(state)

    # -----------------------------–––––––--------------
    # Managing DSC Control Loops
    # -----------------------------–––––––--------------

    def dsc_outside_graph(self, episode, num_steps, goal_salient_event):
        dsc_goal_state = self.sample_from_vertex(goal_salient_event)
        dsc_run_loop_budget = self.chainer.max_steps - num_steps

        print(f"Rolling out DSC for {dsc_run_loop_budget} steps with interrupt handle {goal_salient_event} and goal {dsc_goal_state}")

        # Deliberately did not add `eval_mode` to the DSC rollout b/c we usually do this when we
        # want to fall off the graph, and it is always good to do some exploration when outside the graph
        score, step_number, newly_created_options = self.chainer.dsc_rollout(episode_number=episode,
                                                                             step_number=num_steps,
                                                                             interrupt_handle=goal_salient_event,
                                                                             overall_goal=dsc_goal_state)

        print(f"Returned from DSC runloop having learned {newly_created_options}")

        self.manage_options_learned_during_rollout(newly_created_options)

        return deepcopy(self.mdp.cur_state), step_number

    def perform_option_rollout(self, option, episode, step, eval_mode, policy_over_options_goal):

        state_before_rollout = deepcopy(self.mdp.cur_state)

        # Execute the option: if the option is the global-option, it doesn't have a default
        # goal in the task-agnostic setting. As a result, we ask it to target the same goal
        # as the policy-over-options.
        option_goal = policy_over_options_goal if option.name == "global_option" else None
        option_transitions, option_reward = option.execute_option_in_mdp(self.mdp,
                                                                         episode=episode,
                                                                         step_number=step,
                                                                         eval_mode=eval_mode,
                                                                         poo_goal=option_goal)

        new_options = self.chainer.manage_skill_chain_after_option_rollout(state_before_rollout=state_before_rollout,
                                                                           executed_option=option,
                                                                           created_options=[],
                                                                           episode_number=episode,
                                                                           option_transitions=option_transitions)

        state_after_rollout = deepcopy(self.mdp.cur_state)

        # With off-policy triggers, you can trigger the termination condition of an option w/o rolling it out
        self.manage_options_learned_during_rollout(newly_created_options=new_options)

        # Modify the edge weight associated with the executed option
        self.modify_edge_weight(executed_option=option, final_state=state_after_rollout)

        return step + len(option_transitions), option_transitions

    def manage_options_learned_during_rollout(self, newly_created_options):

        # TODO: Adding new options is not enough - we also have to create their children
        for newly_created_option in newly_created_options:  # type: Option
            self.add_newly_created_option_to_plan_graph(newly_created_option)

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
            new_weight = 2 ** (-int(success)) * edge_weight
            self.plan_graph.set_edge_weight(executed_option, node, new_weight)

        outgoing_edges = [edge for edge in self.plan_graph.plan_graph.edges if edge[0] == executed_option]
        neighboring_vertices = [edge[1] for edge in outgoing_edges]
        successfully_reached_vertices = [vertex for vertex in neighboring_vertices if vertex.is_init_true(final_state)]
        failed_reaching_vertices = [vertex for vertex in neighboring_vertices if not vertex.is_init_true(final_state)]

        for vertex in successfully_reached_vertices:
            modify(vertex, True)

        for vertex in failed_reaching_vertices:
            modify(vertex, False)

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

        if chain.is_chain_completed(self.chainer.chains):
            visualize_graph(self.chainer.chains, self.chainer.experiment_name, True)

    # -----------------------------–––––––--------------
    # Utility Functions
    # -----------------------------–––––––--------------

    @staticmethod
    def is_state_inside_vertex(state, vertex):
        assert isinstance(state, (State, np.ndarray)), f"{type(state)}"
        assert isinstance(vertex, (Option, SalientEvent)), f"{type(vertex)}"

        # State satisfies the salient event
        if isinstance(vertex, SalientEvent):
            return vertex(state)

        # State is in the effect set of the option
        return vertex.is_term_true(state)

    def does_exist_chain_for_event(self, event):
        assert isinstance(event, SalientEvent)

        for chain in self.chainer.chains:  # type: SkillChain
            if chain.target_salient_event == event:
                return True
        return False

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

    def create_goal_option_outside_graph(self, target_salient_event):
        chain_id = max([chain.chain_id for chain in self.chainer.chains]) + 1

        # The init salient event of this new chain is initialized to be the start state of the MDP
        # However, as DSC adds skills to this chain, it will eventually stop when it finds a chain intersection
        # At that point it will rewire the chain to set its init_salient_event to be the region of intersection
        init_salient_event = self.mdp.get_start_state_salient_event()
        new_skill_chain = SkillChain(init_salient_event=init_salient_event,
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
            self.pretrain_option_policy(new_untrained_option)

        self.chainer.untrained_options.append(new_untrained_option)
        self.chainer.augment_agent_with_new_option(new_untrained_option, 0.)

        # Note: we are not adding the new option to the plan-graph

        return new_untrained_option

    def pretrain_option_policy(self, new_untrained_option):
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

    @staticmethod
    def sample_from_vertex(vertex):
        assert isinstance(vertex, (Option, SalientEvent)), f"{type(vertex)}"

        if isinstance(vertex, Option):
            return vertex.get_goal_for_option_rollout(method="use_effect_set")
        if vertex.get_target_position() is not None:
            return vertex.get_target_position()
        return random.sample(vertex.trigger_points, k=1)[0]

    # -----------------------------–––––––--------------
    # Evaluation Functions
    # -----------------------------–––––––--------------

    def reset(self, start_state=None):
        """ Reset the MDP to either the default start state or to the specified one. """
        self.mdp.reset()

        if start_state is not None:
            start_position = start_state.position if isinstance(start_state, State) else start_state[:2]
            self.mdp.set_xy(start_position)

    def measure_success(self, goal_salient_event, start_state=None, num_episodes=100):
        successes = 0
        self.chainer.create_backward_options = False
        self.chainer.learn_backward_options_offline = False  # TODO: Need a better way to say that we should not create back options at test time
        for episode in range(num_episodes):
            num_steps, reached_goal = self.run_loop(state=start_state,
                                                    goal_salient_event=goal_salient_event,
                                                    episode=episode,
                                                    step=0,
                                                    eval_mode=True,
                                                    to_reset=True)
            if reached_goal:
                successes += 1
        return successes
