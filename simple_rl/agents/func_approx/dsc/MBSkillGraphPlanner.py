import ipdb
import torch
import random
import itertools
import numpy as np
from copy import deepcopy
from scipy.special import softmax
from scipy.spatial.distance import cdist
from simple_rl.mdp.StateClass import State
from simple_rl.mdp.GoalDirectedMDPClass import GoalDirectedMDP
from simple_rl.agents.func_approx.dsc.ModelBasedDSC import ModelBasedSkillChaining
from simple_rl.agents.func_approx.dsc.ChainClass import SkillChain
from simple_rl.agents.func_approx.dsc.MBOptionClass import ModelBasedOption
from simple_rl.agents.func_approx.dsc.SalientEventClass import SalientEvent
from simple_rl.agents.func_approx.dsc.utils import visualize_graph, visualize_mpc_rollout_and_graph_result, plot_trajectory, visualize_graph_nodes_with_expansion_probabilities, visualize_graph_nodes_with_vf_expansion_probabilities
from simple_rl.agents.func_approx.dsc.PlanGraphClass import PlanGraph
from simple_rl.agents.func_approx.td3.TD3RNDAgentClass import TD3
from simple_rl.agents.func_approx.td3.utils import make_chunked_value_function_plot
from simple_rl.agents.func_approx.dsc.dynamics.mpc import MPC
from simple_rl.agents.func_approx.dsc.dynamics.mpc_rnd import MPCRND


class SkillGraphPlanner(object):
    def __init__(self, *, mdp, chainer, use_vf, extrapolator, experiment_name, seed):
        """
        The Skill Graph Planning Agent uses graph search to get to target states given
        during test time. If the goal state is in a salient event already known to the agent,
        it can get there without any learning involved. If it is not a known salient event,
        it will create new skill chains to reliably target it.
        Args:
            mdp (GoalDirectedMDP)
            chainer (ModelBasedSkillChaining)
            experiment_name (str)
            seed (int)
        """
        self.mdp = mdp
        self.chainer = chainer  # type: ModelBasedSkillChaining
        self.use_vf = use_vf
        self.extrapolator = extrapolator
        self.experiment_name = experiment_name
        self.seed = seed

        self.mpc = self._get_mpc()
        self.exploration_agent = self._get_exploration_agent()

        assert isinstance(self.chainer, ModelBasedSkillChaining)
        assert isinstance(self.mdp, GoalDirectedMDP)
        assert extrapolator in ("model-free", "model-based"), extrapolator

        self.plan_graph = PlanGraph()

    def _get_exploration_agent(self):
        if self.extrapolator == "model-free":
            return TD3(state_dim=self.mdp.state_space_size(),
                    action_dim=self.mdp.action_space_size(),
                    max_action=1.,
                    name="td3-rnd-exploration-agent",
                    device=self.chainer.device,
                    augment_with_rewards=False,
                    use_obs_normalization=False,
                    )

        return MPCRND(mdp=self.mdp,
                      state_size=self.mdp.state_space_size(),
                      action_size=self.mdp.action_space_size(),
                      device=self.chainer.device,
                      multithread=self.chainer.global_option.multithread_mpc)

    def _get_mpc(self):
        if self.chainer.use_model:
            print(f"[Model-Based Controller] Using the global-option's solver as DSG's MPC.")
            return self.chainer.global_option.solver

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
        candidate_vertices_to_fall_from = list(candidate_vertices_to_fall_from) + self.get_corresponding_events(state)
        candidate_vertices_to_jump_to = self.plan_graph.get_nodes_that_reach_target_node(goal_salient_event)
        candidate_vertices_to_jump_to = list(candidate_vertices_to_jump_to) + [goal_salient_event]

        if choose_among_events:
            candidate_vertices_to_fall_from = [v for v in candidate_vertices_to_fall_from if isinstance(v, SalientEvent)]
            candidate_vertices_to_jump_to = [v for v in candidate_vertices_to_jump_to if isinstance(v, SalientEvent)]

        candidate_vertices_to_fall_from = list(set(candidate_vertices_to_fall_from))
        candidate_vertices_to_jump_to = list(set(candidate_vertices_to_jump_to))

        return self.get_closest_pair_of_vertices(candidate_vertices_to_fall_from, candidate_vertices_to_jump_to)

    def act(self, state, goal_vertex, sampled_goal):
        assert isinstance(state, (State, np.ndarray)), f"{type(state)}"
        assert isinstance(goal_vertex, (SalientEvent, ModelBasedOption)), f"{type(goal_vertex)}"

        def _pick_option_to_execute_from_plan(selected_plan):
            admissible_options = [o for o in selected_plan if o.is_init_true(state) and not o.is_term_true(state)]
            if len(admissible_options) > 0:
                print(f"Got plan {selected_plan} and chose to execute {admissible_options[0]}")
                return admissible_options[0]  # type: ModelBasedOption
            return None

        if self.plan_graph.does_path_exist(state, goal_vertex):
            plan = self.plan_graph.get_path_to_execute(state, goal_vertex)
            if len(plan) > 0:
                selected_option = _pick_option_to_execute_from_plan(plan)  # type: ModelBasedOption
                if selected_option is not None:
                    return selected_option

        print(f"Reverting to the DSC policy over options targeting {sampled_goal}")
        option = self.chainer.act(state, goal=sampled_goal)[0]

        return option

    def planner_rollout_inside_graph(self, *, state, goal_vertex, goal_salient_event,
                                     episode_number, step_number, eval_mode):
        """ Control loop that drives the agent from `state` to `goal_vertex` or times out. """
        assert isinstance(state, State)
        assert isinstance(goal_vertex, (SalientEvent, ModelBasedOption))
        assert isinstance(episode_number, int)
        assert isinstance(step_number, int)
        assert isinstance(eval_mode, bool)

        # Note: I removed an assertion that goal_salient_event be a SalientEvent b/c in the absence of 
        #       a state sampler, goal_vertex and goal_salient_event are the same node and they could both be options

        episodic_trajectory = []
        option_trajectory = []

        poo_goal = self.sample_from_vertex(goal_vertex)
        should_terminate = lambda s, o, inside, outside, step: inside(s, o) or outside(s, o) or step >= self.chainer.max_steps
        inside_predicate = lambda s, o: goal_vertex(s) if isinstance(goal_vertex, SalientEvent) else goal_vertex.is_term_true(s)
        outside_predicate = lambda s, o: goal_salient_event(s)

        option = None

        while not should_terminate(state, option, inside_predicate, outside_predicate, step_number):
            option = self.act(state, goal_vertex, sampled_goal=poo_goal)

            # Revise the sub-goal for the global-option
            if option.global_init:
                revised_goal = self.pick_subgoal_for_global_option(state, goal_vertex)
                if revised_goal is not None:
                    poo_goal = revised_goal

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
            if option.global_init and self.mdp.dense_gc_reward_function(state, poo_goal, {})[1]:
                break

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
        filtering_condition = lambda v: isinstance(v, SalientEvent)

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

    def salient_event_discovery_run_loop(self, episode):
        assert isinstance(episode, int)

        step = 0
        state = deepcopy(self.mdp.cur_state)
        planner_goal_vertex = self.get_node_to_expand(state)
        print(f"[Salient-Event-Discovery] Planner goal: {planner_goal_vertex}")

        # Run loop to get to the planner-goal-vertex
        if not planner_goal_vertex(state):
            step, success = self.run_loop(state=state,
                                          goal_salient_event=planner_goal_vertex,
                                          episode=episode,
                                          step=step,
                                          eval_mode=False)
            state = deepcopy(self.mdp.cur_state)

        # Run loop that does exploration outside the graph
        if planner_goal_vertex(state) or self.mdp.get_start_state_salient_event()(state):
            extrapolation_func = self.model_free_extrapolation if self.extrapolator == "model-free" else self.model_based_extrapolation
            state, steps_taken = extrapolation_func(episode, step)
            return state, step + steps_taken

        return state, step

    def model_free_extrapolation(self, episode, step_number):
        """ Single episodic rollout of the exploration policy to extend the graph. """

        transitions = []
        intrinsic_reward = 0.
        step_budget = max(200, self.chainer.max_steps - step_number)
        state = deepcopy(self.mdp.cur_state)

        events = self.mdp.all_salient_events_ever + [self.mdp.get_start_state_salient_event()]
        starting_nodes = [event for event in events if event(state)]

        print(f"Performing model-free extrapolation from {state.position} and {starting_nodes} for {step_budget} steps")

        for step in range(self.chainer.max_steps):
            augmented_state = self.exploration_agent.get_augmented_state(state, intrinsic_reward)
            action = self.exploration_agent.act(augmented_state)
            reward, next_state = self.mdp.execute_agent_action(action)
            intrinsic_reward = self.exploration_agent.get_intrinsic_reward(next_state.features())
            augmented_next_state = self.exploration_agent.get_augmented_state(next_state, intrinsic_reward)
            
            # [RND] Done flag is always false for the intrinsic value function
            # [RND] Setting the extrinsic reward to zero to only engage the intrinsic value head of RND value function
            self.exploration_agent.step(augmented_state, action, intrinsic_reward, 0., augmented_next_state, False)

            transitions.append((state, action, intrinsic_reward, next_state))
            state = next_state
            
            if state.is_terminal():
                break

        make_chunked_value_function_plot(self.exploration_agent, episode, self.experiment_name)

        target_state = self.create_target_state(transitions)
        self._update_nodes_with_monte_carlo_exploration_return(starting_nodes, transitions)

        return target_state, len(transitions)

    def model_based_extrapolation(self, episode, step_number):
        transitions = []
        min_step = max(100, (episode // 100) * 100)
        step_budget = min(min_step, self.chainer.max_steps - step_number)
        state = deepcopy(self.mdp.cur_state)

        events = self.mdp.all_salient_events_ever + [self.mdp.get_start_state_salient_event()]
        starting_nodes = [event for event in events if event(state)]

        print(f"Performing model-based extrapolation from {state.position} from {starting_nodes} for {step_budget} steps")

        for step in range(self.chainer.max_steps):
            action = self.exploration_agent.act(state, num_steps=21)  # Trying a longer horizon for action selection
            reward, next_state = self.mdp.execute_agent_action(action)
            intrinsic_reward = self.exploration_agent.get_intrinsic_reward(next_state.features())
            transitions.append((state, action, intrinsic_reward, next_state))
            state = next_state

            if state.is_terminal():
                break

        # Update RND intrinsic reward function
        self.update_exploration_agent(transitions, False)
        self._update_nodes_with_monte_carlo_exploration_return(starting_nodes, transitions)

        target_state = self.create_target_state(transitions)
        return target_state, len(transitions)

    def create_target_state(self, transitions):
        best_state = None
        max_intrinsic_reward = -np.inf

        for state, action, intrinsic_reward, next_state in transitions:
            
            if intrinsic_reward > max_intrinsic_reward:
                best_state = next_state
                max_intrinsic_reward = intrinsic_reward

        return best_state

    @staticmethod
    def _update_nodes_with_monte_carlo_exploration_return(nodes, transitions):
        """ After the exploration rollout, update the MC returns of the nodes exploration started from. """

        for node in nodes:
            node.expansion_count += 1
            mean_intrinsic_reward = np.mean([transition[2] for transition in transitions])
            node.exploration_returns.append(mean_intrinsic_reward)

    def update_exploration_agent(self, transitions, recompute_int_rewards):
        for state, action, intrinsic_reward, next_state in transitions:
            if recompute_int_rewards:
                intrinsic_reward = self.exploration_agent.get_intrinsic_reward(next_state.features())
            self.exploration_agent.step(state.features(), action, intrinsic_reward, next_state.features(), next_state.is_terminal())

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
        num_steps_taken, newly_created_options = self.chainer.dsc_rollout(episode_number=episode,
                                                                          num_steps=dsc_run_loop_budget,
                                                                          interrupt_handle=dsc_interrupt_handle,
                                                                          goal=dsc_goal_state)
        step_number = num_steps + num_steps_taken
        print(f"Returned from DSC runloop having learned {newly_created_options}")

        self.manage_options_learned_during_rollout(newly_created_options, episode)

        return deepcopy(self.mdp.cur_state), step_number

    def perform_option_rollout(self, option, episode, step, eval_mode, policy_over_options_goal, goal_salient_event):
        assert isinstance(option, ModelBasedOption)

        state_before_rollout = deepcopy(self.mdp.cur_state)

        # Don't (off-policy) trigger the termination condition of options targeting the event we are currently leaving
        self.chainer.disable_triggering_options_targeting_init_event(state=state_before_rollout)

        # Execute the option: if the option is the global-option, it doesn't have a default
        # goal in the task-agnostic setting. As a result, we ask it to target the same goal
        # as the policy-over-options.
        # Note that we are setting eval_mode to False because we need new options to update their VF and init sets
        option_goal = policy_over_options_goal if option.global_init else None
        option_transitions, option_reward = option.rollout(episode=episode, step_number=step,
                                                           rollout_goal=option_goal, eval_mode=False)

        new_options = self.chainer.manage_chain_after_option_rollout(executed_option=option,
                                                                     created_options=[],
                                                                     episode=episode,
                                                                     option_transitions=option_transitions)

        state_after_rollout = deepcopy(self.mdp.cur_state)

        # With off-policy triggers, you can trigger the termination condition of an option w/o rolling it out
        self.manage_options_learned_during_rollout(newly_created_options=new_options, episode=episode)

        # Remove edges that no longer make sense for the executed option
        self.modify_node_connections(executed_option=option)

        # Modify the edge weight associated with the executed option
        self.modify_edge_weight(executed_option=option, final_state=state_after_rollout)

        # Update the RND exploration agent
        if self.extrapolator == "model-based":
            self.update_exploration_agent(option_transitions, True)

        return step + len(option_transitions), option_transitions

    def manage_options_learned_during_rollout(self, newly_created_options, episode):

        # TODO: Adding new options is not enough - we also have to create their children
        for newly_created_option in newly_created_options:  # type: ModelBasedOption
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

    def compute_intrinsic_reward_score(self, node):
        assert isinstance(node, (SalientEvent, ModelBasedOption)), node

        state_set = list(node.trigger_points) if isinstance(node, SalientEvent) else node.effect_set

        if self.extrapolator == "model-free":
            states = [s.features() if isinstance(s, State) else s for s in state_set]
            states = np.array([s for s in states if s.shape == (self.mdp.state_space_size(),)])
            scores = self.exploration_agent.value_function(states)
        else:
            states = random.sample(state_set, k=min(len(state_set), 10))
            states = [s.features() if isinstance(s, State) else s for s in states]
            states = np.array([s for s in states if s.shape == (self.mdp.state_space_size(),)])
            scores = [self.exploration_agent.value_function(s) for s in states]
        return np.mean(scores)

    def get_candidate_nodes_for_expansion(self):
        """ Return the nodes with the Top-K expansion scores. """
        descendants = list(set(self.mdp.get_all_target_events_ever()))
        return descendants

    def get_node_to_expand(self, state, temperature=1.):
        """ Given current `state`, use the RND intrinsic reward to find the graph node to expand. """
        
        descendants = self.get_candidate_nodes_for_expansion()

        if len(descendants) > 0:
            print(f"Nodes: {descendants}")
            rnd_scores = self.get_rnd_scores(descendants)
            distance_scores = self.get_graph_distance_scores(descendants)
            count_scores = self.get_count_based_scores(descendants)
            scores = self.combine_scores(rnd_scores, distance_scores, count_scores)
            sampled_node = self.score_based_node_selection(descendants, scores, temperature=temperature)
            return sampled_node
        
        assert len(descendants) == 0, descendants
        return self.mdp.get_start_state_salient_event()

    def get_rnd_scores(self, nodes, rnd_version="monte-carlo"):
        """ Get the normalized RND value function score for each input node. """
        assert rnd_version in ("none", "monte-carlo", "vf"), rnd_version

        if rnd_version == "monte-carlo":
            means = np.array([np.mean(n.exploration_returns) if len(n.exploration_returns) > 0 else 0. for n in nodes])
            return means / means.max()

        if rnd_version == "vf":
            scores = np.array([self.compute_intrinsic_reward_score(n) for n in nodes])
            return scores / scores.max()

        assert rnd_version == "none", rnd_version
        return np.zeros((len(nodes),))

    def get_graph_distance_scores(self, nodes):
        """ Bias the search for expansion nodes based on how far away (on the graph) nodes are to the start node. """

        def _get_path_length(len_dict, node):
            if node in len_dict:
                return len_dict[node]
            return np.inf

        if self.mdp.get_start_state_salient_event() not in self.plan_graph.plan_graph.nodes:
            return np.zeros((len(nodes),))

        path_lengths = self.plan_graph.get_shortest_path_lengths(use_edge_weights=False,
                                                                 source_node=self.mdp.get_start_state_salient_event())

        distances = np.array([_get_path_length(path_lengths, n) for n in nodes])  # Expect this to be 1-100ish
        non_inf_distances = distances[np.isfinite(distances)]

        if len(non_inf_distances) == 0:
            return np.zeros((len(nodes),))

        max_distance = non_inf_distances.max()  # exclude inf in the max computation

        distances[distances == np.inf] = max_distance + 1.
        max_distance = distances.max()  # Recompute the max with no infinities; this is for the normalization factor

        normalization_factor = max_distance if max_distance > 0. else 1e-6
        normalized_distances = distances / normalization_factor

        if any([d == np.inf for d in normalized_distances]):
            ipdb.set_trace()

        return normalized_distances

    @staticmethod
    def get_count_based_scores(nodes):
        """ UCB style count-based bonus to encourage trying different nodes for expansion. """

        def _compute_score_bonus(node):
            assert isinstance(node, SalientEvent), "NotImplemented for Options yet"
            return 1. / np.sqrt(node.expansion_count + 1e-3)

        return np.array([_compute_score_bonus(n) for n in nodes])

    @staticmethod
    def combine_scores(rnd_scores, dist_scores, count_scores):
        assert isinstance(rnd_scores, np.ndarray)
        assert isinstance(dist_scores, np.ndarray)
        assert isinstance(count_scores, np.ndarray)
        assert rnd_scores.all() <= 1., rnd_scores
        assert dist_scores.all() <= 1., dist_scores
        assert count_scores.all() <= 1., count_scores

        rnd_coefficient = 1.
        distance_coefficient = 0.1
        count_coefficient = np.sqrt(2.)

        print(f"[Coef: {rnd_coefficient}] RND scores: {rnd_scores}")
        print(f"[Coef: {distance_coefficient}] Distance scores: {dist_scores}")
        print(f"[Coef: {count_coefficient}] Count scores: {count_scores}")

        return (rnd_coefficient * rnd_scores) + \
               (distance_coefficient * dist_scores) + \
               (count_coefficient * count_scores)

    def score_based_node_selection(self, nodes, scores, temperature, choice_type="deterministic"):
        assert temperature > 0., temperature
        assert isinstance(nodes, list), nodes
        assert isinstance(scores, np.ndarray), scores
        assert choice_type in ("deterministic", "stochastic"), choice_type

        if choice_type == "deterministic":
            return self._deterministic_pick_node(nodes, scores)
        return self._stochastic_pick_node(nodes, scores, temperature)

    @staticmethod
    def _deterministic_pick_node(nodes, scores):
        idx = np.argmax(scores)
        node = nodes[random.choice(idx) if isinstance(idx, np.ndarray) else idx]
        return node

    @staticmethod
    def _stochastic_pick_node(nodes, scores, temperature=1.):
        probabilities = softmax(scores / temperature)

        assert all(probabilities.tolist()) <= 1., probabilities
        np.testing.assert_almost_equal(probabilities.sum(), 1., err_msg=f"{probabilities}", decimal=3)

        sampled_node = np.random.choice(nodes, size=1, p=probabilities)[0]
        print(f"[Temp={temperature}] | Descendants: {nodes} | Probs: {probabilities} | Chose: {sampled_node}")
        return sampled_node

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

    def _delete_outgoing_edges_if_needed(self, executed_option):
        assert isinstance(executed_option, ModelBasedOption)

        outgoing_nodes = self.plan_graph.get_outgoing_nodes(executed_option)

        for node in outgoing_nodes:
            assert isinstance(node, (ModelBasedOption, SalientEvent))

            if isinstance(node, ModelBasedOption):
                should_remove = not SkillChain.should_exist_edge_between_options(executed_option, node) \
                                and len(node.get_effective_effect_set()) > 0 \
                                and executed_option.parent != node
            else:
                assert isinstance(node, SalientEvent)
                should_remove = not SkillChain.should_exist_edge_from_option_to_event(executed_option, node)

            if should_remove:
                print(f"Deleting edge from {executed_option} to {node}")
                self.plan_graph.plan_graph.remove_edge(executed_option, node)

    def _delete_incoming_edges_if_needed(self, executed_option):
        assert isinstance(executed_option, ModelBasedOption)

        incoming_nodes = self.plan_graph.get_incoming_nodes(executed_option)

        for node in incoming_nodes:
            assert isinstance(node, (ModelBasedOption, SalientEvent))

            if isinstance(node, ModelBasedOption):
                should_remove = not SkillChain.should_exist_edge_between_options(node, executed_option) \
                                and len(node.get_effective_effect_set()) > 0 \
                                and node.parent != executed_option
            else:
                assert isinstance(node, SalientEvent)
                should_remove = not SkillChain.should_exist_edge_from_event_to_option(node, executed_option)

            if should_remove:
                print(f"Deleting edge from {node} to {executed_option}")
                self.plan_graph.plan_graph.remove_edge(node, executed_option)

    def add_potential_edges(self, option):

        for node in self.plan_graph.plan_graph.nodes:

            if node != option and not self.plan_graph.plan_graph.has_edge(option, node):
                if isinstance(node, SalientEvent) and SkillChain.should_exist_edge_from_option_to_event(option, node):
                    print(f"Adding edge from {option} to {node}")
                    self.plan_graph.add_edge(option, node)
                if isinstance(node, ModelBasedOption) and SkillChain.should_exist_edge_between_options(option, node):
                    print(f"Adding edge from {option} to {node}")
                    self.plan_graph.add_edge(option, node)

            if node != option and not self.plan_graph.plan_graph.has_edge(node, option):
                if isinstance(node, SalientEvent) and SkillChain.should_exist_edge_from_event_to_option(node, option):
                    print(f"Adding edge from {node} to {option}")
                    self.plan_graph.add_edge(node, option)
                if isinstance(node, ModelBasedOption) and SkillChain.should_exist_edge_between_options(node, option):
                    print(f"Adding edge from {node} to {option}")
                    self.plan_graph.add_edge(node, option)

    def modify_node_connections(self, executed_option):
        self._delete_outgoing_edges_if_needed(executed_option)
        self._delete_incoming_edges_if_needed(executed_option)

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

        # 1. Get intersecting events
        # 2. Add an edge from each intersecting event to the newly_created_option
        events = [event for event in self.mdp.get_all_target_events_ever() if event != chain.target_salient_event]
        intersecting_events = [event for event in events
                               if chain.should_exist_edge_from_event_to_option(event, newly_created_option)]
        for event in intersecting_events:  # type: SalientEvent
            assert isinstance(event, SalientEvent)
            print(f"Adding edge from {event} to {newly_created_option}")
            self.plan_graph.add_edge(event, newly_created_option, edge_weight=0.)


        # 1. Get intersecting options
        # 2. Add an each from each intersecting option to the newly_created_option
        other_options = [o for o in self.get_options_in_known_part_of_the_graph() if o != newly_created_option]
        for other_option in other_options:  # type: ModelBasedOption
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
            #visualize_graph(self, episode, self.chainer.experiment_name, self.chainer.seed, True)
            visualize_graph_nodes_with_vf_expansion_probabilities(self, episode, self.chainer.experiment_name, self.chainer.seed)

    # -----------------------------–––––––--------------
    # Utility Functions
    # -----------------------------–––––––--------------

    def get_corresponding_events(self, state):
        events = [self.mdp.get_start_state_salient_event()] + self.mdp.all_salient_events_ever
        current = [event for event in events if event(state)]
        return current

    # def get_closest_pair_of_vertices(self, src_vertices, dest_vertices):
    #     closest_pair = None
    #     min_distance = np.inf
    #
    #     for source in src_vertices:  # type: SalientEvent or ModelBasedOption
    #         for target in dest_vertices:  # type: SalientEvent or ModelBasedOption
    #             if source != target:  # TODO: Why are we getting repeats?
    #                 distance = self.distance_between_vertices(source, target)
    #                 if distance < min_distance:
    #                     min_distance = distance
    #                     closest_pair = source, target
    #
    #     return closest_pair

    def get_closest_pair_of_vertices(self, src_vertices, dest_vertices):
        def sample(A, num_rows):
            return A[np.random.randint(A.shape[0], size=num_rows), :].squeeze()

        if len(src_vertices) > 0 and len(dest_vertices) > 0:
            distance_matrix = self.single_sample_vf_based_distances(src_vertices, dest_vertices)
            min_array = np.argwhere(distance_matrix == np.min(distance_matrix)).squeeze()

            if len(min_array.shape) > 1 and min_array.shape[0] > 1:
                min_array = sample(min_array, 1)

            return src_vertices[min_array[0]], dest_vertices[min_array[1]]

    def single_sample_vf_based_distances(self, src_vertices, dest_vertices):  # TODO: Test
        def sample(vertex):
            effect = vertex.trigger_points if isinstance(vertex, SalientEvent) else vertex.effect_set
            states = [s.features() if not isinstance(s, np.ndarray) else s for s in effect]
            states = [s for s in states if s.shape == (self.mdp.state_space_size(),)]
            return random.choice(states)

        def batched_pairwise_distances(points1, points2, vf):
            distance_matrix = np.zeros((len(points1), len(points2)))

            for i, point in enumerate(points1):
                x = np.repeat(point[None, ...], len(points2), axis=0)
                with torch.no_grad():
                    distance_matrix[i, :] = -vf(x, points2).squeeze()

            return distance_matrix

        def fully_batched_pairwise_distances(points1, points2, vf):
            N = len(points1)
            M = len(points2)
            D = points1.shape[1]

            # Expand points1 to have shape (N, M, D)
            expanded_points1 = points1[:, np.newaxis, :]
            expanded_points1 = np.repeat(expanded_points1, M, axis=1)
            expanded_points1 = expanded_points1.reshape((-1, D))
            assert expanded_points1.shape == (N * M, D), expanded_points1.shape

            # Expand points2 to have shape (N, M, D)
            expanded_points2 = points2[np.newaxis, :, :]
            expanded_points2 = np.repeat(expanded_points2, N, axis=0)
            expanded_points2 = expanded_points2.reshape((-1, D))
            assert expanded_points2.shape == (N * M, D), expanded_points2.shape

            # Chunked inference
            current_idx = 0
            chunk_size = 2000
            distances = np.zeros((N * M,))
            num_chunks = int(np.ceil(expanded_points1.shape[0] / chunk_size))
            p1_chunks = np.array_split(expanded_points1, num_chunks, axis=0)
            p2_chunks = np.array_split(expanded_points2, num_chunks, axis=0)

            for p1, p2 in zip(p1_chunks, p2_chunks):
                assert len(p1) == len(p2), f"{len(p1), len(p2)}"
                current_chunk_size = len(p1)
                distances[current_idx:current_idx+current_chunk_size] = -vf(p1, p2).squeeze()
                current_idx += current_chunk_size

            assert distances.shape == (N * M,), distances.shape
            distance_matrix = distances.reshape((N, M))
            return distance_matrix

        src_points = np.array([sample(v) for v in src_vertices])
        dest_points = np.array([sample(v) for v in dest_vertices])
        per_point_distance_matrix = fully_batched_pairwise_distances(src_points, dest_points, self.chainer.global_option.value_function)
        return per_point_distance_matrix

    def distance_between_vertices(self, v1, v2):
        if self.use_vf:
            return self.vf_based_distance_between_vertices(v1, v2)
        return self.norm_based_distance_between_vertices(v1, v2)

    def norm_based_distance_between_vertices(self, v1, v2):
        if isinstance(v1, SalientEvent) and isinstance(v2, SalientEvent):
            return v1.distance_to_other_event(v2)
        if isinstance(v1, SalientEvent) and isinstance(v2, ModelBasedOption):
            return v1.distance_to_effect_set(v2.effect_set)
        if isinstance(v1, ModelBasedOption) and isinstance(v2, SalientEvent):
            return v2.distance_to_effect_set(v1.effect_set)  # Assuming a symmetric distance function here
        if isinstance(v1, ModelBasedOption) and isinstance(v2, ModelBasedOption):
            return SalientEvent.set_to_set_distance(v1.effect_set, v2.effect_set)
        raise NotImplementedError(f"{type(v1), type(v2)}")

    def vf_based_distance_between_vertices(self, v1, v2):
        def _extract(state):
            return state if isinstance(state, np.ndarray) else state.features()

        def _filter(x):
            y = map(_extract, x)
            return [s for s in y if isinstance(s, np.ndarray) and s.shape[0] == self.mdp.state_space_size()]

        def _add_init_state(x):
            x.append(self.mdp.init_state.features())
            return x

        def batched_value(states, goals, vf):
            assert states.shape == goals.shape, f"{states.shape, goals.shape}"
            values = vf(states, goals)
            values[values > 0] = 0
            return values.mean()

        def value(state, goals, vf):
            states = np.repeat(state[None, ...], len(goals),  axis=0)
            return batched_value(states, goals, vf) 

        def sample(A, num_rows):
            return A[np.random.randint(A.shape[0], size=num_rows), :]

        assert isinstance(v1, (ModelBasedOption, SalientEvent)), f"{type(v1)}"
        assert isinstance(v2, (ModelBasedOption, SalientEvent)), f"{type(v2)}"

        vf = self.chainer.global_option.value_function
        e1 = _filter(v1.get_effective_effect_set()) if isinstance(v1, ModelBasedOption) else _filter(v1.trigger_points)
        e2 = _filter(v2.get_effective_effect_set()) if isinstance(v2, ModelBasedOption) else _filter(v2.trigger_points)

        if isinstance(v1, SalientEvent) and v1(self.mdp.init_state):
            e1 = _add_init_state(e1)

        if isinstance(v2, SalientEvent) and v2(self.mdp.init_state):
            e2 = _add_init_state(e2)

        if len(e1) > 0 and len(e2) > 0 and len(e1) < 32:
            e2 = np.vstack(e2)
            values = [value(state, e2, vf) for state in e1]
            return -np.mean(values)
        elif len(e1) > 0 and len(e2) > 0:
            e1 = np.vstack(e1)
            e2 = np.vstack(e2)
            l0 = min(len(e1), len(e2))
            e1 = sample(e1, num_rows=l0)
            e2 = sample(e2, num_rows=l0)
            values = batched_value(e1, e2, vf)
            estimated_distance = -np.mean(values)
            return estimated_distance

        return np.inf

    @staticmethod
    def is_state_inside_vertex(state, vertex):
        assert isinstance(state, (State, np.ndarray)), f"{type(state)}"
        assert isinstance(vertex, (ModelBasedOption, SalientEvent)), f"{type(vertex)}"

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
        assert isinstance(vertex, (ModelBasedOption, SalientEvent)), f"{type(vertex)}"

        if isinstance(vertex, ModelBasedOption):
            return vertex.get_goal_for_rollout()
        if vertex.get_target_position() is not None:
            return vertex.get_target_position()
        return random.sample(vertex.trigger_points, k=1)[0]

    def find_nearest_option(self, state, options):
        if len(options) > 0:
            distances = [(option, option.distance_to_state(state)) for option in options]
            nearest_option = sorted(distances, key=lambda x: x[1])[0][0]  # type: ModelBasedOption
            return nearest_option

    def pick_subgoal_for_global_option(self, state, goal_vertex):
        assert isinstance(goal_vertex, (ModelBasedOption, SalientEvent))

        options = self.plan_graph.get_nodes_that_reach_target_node(goal_vertex)
        options = [option for option in options if not option.pessimistic_is_init_true(state)]
        nearest_option = self.find_nearest_option(state, options)

        if nearest_option is not None:
            sampled_goal = None
            num_tries = 0
            while sampled_goal is None and num_tries < 10:
                sampled_goal = nearest_option.sample_from_initiation_region_fast_and_epsilon()

                if isinstance(sampled_goal, np.ndarray):
                    return sampled_goal.squeeze()

                sampled_goal = self.mdp.get_position(sampled_goal)
                if self.mdp.sparse_gc_reward_function(state, sampled_goal, info={})[1]:
                    sampled_goal = None
                num_tries += 1

            print(f"[Skill-Graph-Planner] Picked {sampled_goal} as a subgoal for the global-option")
            return sampled_goal
