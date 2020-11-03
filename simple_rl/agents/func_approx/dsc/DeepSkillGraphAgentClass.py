import ipdb
import argparse
import random
import networkx as nx
from copy import deepcopy
from collections import defaultdict
import networkx.algorithms.shortest_paths as shortest_paths
from simple_rl.agents.func_approx.dsc.SalientEventClass import SalientEvent, DSCOptionSalientEvent
from simple_rl.agents.func_approx.dsc.SkillChainingAgentClass import SkillChaining
from simple_rl.agents.func_approx.dsc.SkillGraphPlannerClass import SkillGraphPlanner
from simple_rl.agents.func_approx.dsc.OptionClass import Option
from simple_rl.agents.func_approx.dsc.utils import *
from simple_rl.mdp import MDP, State
from simple_rl.mdp.GoalDirectedMDPClass import GoalDirectedMDP


class DeepSkillGraphAgent(object):
    def __init__(self, mdp, dsc_agent, planning_agent, salient_event_freq, event_after_reject_freq,
                 experiment_name, seed, threshold, use_smdp_replay_buffer,
                 rejection_criteria, pick_targets_stochastically, plot_gc_value_functions):
        """
        This agent will interleave planning with the `planning_agent` and chaining with
        the `dsc_agent`.
        Args:
            mdp (GoalDirectedMDP)
            dsc_agent (SkillChaining)
            planning_agent (SkillGraphPlanner)
            salient_event_freq (int)
            experiment_name (str)
            seed (int)
            rejection_criteria (str)
            pick_targets_stochastically (bool)
        """
        self.mdp = mdp
        self.dsc_agent = dsc_agent
        self.planning_agent = planning_agent
        self.salient_event_freq = salient_event_freq
        self.experiment_name = experiment_name
        self.seed = seed
        self.threshold = threshold
        self.salient_event_freq = salient_event_freq
        self.event_after_reject_freq = event_after_reject_freq
        self.use_smdp_replay_buffer = use_smdp_replay_buffer
        self.rejection_criteria = rejection_criteria
        self.pick_targets_stochastically = pick_targets_stochastically
        self.plot_gc_value_functions = plot_gc_value_functions

        assert rejection_criteria in ("use_effect_sets", "use_init_sets"), rejection_criteria

        self.generated_salient_events = []
        self.last_event_creation_episode = -1
        self.pursued_source_target_map = defaultdict(lambda: defaultdict(int))

    def __getstate__(self):
        excluded_keys = ("mdp")
        state_dictionary = {x: self.__dict__[x] for x in self.__dict__ if x not in excluded_keys}
        return state_dictionary

    def __setstate__(self, state_dictionary):
        for key in state_dictionary:
            self.__dict__[key] = state_dictionary[key]

    @staticmethod
    def _randomly_select_salient_event(state, candidate_salient_events):
        num_tries = 0
        target_event = None
        while target_event is None and num_tries < 100 and len(candidate_salient_events) > 0:
            target_event = random.sample(candidate_salient_events, k=1)[0]

            # If you are already at the target_event, then re-sample
            if target_event(state):
                target_event = None

            num_tries += 1

        if target_event is not None:
            print(f"[Random] Deep skill graphs target event: {target_event}")

        return target_event

    def get_weakest_link_in_graph(self, state):
        """ From the set of options that we can get to, pick the weakest. """
        cond = lambda s, o: not o.is_close_to_effect_set(s) and self.planning_agent.plan_graph.does_path_exist(s, o)
        options = [o for o in self.planning_agent.plan_graph.option_nodes if cond(state, o)]
        if len(options) > 0:
            success_rates = [np.mean(o.on_policy_success_curve) for o in options]
            rates_and_options = [(rate, option) for rate, option in zip(success_rates, options)]
            sorted_rates_and_options = sorted(rates_and_options, key=lambda x: x[0])
            return sorted_rates_and_options[0][1]
        return None

    def _select_weakest_link_salient_event(self, state):
        weakest_option = self.get_weakest_link_in_graph(state)

        if weakest_option is not None:
            return self.get_corresponding_salient_event(weakest_option)

    def get_corresponding_salient_event(self, option_vertex):
        assert isinstance(option_vertex, Option), option_vertex

        def add_option_salient_event_to_graph(original_option, corresponding_event):
            neighbors = self.planning_agent.plan_graph.get_outgoing_nodes(original_option)
            weights = [self.planning_agent.plan_graph.plan_graph[original_option][n]["weight"] for n in neighbors]
            new_edge_weight = max(weights)

            self.add_salient_event(corresponding_event)
            self.planning_agent.plan_graph.add_node(corresponding_event)
            self.planning_agent.plan_graph.add_edge(original_option, corresponding_event, new_edge_weight)

        def get_corresponding_salient_event(option):
            for event in self.planning_agent.plan_graph.salient_nodes:
                if isinstance(event, DSCOptionSalientEvent):
                    if event.option == option:
                        return event
            return None

        salient_event = get_corresponding_salient_event(option_vertex)

        if salient_event is None:
            event_idx = len(self.planning_agent.plan_graph.salient_nodes) + 1
            salient_event = DSCOptionSalientEvent(option_vertex, event_idx)
            add_option_salient_event_to_graph(option_vertex, salient_event)

        return salient_event

    def _select_closest_unconnected_salient_event(self, state, events):
        graph = self.planning_agent.plan_graph
        candidate_salient_events = self.generate_candidate_salient_events(state)

        current_events = [event for event in events if event(state)]
        descendant_events = self.planning_agent.plan_graph.get_reachable_nodes_from_source_state(state)
        descendant_events += current_events

        if not all([isinstance(e, (SalientEvent, Option)) for e in descendant_events]):
            ipdb.set_trace()

        # Grab all the ancestor Salient Events of each candidate salient event
        ancestor_events = []
        for salient_event in candidate_salient_events:
            ancestors = graph.get_nodes_that_reach_target_node(salient_event)
            if any([ancestor in descendant_events for ancestor in ancestors]):
                ipdb.set_trace()
            filtered_ancestors = [e for e in ancestors if isinstance(e, SalientEvent)]
            if len(filtered_ancestors) > 0:
                ancestor_events += filtered_ancestors
        ancestor_events += candidate_salient_events

        # -- Compress to delete repeating events
        # -- Note that we are using the __hash__ function of SalientEvents here
        ancestor_events = list(set(ancestor_events))

        if not all([isinstance(e, SalientEvent) for e in ancestor_events]):
            ipdb.set_trace()

        if self.pick_targets_stochastically:
            event_pairs = [(e1, e2) for e1 in descendant_events for e2 in ancestor_events]
            closest_event_pair = self._choose_node_pair(node_pairs=event_pairs) if len(event_pairs) > 0 else None
        else:
            closest_event_pair = self.planning_agent.get_closest_pair_of_vertices(descendant_events, ancestor_events)

        # Finally, return the ancestor event closest to one of the descendants
        if closest_event_pair is not None:
            assert len(closest_event_pair) == 2, closest_event_pair
            closest_event = closest_event_pair[1]
            assert isinstance(closest_event, SalientEvent), f"{type(closest_event)}"
            if not closest_event(state):
                return closest_event

    @staticmethod
    def _choose_node_pair(node_pairs):
        """ Given a list of descendant-ancestor pairs, compute their distances and pick one to target. """
        node_pair_distances = [1. / SkillGraphPlanner.distance_between_vertices(d, a) for d, a in node_pairs]
        sum_of_distances = sum(node_pair_distances)
        probabilities = [distance / sum_of_distances for distance in node_pair_distances]
        np.testing.assert_almost_equal(np.sum(probabilities), 1.)
        idx = np.random.choice(range(len(node_pairs)), size=1, p=probabilities)
        if len(idx) > 0:
            return node_pairs[idx[0]]

    def select_goal_salient_event(self, state):
        """

        Args:
            state (State)

        Returns:
            target_event (SalientEvent)
        """
        stochastic = self.pick_targets_stochastically
        epsilon = 0.2 if stochastic else 0.
        selection_criteria = "closest" if random.random() > epsilon else "random"
        events = self.mdp.get_all_target_events_ever() + [self.mdp.get_start_state_salient_event()]
        if len(self.mdp.get_all_target_events_ever()) > 0:
            if selection_criteria == "closest":
                selected_event = self._select_closest_unconnected_salient_event(state, events)
                if selected_event is not None:
                    print(f"[Closest] Deep skill graphs target event: {selected_event}")
                    return selected_event
                else:
                    selected_event = self._select_weakest_link_salient_event(state)
                    if selected_event is not None:
                        print(f"[Weakest] Deep skill graphs target event: {selected_event}")
                        return selected_event
            return self._randomly_select_salient_event(state, events)

    def dsg_run_loop(self, episodes, num_steps):
        successes = []

        for episode in range(episodes):

            if self.should_generate_new_salient_events(episode):
                self.generate_new_salient_events(episode)

            step_number = 0
            random_episodic_trajectory = []
            self.reset(episode, start_state=None)

            state = deepcopy(self.mdp.cur_state)

            while step_number < num_steps:
                goal_salient_event = self.select_goal_salient_event(state)

                if goal_salient_event is None:
                    random_transition = self.take_random_action()
                    step_number += 1
                    success = False
                    random_episodic_trajectory.append(random_transition)
                else:

                    if not isinstance(goal_salient_event, DSCOptionSalientEvent):
                        self.create_skill_chains_if_needed(state, goal_salient_event, eval_mode=False)

                    self.pursued_source_target_map[self._get_current_salient_event(state)][goal_salient_event] += 1

                    step_number, success = self.planning_agent.run_loop(state=state,
                                                                        goal_salient_event=goal_salient_event,
                                                                        episode=episode,
                                                                        step=step_number,
                                                                        eval_mode=False)

                state = deepcopy(self.mdp.cur_state)

                if success:
                    print(f"[DeepSkillGraphAgentClass] successfully reached {goal_salient_event}")

                successes.append(success)

            if episode < 5:
                goal_state = self.mdp.get_position(self.mdp.sample_random_state())
                self.dsc_agent.global_option_experience_replay(random_episodic_trajectory, goal_state=goal_state)

            if episode > 0 and episode % 50 == 0 and self.plot_gc_value_functions:
                assert goal_salient_event is not None
                make_chunked_goal_conditioned_value_function_plot(self.dsc_agent.global_option.solver,
                                                                  goal_salient_event.get_target_position(),
                                                                  episode, self.seed, self.experiment_name)
                make_chunked_goal_conditioned_value_function_plot(self.dsc_agent.agent_over_options,
                                                                  goal_salient_event.get_target_position(),
                                                                  episode, self.seed, self.experiment_name)
                visualize_best_option_to_take(self.dsc_agent.agent_over_options,
                                              episode, self.seed, self.experiment_name)

        return successes

    def dsg_test_loop(self, episodes, test_event, start_state=None):
        assert isinstance(episodes, int), f"{type(episodes)}"
        assert isinstance(test_event, SalientEvent), test_event

        successes = []
        final_states = []

        self.reset(0, start_state)
        state = deepcopy(self.mdp.cur_state)
        start_state_event = self._get_current_salient_event(state)
        num_descendants = len(self.planning_agent.plan_graph.get_reachable_nodes_from_source_state(state))

        if start_state_event is None:
            mdp_events = self.mdp.get_all_target_events_ever() + [self.mdp.get_start_state_salient_event()]
            graph_events = self.planning_agent.plan_graph.salient_nodes
            event_idx = [event.event_idx for event in mdp_events + graph_events]
            start_state_event = SalientEvent(target_state=state.position, event_idx=max(event_idx)+1)

        if num_descendants == 0:
            self.create_skill_chains_from_outside_graph(state, start_state_event, test_event)
        else:
            self.create_skill_chains_if_needed(state, test_event, eval_mode=True, current_event=start_state_event)

        for episode in range(episodes):
            self.reset(episode, start_state)
            state = deepcopy(self.mdp.cur_state)

            step_number, state = self.planning_agent.test_loop(state=state,
                                                               start_state_event=start_state_event,
                                                               goal_salient_event=test_event,
                                                               episode=episode,
                                                               step=0)
            success = test_event(state)
            successes.append(success)
            final_states.append(deepcopy(state))

        return successes, final_states

    def generate_new_salient_events(self, episode):

        event_idx = len(self.mdp.all_salient_events_ever) + 1
        target_state = self.mdp.sample_random_state()[:2]
        low_salient_event = SalientEvent(target_state, event_idx)
        reject_low = self.add_salient_event(low_salient_event)

        print(f"Generated {low_salient_event}; Rejected: {reject_low}")
        self.last_event_creation_episode = episode

    def add_salient_event(self, salient_event):
        reject = self.should_reject_discovered_salient_event(salient_event)

        if not reject:
            print(f"[DSG Agent] Accepted {salient_event}")

            self.generated_salient_events.append(salient_event)
            self.mdp.add_new_target_event(salient_event)

        return reject

    def is_path_under_construction(self, state, init_salient_event, target_salient_event):
        assert isinstance(state, (State, np.ndarray)), f"{type(State)}"
        assert isinstance(init_salient_event, SalientEvent), f"{type(init_salient_event)}"
        assert isinstance(target_salient_event, SalientEvent), f"{type(target_salient_event)}"

        match = lambda c: c.init_salient_event == init_salient_event and c.target_salient_event == target_salient_event
        if any([match(c) for c in self.dsc_agent.chains]):
            return True

        events = self.mdp.get_all_target_events_ever() + [self.mdp.get_start_state_salient_event()]
        current_salient_events = [event for event in events if event(state)]
        under_construction = any([self.does_path_exist_in_optimistic_graph(current_salient_event, target_salient_event)
                                  for current_salient_event in current_salient_events])
        return under_construction

    def does_path_exist_in_optimistic_graph(self, vertex1, vertex2):

        # Create a lightweight copy of the plan-graph
        graph = nx.DiGraph()
        for edge in self.planning_agent.plan_graph.plan_graph.edges:
            graph.add_edge(str(edge[0]), str(edge[1]))

        # Pretend as if all unfinished chains have been learned and add them to the new graph
        unfinished_chains = [chain for chain in self.dsc_agent.chains if not chain.is_chain_completed()]
        for chain in unfinished_chains:
            graph.add_edge(str(chain.init_salient_event), str(chain.target_salient_event))

        # Return if there is a path in this "optimistic" graph
        if str(vertex1) not in graph or str(vertex2) not in graph:
            return False

        return shortest_paths.has_path(graph, str(vertex1), str(vertex2))

    def should_reject_discovered_salient_event(self, salient_event):
        """
        Reject the discovered salient event if it is the same as an old one or inside the initiation set of an option.

        Args:
            salient_event (SalientEvent)

        Returns:
            should_reject (bool)
        """
        existing_target_events = self.mdp.get_all_target_events_ever() + [self.mdp.get_start_state_salient_event()]
        if any([event(salient_event.target_state) for event in existing_target_events]):
            return True

        rejection_function = self.planning_agent.is_state_inside_any_vertex if self.rejection_criteria == "use_effect_sets"\
                             else self.dsc_agent.state_in_any_completed_option

        if rejection_function(salient_event.target_state):
            return True

        return False

    def should_generate_new_salient_events(self, episode):
        if episode < 5:
            return False
        elif episode == 5:
            return True

        return episode - self.last_event_creation_episode >= self.salient_event_freq

    def generate_candidate_salient_events(self, state):
        """ Return the events that we are currently NOT in and to whom there is no path on the graph. """
        connected = lambda s, e: self.planning_agent.plan_graph.does_path_exist(s, e)
        events = self.mdp.get_all_target_events_ever() + [self.mdp.get_start_state_salient_event()]
        unconnected_events = [event for event in events if not connected(state, event) and not event(state)]
        return unconnected_events

    def take_random_action(self):
        state = deepcopy(self.mdp.cur_state)
        action = self.mdp.sample_random_action()
        reward, next_state = self.mdp.execute_agent_action(action)
        done = self.mdp.is_goal_state(next_state)

        if not self.planning_agent.use_her:
            self.dsc_agent.global_option.solver.step(state.features(), action, reward, next_state.features(), done)
            self.dsc_agent.agent_over_options.step(state.features(), 0, reward, next_state.features(), done, 1)

        return state, action, reward, next_state

    def create_skill_chains_if_needed(self, state, goal_salient_event, eval_mode, current_event=None):
        current_salient_event = self._get_current_salient_event(state) if current_event is None else current_event

        if current_salient_event is not None:
            if not self.planning_agent.plan_graph.does_path_exist(state, goal_salient_event) and \
                    not self.is_path_under_construction(state, current_salient_event, goal_salient_event):

                closest_event_pair = self.planning_agent.choose_closest_source_target_vertex_pair(state,
                                                                                                  goal_salient_event,
                                                                                                  choose_among_events=True)


                init, target = current_salient_event, goal_salient_event
                if closest_event_pair is not None:
                    init, target = closest_event_pair[0], closest_event_pair[1]

                print(f"[DeepSkillGraphsAgent] Creating chain from {init} -> {target}")

                self.dsc_agent.create_chain_targeting_new_salient_event(salient_event=target,
                                                                        init_salient_event=init,
                                                                        eval_mode=eval_mode)

    def create_skill_chains_from_outside_graph(self, state, start_state_event, goal_salient_event):

        beta1, beta2 = self.planning_agent.get_test_time_subgoals(start_state_event, goal_salient_event)
        creation_condition = lambda s, b0, bg: not self.planning_agent.plan_graph.does_path_exist(s, bg) and \
                                               not self.is_path_under_construction(s, b0, bg)

        if not self.planning_agent.plan_graph.does_path_exist(state, goal_salient_event):
            if creation_condition(state, start_state_event, beta1):
                print(f"[DSG-Test-Time] Creating chain from {start_state_event} -> {beta1}")
                self.dsc_agent.create_chain_targeting_new_salient_event(salient_event=beta1,
                                                                        init_salient_event=start_state_event,
                                                                        eval_mode=True)
            if creation_condition(state, beta2, goal_salient_event):
                print(f"[DSG-Test-Time] Creating chain from {beta2} -> {goal_salient_event}")
                self.dsc_agent.create_chain_targeting_new_salient_event(salient_event=goal_salient_event,
                                                                        init_salient_event=beta2,
                                                                        eval_mode=True)

    def _get_current_salient_event(self, state):
        assert isinstance(state, (State, np.ndarray)), f"{type(state)}"
        events = self.mdp.get_all_target_events_ever() + [self.mdp.get_start_state_salient_event()]
        for event in events:  # type: SalientEvent
            if event(state):
                return event
        return None

    # -----------------------------–––––––--------------
    # Evaluation Functions
    # -----------------------------–––––––--------------
    def reset(self, episode, start_state=None):
        """ Reset the MDP to either the default start state or to the specified one. """

        print("*" * 80)
        if start_state is None:
            self.mdp.reset()
            print(f"[DeepSkillGraphAgentClass] Episode {episode}: Resetting MDP to start state")
        else:
            start_position = start_state.position if isinstance(start_state, State) else start_state[:2]
            print(f"[DeepSkillGraphAgentClass] Episode {episode}: Resetting MDP to manual state {start_position}")
            self.mdp.reset()
            self.mdp.set_xy(start_position)
        print("*" * 80)


if __name__ == "__main__":
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
    parser.add_argument("--tensor_log", type=bool, help="Enable tensorboard logging", default=False)
    parser.add_argument("--control_cost", type=bool, help="Penalize high actuation solutions", default=False)
    parser.add_argument("--dense_reward", type=bool, help="Use dense/sparse rewards", default=False)
    parser.add_argument("--max_num_options", type=int, help="Max number of options we can learn", default=5)
    parser.add_argument("--num_subgoal_hits", type=int, help="Number of subgoal hits to learn an option", default=3)
    parser.add_argument("--buffer_len", type=int, help="buffer size used by option to create init sets", default=20)
    parser.add_argument("--classifier_type", type=str, help="ocsvm/elliptic for option initiation clf", default="ocsvm")
    parser.add_argument("--init_q", type=str, help="compute/zero", default="zero")
    parser.add_argument("--use_smdp_update", type=bool, help="sparse/SMDP update for option policy", default=False)
    parser.add_argument("--use_start_state_salience", action="store_true", default=False)
    parser.add_argument("--use_option_intersection_salience", action="store_true", default=False)
    parser.add_argument("--use_event_intersection_salience", action="store_true", default=False)
    parser.add_argument("--pretrain_option_policies", action="store_true", default=False)
    parser.add_argument("--use_warmup_phase", action="store_true", default=False)
    parser.add_argument("--update_global_solver", action="store_true", default=False)
    parser.add_argument("--salient_event_freq", type=int, help="Create a salient event every salient_event_freq episodes", default=50)
    parser.add_argument("--event_after_reject_freq", type=int, help="Create a salient event only event_after_reject_freq episodes after a double rejection", default=10)
    parser.add_argument("--plot_rejected_events", action="store_true", default=False)
    parser.add_argument("--use_ucb", action="store_true", default=False)
    parser.add_argument("--threshold", type=int, help="Threshold determining size of termination set", default=0.1)
    parser.add_argument("--use_smdp_replay_buffer", action="store_true", help="Whether to use a replay buffer that has options", default=False)
    parser.add_argument("--use_her", action="store_true", default=False)
    parser.add_argument("--use_her_locally", action="store_true", help="HER for local options", default=False)
    parser.add_argument("--off_policy_update_type", type=str, default="none")
    parser.add_argument("--plot_gc_value_functions", action="store_true", default=False)
    parser.add_argument("--allow_her_initialization", action="store_true", default=False)
    parser.add_argument("--rejection_criteria", type=str, help="Reject events in known init/effect sets", default="use_init_sets")
    parser.add_argument("--pick_targets_stochastically", action="store_true", default=False)
    args = parser.parse_args()

    if args.env == "point-reacher":
        from simple_rl.tasks.point_reacher.PointReacherMDPClass import PointReacherMDP

        overall_mdp = PointReacherMDP(seed=args.seed,
                                      dense_reward=args.dense_reward,
                                      render=args.render,
                                      use_hard_coded_events=args.use_hard_coded_events)
        state_dim = 6
        action_dim = 2
    elif args.env == "ant-reacher":
        from simple_rl.tasks.ant_reacher.AntReacherMDPClass import AntReacherMDP
        overall_mdp = AntReacherMDP(seed=args.seed,
                                    render=args.render)
        state_dim = overall_mdp.state_space_size()
        action_dim = overall_mdp.action_space_size()
    elif args.env == "d4rl-ant-maze":
        from simple_rl.tasks.d4rl_ant_maze.D4RLAntMazeMDPClass import D4RLAntMazeMDP
        overall_mdp = D4RLAntMazeMDP(maze_size="umaze", seed=args.seed, render=args.render)
        state_dim = overall_mdp.state_space_size()
        action_dim = overall_mdp.action_space_size()
    elif args.env == "d4rl-medium-ant-maze":
        from simple_rl.tasks.d4rl_ant_maze.D4RLAntMazeMDPClass import D4RLAntMazeMDP
        overall_mdp = D4RLAntMazeMDP(maze_size="medium", seed=args.seed, render=args.render)
        state_dim = overall_mdp.state_space_size()
        action_dim = overall_mdp.action_space_size()
    elif args.env == "d4rl-medium-point-maze":
        from simple_rl.tasks.d4rl_point_maze.D4RLPointMazeMDPClass import D4RLPointMazeMDP
        overall_mdp = D4RLPointMazeMDP(seed=args.seed,
                                       render=args.render,
                                       difficulty="medium",
                                       goal_directed=False)
        state_dim = overall_mdp.state_space_size()
        action_dim = overall_mdp.action_space_size()
    elif args.env == "d4rl-hard-point-maze":
        from simple_rl.tasks.d4rl_point_maze.D4RLPointMazeMDPClass import D4RLPointMazeMDP
        overall_mdp = D4RLPointMazeMDP(seed=args.seed,
                                       render=args.render,
                                       difficulty="hard",
                                       goal_directed=False)
        state_dim = overall_mdp.state_space_size()
        action_dim = overall_mdp.action_space_size()
    else:
        raise NotImplementedError(args.env)

    # Create folders for saving various things
    logdir = create_log_dir(args.experiment_name)
    create_log_dir("saved_runs")
    create_log_dir("value_function_plots")
    create_log_dir("initiation_set_plots")
    create_log_dir("value_function_plots/{}".format(args.experiment_name))
    create_log_dir("initiation_set_plots/{}".format(args.experiment_name))

    print("Training skill chaining agent from scratch with a subgoal reward {}".format(args.subgoal_reward))
    print("MDP InitState = ", overall_mdp.init_state)

    q0 = 0. if args.init_q == "zero" else None

    chainer = SkillChaining(overall_mdp, args.steps, args.lr_a, args.lr_c, args.ddpg_batch_size,
                            seed=args.seed, subgoal_reward=args.subgoal_reward,
                            log_dir=logdir, num_subgoal_hits_required=args.num_subgoal_hits,
                            enable_option_timeout=args.option_timeout, init_q=q0,
                            use_full_smdp_update=args.use_smdp_update,
                            generate_plots=args.generate_plots, tensor_log=args.tensor_log,
                            device=args.device, buffer_length=args.buffer_len,
                            start_state_salience=args.use_start_state_salience,
                            option_intersection_salience=args.use_option_intersection_salience,
                            event_intersection_salience=args.use_event_intersection_salience,
                            pretrain_option_policies=args.pretrain_option_policies,
                            dense_reward=args.dense_reward,
                            update_global_solver=args.update_global_solver,
                            use_warmup_phase=args.use_warmup_phase,
                            experiment_name=args.experiment_name,
                            use_her=args.use_her,
                            use_her_locally=args.use_her_locally,
                            off_policy_update_type=args.off_policy_update_type,
                            allow_her_initialization=args.allow_her_initialization)

    assert any([args.use_start_state_salience, args.use_option_intersection_salience, args.use_event_intersection_salience])

    planner = SkillGraphPlanner(mdp=overall_mdp,
                                chainer=chainer,
                                experiment_name=args.experiment_name,
                                seed=args.seed,
                                use_her=args.use_her,
                                pretrain_option_policies=args.pretrain_option_policies)

    dsg_agent = DeepSkillGraphAgent(mdp=overall_mdp,
                                    dsc_agent=chainer,
                                    planning_agent=planner,
                                    salient_event_freq=args.salient_event_freq,
                                    event_after_reject_freq=args.event_after_reject_freq,
                                    experiment_name=args.experiment_name,
                                    seed=args.seed,
                                    threshold=args.threshold,
                                    use_smdp_replay_buffer=args.use_smdp_replay_buffer,
                                    rejection_criteria=args.rejection_criteria,
                                    pick_targets_stochastically=args.pick_targets_stochastically,
                                    plot_gc_value_functions=args.plot_gc_value_functions)
    num_successes = dsg_agent.dsg_run_loop(episodes=args.episodes, num_steps=args.steps)
