import ipdb
import argparse
import random
import networkx as nx
from copy import deepcopy
from collections import defaultdict
import networkx.algorithms.shortest_paths as shortest_paths
from simple_rl.agents.func_approx.dsc.SalientEventClass import SalientEvent
from simple_rl.agents.func_approx.dsc.ModelBasedDSC import ModelBasedSkillChaining
from simple_rl.agents.func_approx.dsc.MBSkillGraphPlanner import SkillGraphPlanner
from simple_rl.agents.func_approx.dsc.MBOptionClass import ModelBasedOption
from simple_rl.agents.func_approx.dsc.utils import *
from simple_rl.mdp import MDP, State
from simple_rl.mdp.GoalDirectedMDPClass import GoalDirectedMDP


class DeepSkillGraphAgent(object):
    def __init__(self, mdp, dsc_agent, planning_agent, salient_event_freq,
                 experiment_name, seed, plot_gc_value_functions):
        """
        This agent will interleave planning with the `planning_agent` and chaining with
        the `dsc_agent`.
        Args:
            mdp (GoalDirectedMDP)
            dsc_agent (ModelBasedSkillChaining)
            planning_agent (SkillGraphPlanner)
            salient_event_freq (int)
            experiment_name (str)
            seed (int)
            plot_gc_value_functions (bool)
        """
        self.mdp = mdp
        self.dsc_agent = dsc_agent
        self.planning_agent = planning_agent
        self.salient_event_freq = salient_event_freq
        self.experiment_name = experiment_name
        self.seed = seed
        self.salient_event_freq = salient_event_freq
        self.plot_gc_value_functions = plot_gc_value_functions

        assert isinstance(self.dsc_agent, ModelBasedSkillChaining)

        self.num_warmup_episodes = 50

        self.generated_salient_events = []

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

    def _select_closest_unconnected_salient_event(self, state, events):
        graph = self.planning_agent.plan_graph
        candidate_salient_events = self.generate_candidate_salient_events(state)

        current_events = [event for event in events if event(state)]
        descendant_events = self.planning_agent.plan_graph.get_reachable_nodes_from_source_state(state)
        descendant_events += current_events

        if not all([isinstance(e, (SalientEvent, ModelBasedOption)) for e in descendant_events]):
            ipdb.set_trace()

        # Grab all the ancestor Salient Events of each candidate salient event
        ancestor_events = []
        for salient_event in candidate_salient_events:
            ancestors = graph.get_nodes_that_reach_target_node(salient_event)
            filtered_ancestors = [e for e in ancestors if isinstance(e, SalientEvent)]
            if len(filtered_ancestors) > 0:
                ancestor_events += filtered_ancestors
        ancestor_events += candidate_salient_events

        # -- Compress to delete repeating events
        # -- Note that we are using the __hash__ function of SalientEvents here
        ancestor_events = list(set(ancestor_events))

        if not all([isinstance(e, SalientEvent) for e in ancestor_events]):
            ipdb.set_trace()

        closest_event_pair = self.planning_agent.get_closest_pair_of_vertices(descendant_events, ancestor_events)

        # Finally, return the ancestor event closest to one of the descendants
        if closest_event_pair is not None:
            assert len(closest_event_pair) == 2, closest_event_pair
            closest_event = closest_event_pair[1]
            assert isinstance(closest_event, SalientEvent), f"{type(closest_event)}"
            if not closest_event(state):
                return closest_event

    def _select_closest_unfinished_chain_init_event(self, state):
        unfinished_chains = [chain for chain in self.dsc_agent.chains if not chain.is_chain_completed()]
        unfinished_chains = [chain for chain in unfinished_chains if not chain.init_salient_event(state)]
        init_salient_events = [chain.init_salient_event for chain in unfinished_chains]
        if len(init_salient_events) > 0:
            return random.choice(init_salient_events)

    def select_goal_salient_event(self, state):
        """

        Args:
            state (State)

        Returns:
            target_event (SalientEvent)
        """
        selection_criteria = "closest"
        events = self.mdp.get_all_target_events_ever() + [self.mdp.get_start_state_salient_event()]
        if len(self.mdp.get_all_target_events_ever()) > 0:
            if selection_criteria == "closest":
                selected_event = self._select_closest_unconnected_salient_event(state, events)
                if selected_event is not None:
                    print(f"[Closest] Deep skill graphs target event: {selected_event}")
                    return selected_event
                else:
                    # selected_event = self._select_closest_unfinished_chain_init_event(state)
                    # if selected_event is not None:
                    #     print(f"[ChainInit] Deep skill graphs target event: {selected_event}")
                    #     return selected_event
                    pass
            return self._randomly_select_salient_event(state, events)

    def dsg_run_loop(self, episodes, num_steps, starting_episode=0):
        successes = []

        for episode in range(starting_episode, starting_episode+episodes):

            if self.should_generate_new_salient_events(episode):
                accepted, num_tries = self.dsg_event_discovery_loop(episode, num_tries_allowed=10, num_steps=num_steps)
                print(f"[Salient-Event-Discovery] Event discovery accepted={accepted} after {num_tries} tries")
                continue

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

                    self.create_skill_chains_if_needed(state, goal_salient_event, eval_mode=False)

                    step_number, success = self.planning_agent.run_loop(state=state,
                                                                        goal_salient_event=goal_salient_event,
                                                                        episode=episode,
                                                                        step=step_number,
                                                                        eval_mode=False)

                state = deepcopy(self.mdp.cur_state)

                if success:
                    print(f"[DeepSkillGraphAgentClass] successfully reached {goal_salient_event}")

                successes.append(success)

            if episode > 0 and episode % 50 == 0 and self.plot_gc_value_functions:
                assert goal_salient_event is not None
                make_chunked_goal_conditioned_value_function_plot(self.dsc_agent.global_option.solver,
                                                                  goal_salient_event.get_target_position(),
                                                                  episode, self.seed, self.experiment_name)

            if episode > 0 and episode % 500 == 0:
                option_num_executions = [o.num_executions for o in self.planning_agent.plan_graph.option_nodes]
                option_success_rates = [o.get_success_rate() for o in planner.plan_graph.option_nodes]
                plt.scatter(option_num_executions, option_success_rates)
                plt.title(f"Episode: {episode}")
                plt.savefig(f"{self.experiment_name}/option-success-rates-episode-{episode}.png")
                plt.close()

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

    def dsg_event_discovery_loop(self, current_episode, num_tries_allowed, num_steps):
        assert isinstance(current_episode, int)
        assert isinstance(num_tries_allowed, int)
        assert isinstance(num_steps, int)

        num_tries = 0
        accepted = False

        epochs = 50 if current_episode <= self.num_warmup_episodes else 5
        self.learn_dynamics_model(epochs=epochs)

        while not accepted and num_tries < num_tries_allowed:
            num_tries += 1
            self.reset(current_episode)
            goal_salient_event, reject = self.generate_new_salient_event()

            if goal_salient_event is not None and not reject:
                state, step, accepted = self.planning_agent.salient_event_discovery_run_loop(goal_salient_event,
                                                                                             current_episode)
                print(f"[Salient-Event-Discovery] Try # {num_tries}, final-state: {state.position}, step #{step}, accepted={accepted}")

        if accepted:
            self.add_salient_event(goal_salient_event)

        return accepted, num_tries

    def learn_dynamics_model(self, epochs=50, batch_size=1024):
        self.dsc_agent.global_option.solver.load_data()
        self.dsc_agent.global_option.solver.train(epochs=epochs, batch_size=batch_size)
        for option in self.dsc_agent.get_all_options():
            option.solver.model = self.dsc_agent.global_option.solver.model

    def generate_new_salient_event(self):
        num_tries = 0
        reject = True
        salient_event = None

        while reject and num_tries < 50:
            num_tries += 1
            event_idx = len(self.mdp.all_salient_events_ever) + 1
            target_state = self.mdp.sample_random_state()[:2]
            salient_event = SalientEvent(target_state, event_idx)

            reject = self.should_reject_discovered_salient_event(salient_event)

            print(f"Generated {salient_event}; Rejected: {reject}")

        return salient_event, reject

    def add_salient_event(self, salient_event):
        print(f"[DSG Agent] Accepted {salient_event} (revised = {salient_event.revised_by_mpc})")
        self.generated_salient_events.append(salient_event)
        self.mdp.add_new_target_event(salient_event)

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
        return self.planning_agent.should_reject_mpc_revision(state=salient_event.target_state,
                                                              goal_salient_event=salient_event)

    def should_generate_new_salient_events(self, episode):
        if episode < self.num_warmup_episodes:
            return False
        elif episode == self.num_warmup_episodes:
            return True

        return episode > 0 and episode % self.salient_event_freq == 0

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
        self.dsc_agent.global_option.update_model(state, action, reward, next_state)
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

                if not self.is_path_under_construction(state, init, target):
                    print(f"[DeepSkillGraphsAgent] Creating chain from {init} -> {target}")
                    self.dsc_agent.create_new_chain(init_event=init, target_event=target)

    def create_skill_chains_from_outside_graph(self, state, start_state_event, goal_salient_event):

        beta1, beta2 = self.planning_agent.get_test_time_subgoals(start_state_event, goal_salient_event)
        creation_condition = lambda s, b0, bg: not self.planning_agent.plan_graph.does_path_exist(s, bg) and \
                                               not self.is_path_under_construction(s, b0, bg)

        if not self.planning_agent.plan_graph.does_path_exist(state, goal_salient_event):
            if creation_condition(state, start_state_event, beta1):
                print(f"[DSG-Test-Time] Creating chain from {start_state_event} -> {beta1}")
                self.dsc_agent.create_new_chain(target_event=beta1, init_event=start_state_event)
            if creation_condition(state, beta2, goal_salient_event):
                print(f"[DSG-Test-Time] Creating chain from {beta2} -> {goal_salient_event}")
                self.dsc_agent.create_new_chain(target_event=goal_salient_event, init_event=beta2)

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
    parser.add_argument("--seed", type=int, help="Random seed for this run (default=0)", default=0)
    parser.add_argument("--episodes", type=int, help="# episodes", default=200)
    parser.add_argument("--steps", type=int, help="# steps", default=1000)
    parser.add_argument("--subgoal_reward", type=float, help="SkillChaining subgoal reward", default=0.)
    parser.add_argument("--render", type=bool, help="Render the mdp env", default=False)
    parser.add_argument("--use_dense_rewards", action="store_true", help="Use dense/sparse rewards", default=False)
    parser.add_argument("--gestation_period", type=int, help="Number of subgoal hits to learn an option", default=5)
    parser.add_argument("--buffer_len", type=int, help="buffer size used by option to create init sets", default=50)
    parser.add_argument("--salient_event_freq", type=int, help="Create a salient event every salient_event_freq episodes", default=50)
    parser.add_argument("--plot_rejected_events", action="store_true", default=False)
    parser.add_argument("--plot_gc_value_functions", action="store_true", default=False)
    parser.add_argument("--use_model", action="store_true", default=False)
    parser.add_argument("--use_vf", action="store_true", default=False)
    parser.add_argument("--multithread_mpc", action="store_true", default=False)
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
    elif args.env == "d4rl-hard-ant-maze":
        from simple_rl.tasks.d4rl_ant_maze.D4RLAntMazeMDPClass import D4RLAntMazeMDP
        overall_mdp = D4RLAntMazeMDP(maze_size="large", seed=args.seed, render=args.render)
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

    chainer = ModelBasedSkillChaining(mdp=overall_mdp, max_steps=args.steps, use_vf=args.use_vf,
                                      use_model=args.use_model, use_dense_rewards=args.use_dense_rewards,
                                      experiment_name=args.experiment_name, device=args.device,
                                      gestation_period=args.gestation_period, buffer_length=args.buffer_len,
                                      generate_init_gif=False, seed=args.seed, multithread_mpc=args.multithread_mpc)

    planner = SkillGraphPlanner(mdp=overall_mdp,
                                chainer=chainer,
                                experiment_name=args.experiment_name,
                                seed=args.seed,
                                use_vf=args.use_vf)

    dsg_agent = DeepSkillGraphAgent(mdp=overall_mdp,
                                    dsc_agent=chainer,
                                    planning_agent=planner,
                                    salient_event_freq=args.salient_event_freq,
                                    experiment_name=args.experiment_name,
                                    seed=args.seed,
                                    plot_gc_value_functions=args.plot_gc_value_functions)
    num_successes = dsg_agent.dsg_run_loop(episodes=args.episodes, num_steps=args.steps)