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
from simple_rl.agents.func_approx.dsc.UCBGraphNodeSelectionAgentClass import UCBNodeSelectionAgent


class DeepSkillGraphAgent(object):
    def __init__(self, mdp, dsc_agent, planning_agent, discover_salient_events,
                 salient_event_freq, experiment_name, seed, plot_gc_value_functions):
        """
        This agent will interleave planning with the `planning_agent` and chaining with
        the `dsc_agent`.
        Args:
            mdp (GoalDirectedMDP)
            dsc_agent (ModelBasedSkillChaining)
            planning_agent (SkillGraphPlanner)
            discover_salient_events (bool)
            salient_event_freq (int)
            experiment_name (str)
            seed (int)
            plot_gc_value_functions (bool)
        """
        self.mdp = mdp
        self.dsc_agent = dsc_agent
        self.planning_agent = planning_agent
        self.discover_salient_events = discover_salient_events
        self.salient_event_freq = salient_event_freq
        self.experiment_name = experiment_name
        self.seed = seed
        self.salient_event_freq = salient_event_freq
        self.plot_gc_value_functions = plot_gc_value_functions

        assert isinstance(self.dsc_agent, ModelBasedSkillChaining)

        self.num_warmup_episodes = 50
        self.current_event_idx = 0

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

    def _select_closest_unconnected_salient_event(self, current_event):
        if current_event is not None:
            node_selector = self.planning_agent.ucb_selectors[current_event]
            selected_pair = node_selector.act()
            return selected_pair

    def _select_closest_unfinished_chain_init_event(self, state):
        unfinished_chains = [chain for chain in self.dsc_agent.chains if not chain.is_chain_completed()]
        unfinished_chains = [chain for chain in unfinished_chains if not chain.init_salient_event(state)]
        init_salient_events = [chain.init_salient_event for chain in unfinished_chains]
        if len(init_salient_events) > 0:
            return random.choice(init_salient_events)

    def select_goal_salient_event(self, state, current_salient_event):
        """

        Args:
            state (State)
            current_salient_event (SalientEvent)

        Returns:
            target_event (SalientEvent)
        """

        if len(self.mdp.get_all_target_events_ever()) > 0:
            selected_pair = self._select_closest_unconnected_salient_event(current_salient_event)
            if selected_pair is not None:
                print(f"[Closest] DSG selected pair: {selected_pair}")
                return selected_pair
            
            events = self.mdp.get_all_target_events_ever() + [self.mdp.get_start_state_salient_event()]
            target = self._randomly_select_salient_event(state, events)
            return current_salient_event, target

    def monte_warm_start(self, mdp):
        " Random rollouts until we have at least one eg for each salient event. "
        done = False
        reset = False

        while not done:
            state = deepcopy(mdp.cur_state)
            action = mdp.sample_random_action()
            reward, next_state = mdp.execute_agent_action(action)
            events = mdp.all_salient_events_ever + [mdp.get_start_state_salient_event()]
            
            for event in events:
                if event(state):
                    event.add_trigger_point(state)
                    reset = True
                if event(next_state):
                    reset = True
                    event.add_trigger_point(next_state)
            
            num_completed = [len(event.trigger_points) > 0 for event in events] 

            if all(num_completed):
                done = True

            if mdp.cur_state.is_terminal() or reset:
                reset = False
                self.reset(0, random.choice(mdp.spawn_states))

    def dsg_event_discovery_loop(self, current_episode):
        if self.current_event_idx < len(self.mdp.candidate_salient_positions):
            salient_position = self.mdp.candidate_salient_positions[self.current_event_idx]

            if salient_position == self.mdp.key_location:
                predicate = lambda s: self.mdp.has_key(s) and s.position in self.mdp.key_locations
                goal_salient_event = SalientEvent(target_state=None,
                                                  predicate=predicate,
                                                  event_idx=self.current_event_idx+1,
                                                  tolerance=2.,
                                                  is_key_event=True)
            elif salient_position == self.mdp.left_door_location:
                predicate = lambda s: self.mdp.has_key(s) and \
                                        s.position[0] <= self.mdp.left_door_location[0] and \
                                        s.position[1] >= self.mdp.left_door_location[1]
                goal_salient_event = SalientEvent(target_state=None, 
                                                  predicate=predicate,
                                                  event_idx=self.current_event_idx+1,
                                                  tolerance=2.)
            elif salient_position == self.mdp.right_door_location:
                predicate = lambda s: self.mdp.has_key(s) and \
                                        s.position[0] >= self.mdp.right_door_location[0] and \
                                        s.position[1] >= self.mdp.right_door_location[1]
                goal_salient_event = SalientEvent(target_state=None, 
                                                  predicate=predicate,
                                                  event_idx=self.current_event_idx+1,
                                                  tolerance=2.)
            else:
                goal_salient_event = SalientEvent(target_state=salient_position,
                                                  event_idx=self.current_event_idx+1,
                                                  tolerance=2.)

            target_events = self.mdp.all_salient_events_ever + [self.mdp.get_start_state_salient_event()]

            self.current_event_idx += 1
            self.add_salient_event(goal_salient_event)

            self.monte_warm_start(self.mdp)
            self._create_node_selector(goal_salient_event, target_events)

    def _create_node_selector(self, root_event, target_events):
        # A new salient event has no descendants yet and all other events are targets
        print(target_events)
        ucb_agent = UCBNodeSelectionAgent(root_event, 
                                          descendant_events=[],
                                          target_events=target_events)
        self.planning_agent.ucb_selectors[root_event] = ucb_agent

        # Make sure that `target_events` is not modified with the new event
        # We want to make sure that the `root_event` doesn't get added as a target_event
        num_new_events = len([self.mdp.get_start_state_salient_event()] + self.mdp.all_salient_events_ever)
        num_ucb_events = len(ucb_agent.target_events)
        assert num_new_events == num_ucb_events + 1, f"{num_new_events, num_ucb_events}"
        assert ucb_agent.root_event not in ucb_agent.target_events, ucb_agent.target_events

        # Update all the node selectors to account for the newly created one
        for event in self.planning_agent.ucb_selectors:
            selector = self.planning_agent.ucb_selectors[event]
            self.planning_agent._update_node_selector(selector)

    def add_salient_event(self, salient_event):
        print(f"[DSG Agent] Accepted {salient_event}")

        # Add the new event to the MDP
        self.mdp.add_new_target_event(salient_event)

    def should_generate_new_salient_events(self):
        f = self.planning_agent.plan_graph.does_path_exist
        reachable = [f(self.mdp.init_state, event) for event in self.mdp.all_salient_events_ever]
        return len(reachable) == 0 or all(reachable)

    def remove_salient_events(self):
        """ If the i-th event no longer is connected to the start-state, remove all the events that come after it. """
        def get_first_unconnected_event():
            for i, event in enumerate(self.mdp.all_salient_events_ever):
                if not self.planning_agent.plan_graph.does_path_exist(self.mdp.init_state, event):
                    return i

        def remove_events(starting_idx):
            if len(self.mdp.all_salient_events_ever) > starting_idx:
                for event in self.mdp.all_salient_events_ever[starting_idx:]:
                    print(f"************* Removing {event} ****************")
                    self.mdp.all_salient_events_ever.remove(event)
        
        idx = get_first_unconnected_event()

        if idx is not None:
            remove_events(idx+1)
            self.current_event_idx = idx + 1
            print("Set the current event idx to ", self.current_event_idx)

    def dsg_run_loop(self, episodes, num_steps, starting_episode=0):
        successes = []

        for episode in range(starting_episode, starting_episode+episodes):

            self.remove_salient_events()

            if self.should_generate_new_salient_events():
                self.dsg_event_discovery_loop(episode)

            step_number = 0
            random_episodic_trajectory = []
            self.reset(episode, start_state=None)

            state = deepcopy(self.mdp.cur_state)

            while step_number < num_steps and not state.is_terminal():
                current_salient_event = self._get_current_salient_event(state)
                selected_pair = self.select_goal_salient_event(state, current_salient_event)
                goal_salient_event = selected_pair[1]

                if goal_salient_event is None:
                    random_transition = self.take_random_action()
                    step_number += 1
                    success = False
                    random_episodic_trajectory.append(random_transition)
                else:

                    self.create_skill_chains_if_needed(state=state,
                                                       current_salient_event=current_salient_event,
                                                       goal_salient_event=goal_salient_event,
                                                       closest_event_pair=selected_pair)

                    step_number, success = self.planning_agent.run_loop(state=state,
                                                                        current_salient_event=current_salient_event,
                                                                        planner_goal_vertex=selected_pair[0],
                                                                        dsc_goal_vertex=selected_pair[1],
                                                                        goal_salient_event=goal_salient_event,
                                                                        episode=episode,
                                                                        step=step_number,
                                                                        eval_mode=False)

                state = deepcopy(self.mdp.cur_state)

                if success:
                    print(f"[DeepSkillGraphAgentClass] successfully reached {goal_salient_event}")

                successes.append(success)

            if episode > 0 and episode % 10 == 0:
                t0 = time.time()
                for option in self.planning_agent.plan_graph.option_nodes:
                    self.planning_agent.add_potential_edges(option)
                print(f"Took {time.time() - t0}s to add potential edges")

            if episode > 0 and episode % 100 == 0 and self.plot_gc_value_functions:
                ppo = self.dsc_agent.global_option.solver
                states = get_all_states(self.dsc_agent)
                events = self.mdp.get_all_target_events_ever() + [self.mdp.get_start_state_salient_event()]

                visualize_value_function(ppo, states, events, episode, self.experiment_name)

        return successes

    def is_path_under_construction(self, state, init_salient_event, target_salient_event):
        assert isinstance(state, (State, np.ndarray)), f"{type(State)}"
        assert isinstance(init_salient_event, SalientEvent), f"{type(init_salient_event)}"
        assert isinstance(target_salient_event, SalientEvent), f"{type(target_salient_event)}"

        match = lambda c: c.init_salient_event == init_salient_event and c.target_salient_event == target_salient_event
        if any([match(c) for c in self.dsc_agent.chains]):
            return True

        return False

        # TODO: Do we still need to check in the optimistic graph when we don't have a metric? 
        # events = self.mdp.get_all_target_events_ever() + [self.mdp.get_start_state_salient_event()]
        # current_salient_events = [event for event in events if event(state)]
        # under_construction = any([self.does_path_exist_in_optimistic_graph(current_salient_event, target_salient_event)
        #                           for current_salient_event in current_salient_events])
        # return under_construction

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
        self.planning_agent.mpc.step(state.features(), action, reward, next_state.features(), next_state.is_terminal())
        return state, action, reward, next_state

    def create_skill_chains_if_needed(self, state, current_salient_event, goal_salient_event, closest_event_pair):

        if current_salient_event is not None:
            if not self.planning_agent.plan_graph.does_path_exist(state, goal_salient_event) and \
                    not self.is_path_under_construction(state, current_salient_event, goal_salient_event):

                init, target = current_salient_event, goal_salient_event
                if closest_event_pair is not None:
                    init, target = closest_event_pair[0], closest_event_pair[1]

                if not self.is_path_under_construction(state, init, target):
                    print(f"[DeepSkillGraphsAgent] Creating chain from {init} -> {target}")
                    self.dsc_agent.create_new_chain(init_event=init, target_event=target)

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
            print(f"[DeepSkillGraphAgentClass] Episode {episode}: Resetting MDP to start state {self.mdp.cur_state.position}")
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
    parser.add_argument("--steps", type=int, help="# steps", default=100000)
    parser.add_argument("--subgoal_reward", type=float, help="SkillChaining subgoal reward", default=0.)
    parser.add_argument("--render", action="store_true", help="Render the mdp env", default=False)
    parser.add_argument("--use_dense_rewards", action="store_true", help="Use dense/sparse rewards", default=False)
    parser.add_argument("--gestation_period", type=int, help="Number of subgoal hits to learn an option", default=5)
    parser.add_argument("--buffer_len", type=int, help="buffer size used by option to create init sets", default=50)
    parser.add_argument("--salient_event_freq", type=int, help="Create a salient event every salient_event_freq episodes", default=50)
    parser.add_argument("--plot_rejected_events", action="store_true", default=False)
    parser.add_argument("--plot_gc_value_functions", action="store_true", default=False)
    parser.add_argument("--use_model", action="store_true", default=False)
    parser.add_argument("--use_vf", action="store_true", default=False)
    parser.add_argument("--multithread_mpc", action="store_true", default=False)
    parser.add_argument("--omit_event_discovery", action="store_true", default=False)
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
    elif args.env == "fetch-reach":
        from simple_rl.tasks.fetch.FetchMDPClass import FetchMDP
        overall_mdp = FetchMDP(env="reach", render=args.render, seed=args.seed, task_agnostic=True)
        state_dim = overall_mdp.state_space_size()
        action_dim = overall_mdp.action_space_size()
    elif args.env == "MontezumaRevenge":
        from simple_rl.tasks.gym.GymMDPClass import GymMDP
        overall_mdp = GymMDP(env_name=args.env, pixel_observation=True, render=args.render, seed=args.seed)
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
                                    discover_salient_events=not args.omit_event_discovery,
                                    experiment_name=args.experiment_name,
                                    seed=args.seed,
                                    plot_gc_value_functions=args.plot_gc_value_functions)
    
    num_successes = dsg_agent.dsg_run_loop(episodes=args.episodes, num_steps=args.steps)