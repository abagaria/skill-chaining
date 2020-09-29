import ipdb
import argparse
import random
import pickle
from copy import deepcopy
from simple_rl.agents.func_approx.dsc.SalientEventClass import SalientEvent
from simple_rl.agents.func_approx.dsc.SkillChainingAgentClass import SkillChaining
from simple_rl.agents.func_approx.dsc.SkillGraphPlannerClass import SkillGraphPlanner
from simple_rl.agents.func_approx.dsc.OptionClass import Option
from simple_rl.agents.func_approx.dsc.utils import *
from simple_rl.mdp import MDP, State
from simple_rl.mdp.GoalDirectedMDPClass import GoalDirectedMDP


class DeepSkillGraphAgent(object):
    def __init__(self, mdp, dsc_agent, planning_agent, salient_event_freq, event_after_reject_freq,
                 experiment_name, seed, threshold, use_smdp_replay_buffer, rejection_criteria):
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

        assert rejection_criteria in ("use_effect_sets", "use_init_sets"), rejection_criteria

        self.num_covering_options_generated = 0
        self.generated_salient_events = []
        self.most_recent_generated_salient_events = (None, None)
        self.last_event_creation_episode = -1

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
        current_events = [event for event in events if event(state)]
        candidate_salient_events = self.generate_candidate_salient_events(state)
        closest_pair = self.planning_agent.get_closest_pair_of_vertices(current_events, candidate_salient_events)
        if closest_pair is not None:
            assert len(closest_pair) == 2, closest_pair
            return closest_pair[1]

    def select_goal_salient_event(self, state, selection_criteria="closest"):
        """

        Args:
            state (State)
            selection_criteria (str)

        Returns:
            target_event (SalientEvent)
        """
        assert selection_criteria in ("closest", "random"), selection_criteria
        events = self.mdp.get_all_target_events_ever() + [self.mdp.get_start_state_salient_event()]
        if len(self.mdp.get_all_target_events_ever()) > 0:
            if selection_criteria == "closest":
                selected_event = self._select_closest_unconnected_salient_event(state, events)
                if selected_event is not None:
                    print(f"[Closest] Deep skill graphs target event: {selected_event}")
                    return selected_event
            return self._randomly_select_salient_event(state, events)

    def dsg_run_loop(self, episodes, num_steps, start_state=None, test_event=None, eval_mode=False):
        successes = []

        for episode in range(episodes):

            if self.should_generate_new_salient_events(episode) and not eval_mode:
                self.generate_new_salient_events(episode)

            step_number = 0
            random_episodic_trajectory = []
            self.reset(episode, start_state)

            state = deepcopy(self.mdp.cur_state)

            while step_number < num_steps:
                goal_salient_event = self.select_goal_salient_event(state) if test_event is None else test_event

                if goal_salient_event is None:
                    random_transition = self.take_random_action()
                    step_number += 1
                    success = False
                    random_episodic_trajectory.append(random_transition)
                else:
                    self.create_skill_chains_if_needed(state, goal_salient_event, eval_mode)

                    step_number, success = self.planning_agent.run_loop(state=state,
                                                                        goal_salient_event=goal_salient_event,
                                                                        episode=episode,
                                                                        step=step_number,
                                                                        eval_mode=eval_mode)

                state = deepcopy(self.mdp.cur_state)

                if success:
                    print(f"[DeepSkillGraphAgentClass] successfully reached {goal_salient_event}")

                successes.append(success)

                if eval_mode:
                    break

            if episode < 5:
                goal_state = self.mdp.get_position(self.mdp.sample_random_state())
                self.dsc_agent.global_option_experience_replay(random_episodic_trajectory, goal_state=goal_state)

            # TODO: commented out because doesn't work with pickling
            # if episode > 0 and episode % 50 == 0 and args.plot_gc_value_functions:
            if episode > 0 and episode % 50 == 0:
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

    def generate_new_salient_events(self, episode):

        event_idx = len(self.mdp.all_salient_events_ever) + 1
        target_state = self.mdp.sample_random_state()[:2]
        low_salient_event = SalientEvent(target_state, event_idx)
        reject_low = self.add_salient_event(low_salient_event)

        event_idx = len(self.mdp.all_salient_events_ever) + 1
        target_state = self.mdp.sample_random_state()[:2]
        high_salient_event = SalientEvent(target_state, event_idx)
        reject_high = self.add_salient_event(high_salient_event)

        print(f"Generated {low_salient_event} and {high_salient_event}")

        self.last_event_creation_episode = episode

        self.most_recent_generated_salient_events = (
            low_salient_event if not reject_low else None,
            high_salient_event if not reject_high else None,
        )

    def add_salient_event(self, salient_event):
        reject = self.should_reject_discovered_salient_event(salient_event)

        if not reject:
            print(f"[DSG Agent] Accepted {salient_event}")

            self.generated_salient_events.append(salient_event)
            self.mdp.add_new_target_event(salient_event)

        return reject

    def is_path_under_construction(self, init_salient_event, target_salient_event):
        assert isinstance(init_salient_event, SalientEvent), f"{type(init_salient_event)}"
        assert isinstance(target_salient_event, SalientEvent), f"{type(target_salient_event)}"

        chains = self.dsc_agent.chains
        unfinished_chains = [chain for chain in chains if not chain.is_chain_completed()]

        match = lambda c: c.init_salient_event == init_salient_event and c.target_salient_event == target_salient_event
        return any([match(chain) for chain in unfinished_chains])

    def should_reject_discovered_salient_event(self, salient_event):
        """
        Reject the discovered salient event if it is the same as an old one or inside the initiation set of an option.

        Args:
            salient_event (SalientEvent)
            rejection_criteria (str)

        Returns:
            should_reject (bool)
        """
        existing_target_events = self.mdp.get_all_target_events_ever() + [self.mdp.get_start_state_salient_event()]
        if any([event(salient_event.target_state) for event in existing_target_events]):
            return True

        rejection_function = self.planning_agent.is_state_inside_vertex if self.rejection_criteria == "use_effect_sets"\
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

    def create_skill_chains_if_needed(self, state, goal_salient_event, eval_mode):
        current_salient_event = self._get_current_salient_event(state)

        if current_salient_event is not None:
            if not self.planning_agent.plan_graph.does_path_exist(state, goal_salient_event) and \
                    not self.is_path_under_construction(current_salient_event, goal_salient_event):
                print(f"[DeepSkillGraphsAgent] Creating chain from {current_salient_event} -> {goal_salient_event}")
                self.dsc_agent.create_chain_targeting_new_salient_event(salient_event=goal_salient_event,
                                                                        init_salient_event=current_salient_event,
                                                                        eval_mode=eval_mode)

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
                                    rejection_criteria=args.rejection_criteria)
    num_successes = dsg_agent.dsg_run_loop(episodes=args.episodes, num_steps=args.steps)
