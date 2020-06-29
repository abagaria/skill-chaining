import ipdb
import argparse
import random
from copy import deepcopy
from simple_rl.agents.func_approx.dsc.SalientEventClass import SalientEvent, LearnedSalientEvent, DCOSalientEvent
from simple_rl.agents.func_approx.dsc.SkillChainingAgentClass import SkillChaining
from simple_rl.agents.func_approx.dsc.OptionClass import Option
from simple_rl.agents.func_approx.dsc.SkillGraphPlanningAgentClass import SkillGraphPlanningAgent
from simple_rl.agents.func_approx.dsc.utils import *
from simple_rl.mdp import MDP, State
from simple_rl.mdp.GoalDirectedMDPClass import GoalDirectedMDP


# from simple_rl.agents.func_approx.dsc.CoveringOptions import CoveringOptions\


class DeepSkillGraphAgent(object):
    def __init__(self, mdp, dsc_agent, planning_agent, salient_event_freq, use_hard_coded_events, use_dco,
                 dco_use_xy_prior, experiment_name, seed, threshold, use_smdp_replay_buffer, plotter):
        """
        This agent will interleave planning with the `planning_agent` and chaining with
        the `dsc_agent`.
        Args:
            mdp (GoalDirectedMDP)
            dsc_agent (SkillChaining)
            planning_agent (SkillGraphPlanningAgent)
            salient_event_freq (int)
            use_hard_coded_events (bool)
            use_dco (bool)
            dco_use_xy_prior (bool)
            experiment_name (str)
            seed (int)
            plotter (SkillChainingPlotter)
        """
        self.mdp = mdp
        self.dsc_agent = dsc_agent
        self.planning_agent = planning_agent
        self.salient_event_freq = salient_event_freq
        self.use_hard_coded_events = use_hard_coded_events
        self.use_dco = use_dco
        self.dco_use_xy_prior = dco_use_xy_prior
        self.experiment_name = experiment_name
        self.seed = seed
        self.threshold = threshold
        self.use_smdp_replay_buffer = use_smdp_replay_buffer
        self.plotter = plotter

        self.num_covering_options_generated = 0
        self.generated_salient_events = []
        self.most_recent_generated_salient_events = (None, None)
        self.last_event_creation_episode = -1
        self.num_successive_rejections = 0

        if self.use_hard_coded_events:
            assert not self.use_dco

    def select_goal_salient_event(self, state):
        """

        Args:
            state (State)

        Returns:
            target_event (SalientEvent)
        """
        # TODO: Use the current_salient_events in the future so that we prioritize new events coming in
        candidate_salient_events = self.generate_candidate_salient_events()

        # When you have finished chaining all the salient events, keep cycling between salient
        # events to continue improving your existing skills until there is a new salient event in town
        num_tries = 0
        target_event = None

        while target_event is None and num_tries < 100 and len(candidate_salient_events) > 0:
            target_event = random.sample(candidate_salient_events, k=1)[0]

            # If you are already at the target_event, then re-sample
            if target_event(state):
                target_event = None

            num_tries += 1

        if target_event is not None:
            print(f"Deep skill graphs target event: {target_event}")

        return target_event

    def dsg_run_loop(self, episodes, num_steps):
        successes = []
        if self.use_smdp_replay_buffer:
            replay_buffer = self.dsc_agent.agent_over_options.replay_buffer
        else:
            replay_buffer = self.dsc_agent.global_option.solver.replay_buffer

        for episode in range(episodes):

            # if hardcoded events are not provided, we can discover them using DCO or random sampling
            if not self.use_hard_coded_events:
                if self.should_generate_new_salient_event(episode):
                    self.discover_new_salient_event(replay_buffer, episode)

            step_number = 0
            self.mdp.reset()

            print("*" * 80)
            print(f"[DeepSkillGraphAgentClass] Episode {episode}: Resetting MDP to start state")
            print("*" * 80)

            state = deepcopy(self.mdp.cur_state)

            while step_number < num_steps:
                goal_salient_event = self.select_goal_salient_event(state)

                if goal_salient_event is None:
                    self.take_random_action()
                    step_number += 1
                    success = False
                else:
                    goal_state = goal_salient_event.target_state
                    step_number, success = self.planning_agent.planning_run_loop(start_episode=episode,
                                                                                 goal_state=goal_state,
                                                                                 goal_salient_event=goal_salient_event,
                                                                                 step_number=step_number,
                                                                                 to_reset=False)

                state = deepcopy(self.mdp.cur_state)

                if success:
                    print(f"[DeepSkillGraphAgentClass] successfully reached {goal_salient_event}")

                successes.append(success)

                # in the task agnostic setting, we end the episode if we have reached the goal
                if not self.mdp.task_agnostic and state.is_terminal():
                    print(f"[DeepSkillGraphAgentClass] successfully reached MDP Goal State")
                    break

            if self.plotter is not None and episode % 20 == 0 and episode > 0:
                self.plotter.generate_episode_plots(self.dsc_agent, episode)

        return successes

    def discover_new_salient_event(self, replay_buffer, episode):
        # currently threshold and beta are hardcoded
        c_option_idx = self.num_covering_options_generated
        self.num_covering_options_generated += 1
        buffer_type = "smdp" if self.use_smdp_replay_buffer else "global"

        if self.use_dco:
            ipdb.set_trace()
            # c_option = CoveringOptions(replay_buffer, obs_dim=self.mdp.state_space_size(), feature=None,
            #                            num_training_steps=1000,
            #                            option_idx=c_option_idx,
            #                            name=f"covering-options-{c_option_idx}_{buffer_type}_threshold-{self.threshold}",
            #                            threshold=self.threshold,
            #                            beta=0.1)
            #                            # use_xy_prior=self.dco_use_xy_prior)

            # low_event_idx = len(self.mdp.all_salient_events_ever) + 1
            # low_salient_event = DCOSalientEvent(c_option, low_event_idx, replay_buffer, is_low=True)
            # reject_low = self.add_salient_event(low_salient_event, episode)

            # high_event_idx = len(self.mdp.all_salient_events_ever) + 1
            # high_salient_event = DCOSalientEvent(c_option, high_event_idx, replay_buffer, is_low=False)
            # reject_high = self.add_salient_event(high_salient_event, episode)

            # self.most_recent_generated_salient_events = (
            #     low_salient_event if not reject_low else None,
            #     high_salient_event if not reject_high else None,
            # )

            # plot_dco_salient_event_comparison(low_salient_event,
            #                                   high_salient_event,
            #                                   replay_buffer,
            #                                   episode,
            #                                   reject_low,
            #                                   reject_high,
            #                                   self.experiment_name)
        else:
            salient_event = self.mdp.sample_salient_event()
            ipdb.set_trace()
            self.add_salient_event(salient_event, episode)
            print(f"Generated {salient_event}")

    def add_salient_event(self, salient_event, episode):
        reject = self.should_reject_discovered_salient_event(salient_event)

        if reject:
            self.num_successive_rejections += 1
        else:
            print(f"Accepted {salient_event}")

            self.generated_salient_events.append(salient_event)
            self.mdp.add_new_target_event(salient_event)
            self.last_event_creation_episode = episode
            self.num_successive_rejections = 0

            # Create a skill chain targeting the new event
            self.dsc_agent.create_chain_targeting_new_salient_event(salient_event)

        return reject

    def should_reject_discovered_salient_event(self, salient_event):
        """
        Reject the discovered salient event if it is the same as an old one or inside the initiation set of an option.

        Args:
            salient_event (SalientEvent)

        Returns:
            should_reject (bool)
        """
        if any([event(salient_event.target_state) for event in self.mdp.get_all_target_events_ever()]):
            return True

        if self.dsc_agent.state_in_any_completed_option(salient_event.target_state):
            return True

        return False

    def should_generate_new_salient_event(self, episode):
        return True
        # if episode < 5:
        #     return False
        # elif episode == 5:
        #     return True

        def _events_chained(low_event, high_event):
            chains = self.dsc_agent.chains
            chains_targeting_low_event = [chain for chain in chains if chain.target_salient_event == low_event and
                                          chain.is_chain_completed(chains)]
            chains_targeting_high_event = [chain for chain in chains if chain.target_salient_event == high_event and
                                           chain.is_chain_completed(chains)]
            return (
                    (len(chains_targeting_low_event) > 0 or low_event is None) and
                    (len(chains_targeting_high_event) > 0 or high_event is None)
            )

        most_recent_event = self.most_recent_generated_salient_events  # type: Tuple[DCOSalientEvent, DCOSalientEvent]

        if _events_chained(*most_recent_event):
            return True

        if episode - self.last_event_creation_episode > self.salient_event_freq:
            return True

        return False

    def generate_candidate_salient_events(self):  # TODO: This needs to happen multiple times, not just once
        if self.should_set_off_learning_backward_options():
            self.set_off_learning_backward_options()

            return self.mdp.get_all_target_events_ever() + [self.mdp.get_start_state_salient_event()]

        return self.mdp.get_all_target_events_ever()

    def are_all_original_salient_events_forward_chained(self):
        original_salient_events = self.mdp.get_original_target_events()
        for event in original_salient_events:  # type: SalientEvent
            forward_chains = [chain for chain in self.dsc_agent.chains if
                              chain.target_salient_event == event and not chain.is_backward_chain]
            if any([not chain.is_chain_completed(self.dsc_agent.chains) for chain in forward_chains]):
                return False
        return True

    def should_set_off_learning_backward_options(self):
        if len(self.mdp.get_original_target_events()) > 0:
            assert self.use_hard_coded_events, "Expected original events to be hard coded ones"
            return self.are_all_original_salient_events_forward_chained() \
                   and not self.dsc_agent.create_backward_options \
                   and not self.dsc_agent.learn_backward_options_offline

        # If we have tried to create an event outside the graph a lot of times and failed,
        # then that probably means that we are done with forward-chaining and we can now
        # begin learning our backward options
        return self.num_successive_rejections >= 100 \
               and not self.dsc_agent.create_backward_options \
               and not self.dsc_agent.learn_backward_options_offline

    def set_off_learning_backward_options(self):
        self.dsc_agent.create_backward_options = True
        self.dsc_agent.learn_backward_options_offline = True

        # Pretend as if we just learned a new leaf option and we want to check
        # if creating a backward chain is in order
        for chain in self.dsc_agent.chains:
            leaf_option = chain.get_leaf_nodes_from_skill_chain()[0]

            if leaf_option is not None:

                trained_back_options = []

                if self.dsc_agent.start_state_salience:
                    trained_back_options += self.dsc_agent.manage_skill_chain_to_start_state(leaf_option)
                if self.dsc_agent.event_intersection_salience:
                    trained_back_options += self.dsc_agent.manage_intersection_between_option_and_events(leaf_option)
                if self.dsc_agent.option_intersection_salience:
                    raise NotImplementedError("Option intersection salience")

                for newly_created_option in trained_back_options:  # type: Option
                    self.planning_agent.add_newly_created_option_to_plan_graph(newly_created_option)

    def take_random_action(self):
        state = deepcopy(self.mdp.cur_state)
        action = self.mdp.sample_random_action()
        reward, next_state = self.mdp.execute_agent_action(action)
        done = self.mdp.is_goal_state(next_state)

        self.dsc_agent.global_option.solver.step(state.features(), action, reward, next_state.features(), done)
        self.dsc_agent.agent_over_options.step(state.features(), self.dsc_agent.global_option.option_idx, reward, next_state.features(),
                                               done, 1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment_name", type=str, help="Experiment Name")
    parser.add_argument("--device", type=str, help="cpu/cuda:0/cuda:1")
    parser.add_argument("--env", type=str, help="name of gym environment", default="Pendulum-v0")
    parser.add_argument("--pretrained", action="store_true", help="whether or not to load pretrained options", default=False)
    parser.add_argument("--seed", type=int, help="Random seed for this run (default=0)", default=0)
    parser.add_argument("--episodes", type=int, help="# episodes", default=200)
    parser.add_argument("--steps", type=int, help="# steps", default=1000)
    parser.add_argument("--subgoal_reward", type=float, help="SkillChaining subgoal reward", default=0.)
    parser.add_argument("--lr_a", type=float, help="DDPG Actor learning rate", default=1e-4)
    parser.add_argument("--lr_c", type=float, help="DDPG Critic learning rate", default=1e-3)
    parser.add_argument("--ddpg_batch_size", type=int, help="DDPG Batch Size", default=64)
    parser.add_argument("--render", action="store_true", help="Render the mdp env", default=False)
    parser.add_argument("--option_timeout", action="store_true", help="Whether option times out at 200 steps", default=False)
    parser.add_argument("--generate_plots", action="store_true", help="Whether or not to generate plots", default=False)
    parser.add_argument("--tensor_log", action="store_true", help="Enable tensorboard logging", default=False)
    parser.add_argument("--control_cost", action="store_true", help="Penalize high actuation solutions", default=False)
    parser.add_argument("--dense_reward", action="store_true", help="Use dense/sparse rewards", default=False)
    parser.add_argument("--max_num_options", type=int, help="Max number of options we can learn", default=5)
    parser.add_argument("--num_subgoal_hits", type=int, help="Number of subgoal hits to learn an option", default=3)
    parser.add_argument("--buffer_len", type=int, help="buffer size used by option to create init sets", default=20)
    parser.add_argument("--classifier_type", type=str, help="ocsvm/elliptic for option initiation clf", default="ocsvm")
    parser.add_argument("--init_q", type=str, help="compute/zero", default="zero")
    parser.add_argument("--use_smdp_update", action="store_true", help="sparse/SMDP update for option policy", default=False)
    parser.add_argument("--use_start_state_salience", action="store_true", default=False)
    parser.add_argument("--use_option_intersection_salience", action="store_true", default=False)
    parser.add_argument("--use_event_intersection_salience", action="store_true", default=False)
    parser.add_argument("--pretrain_option_policies", action="store_true", default=False)
    parser.add_argument("--create_backward_options", action="store_true", default=False)
    parser.add_argument("--learn_backward_options_offline", action="store_true", default=False)
    parser.add_argument("--use_warmup_phase", action="store_true", default=False)
    parser.add_argument("--update_global_solver", action="store_true", default=False)
    parser.add_argument("--salient_event_freq", type=int, help="Create a salient event every salient_event_freq episodes", default=50)
    parser.add_argument("--use_hard_coded_events", action="store_true", help="Whether to use hard-coded salient events", default=False)
    parser.add_argument("--dco_use_xy_prior", action="store_true", default=False)
    parser.add_argument("--plot_rejected_events", action="store_true", default=False)
    parser.add_argument("--use_dco", action="store_true", default=False)
    parser.add_argument("--use_ucb", action="store_true", default=False)
    parser.add_argument("--threshold", type=int, help="Threshold determining size of termination set", default=0.1)
    parser.add_argument("--use_smdp_replay_buffer", action="store_true", help="Whether to use a replay buffer that has options",
                        default=False)  # we should use this
    parser.add_argument("--generate_n_clips", type=int, help="The number of run clips to generate", default=10)
    parser.add_argument("--wait_n_episodes_between_clips", type=int, help="The number of episodes to wait between clip generation",
                        default=0)
    args = parser.parse_args()

    mdp_plotter = None
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
                                    render=args.render,
                                    use_hard_coded_events=args.use_hard_coded_events)
        state_dim = overall_mdp.state_space_size()
        action_dim = overall_mdp.action_space_size()
    elif args.env == "d4rl-ant-maze":
        from simple_rl.tasks.d4rl_ant_maze.D4RLAntMazeMDPClass import D4RLAntMazeMDP

        overall_mdp = D4RLAntMazeMDP(maze_size="medium", seed=args.seed, render=args.render)
        state_dim = overall_mdp.state_space_size()
        action_dim = overall_mdp.action_space_size()
    elif args.env == "d4rl-point-maze":
        from simple_rl.tasks.d4rl_point_maze.D4RLPointMazeMDPClass import D4RLPointMazeMDP

        overall_mdp = D4RLPointMazeMDP(seed=args.seed, render=args.render)
        state_dim = overall_mdp.state_space_size()
        action_dim = overall_mdp.action_space_size()
    elif "reacher" in args.env.lower():
        from simple_rl.tasks.dm_fixed_reacher.FixedReacherMDPClass import FixedReacherMDP

        overall_mdp = FixedReacherMDP(seed=args.seed, difficulty=args.difficulty, render=args.render)
        state_dim = overall_mdp.init_state.features().shape[0]
        action_dim = overall_mdp.env.action_spec().minimum.shape[0]
    elif "maze" in args.env.lower():
        from simple_rl.tasks.point_maze.PointMazeMDPClass import PointMazeMDP

        overall_mdp = PointMazeMDP(dense_reward=args.dense_reward, seed=args.seed, render=args.render)
        state_dim = 6
        action_dim = 2
    elif "point" in args.env.lower():
        from simple_rl.tasks.point_env.PointEnvMDPClass import PointEnvMDP

        overall_mdp = PointEnvMDP(control_cost=args.control_cost, render=args.render)
        state_dim = 4
        action_dim = 2
    elif "sawyer" in args.env.lower():
        from simple_rl.tasks.leap_wrapper.LeapWrapperMDPClass import LeapWrapperMDP
        task_agnostic = 'agnostic' in args.env.lower()

        overall_mdp = LeapWrapperMDP(
            task_agnostic=task_agnostic,
            episode_length=args.steps,
            use_hard_coded_events=args.use_hard_coded_events,
            dense_reward=args.dense_reward,
            render=args.render,
            generate_n_clips=args.generate_n_clips,
            wait_n_episodes_between_clips=args.wait_n_episodes_between_clips,
            movie_output_folder=args.experiment_name)
        overall_mdp.env.seed(args.seed)
        if args.generate_plots:
            from simple_rl.tasks.leap_wrapper.LeapWrapperPlotter import LeapWrapperPlotter

            mdp_plotter = LeapWrapperPlotter("sawyer", args.experiment_name, overall_mdp)
            overall_mdp.movie_renderer.create_folder(mdp_plotter.path)

    else:
        from simple_rl.tasks.gym.GymMDPClass import GymMDP

        overall_mdp = GymMDP(args.env, render=args.render)
        state_dim = overall_mdp.env.observation_space.shape[0]
        action_dim = overall_mdp.env.action_space.shape[0]
        overall_mdp.env.seed(args.seed)

    print("Training skill chaining agent from scratch with a subgoal reward {}".format(args.subgoal_reward))
    print("MDP InitState = ", overall_mdp.init_state)

    q0 = 0. if args.init_q == "zero" else None

    chainer = SkillChaining(overall_mdp, args.steps, args.lr_a, args.lr_c, args.ddpg_batch_size,
                            seed=args.seed, subgoal_reward=args.subgoal_reward,
                            num_subgoal_hits_required=args.num_subgoal_hits,
                            enable_option_timeout=args.option_timeout, init_q=q0,
                            use_full_smdp_update=args.use_smdp_update,
                            tensor_log=args.tensor_log,
                            device=args.device, buffer_length=args.buffer_len,
                            start_state_salience=args.use_start_state_salience,
                            option_intersection_salience=args.use_option_intersection_salience,
                            event_intersection_salience=args.use_event_intersection_salience,
                            pretrain_option_policies=args.pretrain_option_policies,
                            create_backward_options=args.create_backward_options,
                            learn_backward_options_offline=args.learn_backward_options_offline,
                            dense_reward=args.dense_reward,
                            update_global_solver=args.update_global_solver,
                            use_warmup_phase=args.use_warmup_phase,
                            experiment_name=args.experiment_name,
                            plotter=mdp_plotter)

    assert any([args.use_start_state_salience, args.use_option_intersection_salience, args.use_event_intersection_salience])

    planner = SkillGraphPlanningAgent(mdp=overall_mdp,
                                      chainer=chainer,
                                      experiment_name=args.experiment_name,
                                      seed=args.seed,
                                      initialize_graph=False,
                                      pretrain_option_policies=args.pretrain_option_policies,
                                      use_ucb=args.use_ucb)

    dsg_agent = DeepSkillGraphAgent(mdp=overall_mdp,
                                    dsc_agent=chainer,
                                    planning_agent=planner,
                                    salient_event_freq=args.salient_event_freq,
                                    use_hard_coded_events=args.use_hard_coded_events,
                                    use_dco=args.use_dco,
                                    dco_use_xy_prior=args.dco_use_xy_prior,
                                    experiment_name=args.experiment_name,
                                    seed=args.seed,
                                    threshold=args.threshold,
                                    use_smdp_replay_buffer=args.use_smdp_replay_buffer,
                                    plotter=mdp_plotter)

    num_successes = dsg_agent.dsg_run_loop(episodes=args.episodes, num_steps=args.steps)
    ipdb.set_trace()
