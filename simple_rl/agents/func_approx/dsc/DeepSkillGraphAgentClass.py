import ipdb
import argparse
import random
from copy import deepcopy
from simple_rl.agents.func_approx.dsc.SalientEventClass import SalientEvent, DCOSalientEvent
from simple_rl.agents.func_approx.dsc.SkillChainingAgentClass import SkillChaining
from simple_rl.agents.func_approx.dsc.OptionClass import Option
from simple_rl.agents.func_approx.dsc.SkillGraphPlannerClass import SkillGraphPlanner
from simple_rl.agents.func_approx.dsc.utils import *
from simple_rl.mdp import MDP, State
from simple_rl.mdp.GoalDirectedMDPClass import GoalDirectedMDP
from simple_rl.agents.func_approx.dsc.CoveringOptions import CoveringOptions
from simple_rl.agents.func_approx.dsc.dynamics.mpc import MPC

from tqdm import tqdm

class DeepSkillGraphAgent(object):
    def __init__(self, mdp, dsc_agent, planning_agent, salient_event_freq, event_after_reject_freq, use_hard_coded_events,
                 use_dco, dco_use_xy_prior, allow_backward_options, experiment_name, seed, threshold, use_smdp_replay_buffer):
        """
        This agent will interleave planning with the `planning_agent` and chaining with
        the `dsc_agent`.
        Args:
            mdp (GoalDirectedMDP)
            dsc_agent (SkillChaining)
            planning_agent (SkillGraphPlanner)
            salient_event_freq (int)
            use_hard_coded_events (bool)
            use_dco (bool)
            dco_use_xy_prior (bool)
            allow_backward_options (bool)
            experiment_name (str)
            seed (int)
        """
        self.mdp = mdp
        self.dsc_agent = dsc_agent
        self.planning_agent = planning_agent
        self.salient_event_freq = salient_event_freq
        self.use_hard_coded_events = use_hard_coded_events
        self.use_dco = use_dco
        self.dco_use_xy_prior = dco_use_xy_prior
        self.allow_backward_options = allow_backward_options
        self.experiment_name = experiment_name
        self.seed = seed
        self.threshold = threshold
        self.salient_event_freq = salient_event_freq
        self.event_after_reject_freq = event_after_reject_freq
        self.use_smdp_replay_buffer = use_smdp_replay_buffer

        self.num_covering_options_generated = 0
        self.generated_salient_events = []
        self.most_recent_generated_salient_events = (None, None)
        self.last_event_creation_episode = -1
        self.last_event_rejection_episode = -1
        self.num_successive_rejections = 0

        self.mpc = MPC(31, 8, self.dsc_agent.device)

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
            if (episode % 100 == 0 and episode > 0) or episode == 5:
                states, actions, states_p = self._prepare_dataset()
                print("Size of dataset: {}".format(len(states)))
                self.mpc.load_data(states, actions, states_p)
                self.mpc.train(epochs=200, batch_size=512)

            if self.should_generate_new_salient_events(episode):
                self.generate_new_salient_events(replay_buffer, episode)

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
                    step_number, success = self.planning_agent.run_loop(mpc=self.mpc,
                                                                        state=state,
                                                                        goal_salient_event=goal_salient_event,
                                                                        episode=episode,
                                                                        step=step_number,
                                                                        eval_mode=False,
                                                                        to_reset=False)

                state = deepcopy(self.mdp.cur_state)

                if success:
                    print(f"[DeepSkillGraphAgentClass] successfully reached {goal_salient_event}")

                successes.append(success)

        return successes
    
    def _prepare_dataset(self):
        data = self.dsc_agent.global_option.solver.replay_buffer.memory
        states = []
        actions = []
        states_p = []
        for state, action, reward, next_state, terminal in tqdm(data, desc='Preparing dataset for MPC'):
            states.append(state)
            actions.append(action)
            states_p.append(next_state)
        return states, actions, states_p

    def generate_new_salient_events(self, replay_buffer, episode):
        # currently threshold and beta are hardcoded
        c_option_idx = self.num_covering_options_generated
        self.num_covering_options_generated += 1
        buffer_type = "smdp" if self.use_smdp_replay_buffer else "global"

        if self.use_dco and len(self.mdp.all_salient_events_ever) > 0:
            c_option = CoveringOptions(replay_buffer, obs_dim=self.mdp.state_space_size(), feature=None,
                                       num_training_steps=1000,
                                       option_idx=c_option_idx,
                                       name=f"covering-options-{c_option_idx}_{buffer_type}_threshold-{self.threshold}",
                                       threshold=self.threshold,
                                       beta=0.1)
                                       # use_xy_prior=self.dco_use_xy_prior)

            low_event_idx = len(self.mdp.all_salient_events_ever) + 1
            low_salient_event = DCOSalientEvent(c_option, low_event_idx, is_low=True)
            reject_low = self.add_salient_event(low_salient_event, episode)

            high_event_idx = len(self.mdp.all_salient_events_ever) + 1
            high_salient_event = DCOSalientEvent(c_option, high_event_idx, is_low=False)
            reject_high = self.add_salient_event(high_salient_event, episode)

            if reject_low or reject_high:
                for _ in range(1000):
                    self.dsc_agent.agent_over_options.step(low_salient_event.target_state, self.dsc_agent.global_option.option_idx, 0, high_salient_event.target_state, 0, 1)
                    self.dsc_agent.agent_over_options.step(high_salient_event.target_state, self.dsc_agent.global_option.option_idx, 0, low_salient_event.target_state, 0, 1)

            if not reject_low or not reject_high or args.plot_rejected_events:
                plot_dco_salient_event_comparison(low_salient_event,
                                                  high_salient_event,
                                                  replay_buffer,
                                                  episode,
                                                  reject_low,
                                                  reject_high,
                                                  self.experiment_name)
        else:
            event_idx = len(self.mdp.all_salient_events_ever) + 1
            target_state = self.mdp.sample_random_state()[:2]
            low_salient_event = SalientEvent(target_state, event_idx)
            reject_low = self.add_salient_event(low_salient_event, episode)

            event_idx = len(self.mdp.all_salient_events_ever) + 1
            target_state = self.mdp.sample_random_state()[:2]
            high_salient_event = SalientEvent(target_state, event_idx)
            reject_high = self.add_salient_event(high_salient_event, episode)

        print(f"Generated {low_salient_event} and {high_salient_event}")

        self.last_event_creation_episode = episode
        self.last_event_rejection_episode = episode if reject_low and reject_high else -1

        self.most_recent_generated_salient_events = (
            low_salient_event if not reject_low else None,
            high_salient_event if not reject_high else None,
        )

    def add_salient_event(self, salient_event, episode):
        reject = self.should_reject_discovered_salient_event(salient_event)

        if reject:
            self.num_successive_rejections += 1
        else:
            print(f"Accepted {salient_event}")

            self.generated_salient_events.append(salient_event)
            self.mdp.add_new_target_event(salient_event)
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
        existing_target_events = self.mdp.get_all_target_events_ever() + [self.mdp.get_start_state_salient_event()]
        if any([event(salient_event.target_state) for event in existing_target_events]):
            return True

        if self.dsc_agent.state_in_any_completed_option(salient_event.target_state):
            return True

        return False

    def should_generate_new_salient_events(self, episode):
        if episode < 5:
            return False
        elif episode == 5:
            return True

        def _all_events_chained(low_event, high_event):
            chains = self.dsc_agent.chains
            chains_targeting_low_event =  [chain for chain in chains if chain.target_salient_event == low_event and
                                                                        chain.is_chain_completed(chains)]
            chains_targeting_high_event = [chain for chain in chains if chain.target_salient_event == high_event and
                                                                        chain.is_chain_completed(chains)]
            return (
                not (low_event is None and high_event is None) and              # most_recent_events were not both rejected AND
                (len(chains_targeting_low_event) > 0 or low_event is None) and  # low_event is chained to or was rejected AND
                (len(chains_targeting_high_event) > 0 or high_event is None)    # high_event is chained to or was rejected
            )

        most_recent_events = self.most_recent_generated_salient_events  # type: Tuple[DCOSalientEvent, DCOSalientEvent]

        return (
            _all_events_chained(*most_recent_events) or
            episode - self.last_event_creation_episode >= self.salient_event_freq or
            (episode - self.last_event_rejection_episode >= self.event_after_reject_freq and self.last_event_rejection_episode != -1)
        )

    def generate_candidate_salient_events(self):  # TODO: This needs to happen multiple times, not just once
        if self.should_set_off_learning_backward_options() and self.allow_backward_options:
            self.set_off_learning_backward_options()

            return self.mdp.get_all_target_events_ever() + [self.mdp.get_start_state_salient_event()]

        return self.mdp.get_all_target_events_ever()

    def are_all_original_salient_events_forward_chained(self):
        original_salient_events = self.mdp.get_original_target_events()
        for event in original_salient_events:  # type: SalientEvent
            forward_chains = [chain for chain in self.dsc_agent.chains if chain.target_salient_event == event and not chain.is_backward_chain]
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
        return self.num_successive_rejections >= 20 \
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

        if not self.planning_agent.use_her:
            self.dsc_agent.global_option.solver.step(state.features(), action, reward, next_state.features(), done)
            self.dsc_agent.agent_over_options.step(state.features(), self.dsc_agent.global_option.option_idx, reward, next_state.features(), done, 1)

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
    parser.add_argument("--create_backward_options", action="store_true", default=False)
    parser.add_argument("--learn_backward_options_offline", action="store_true", default=False)
    parser.add_argument("--use_warmup_phase", action="store_true", default=False)
    parser.add_argument("--update_global_solver", action="store_true", default=False)
    parser.add_argument("--salient_event_freq", type=int, help="Create a salient event every salient_event_freq episodes", default=50)
    parser.add_argument("--event_after_reject_freq", type=int, help="Create a salient event only event_after_reject_freq episodes after a double rejection", default=10)
    parser.add_argument("--use_hard_coded_events", action="store_true", help="Whether to use hard-coded salient events", default=False)
    parser.add_argument("--dco_use_xy_prior", action="store_true", default=False)
    parser.add_argument("--plot_rejected_events", action="store_true", default=False)
    parser.add_argument("--use_dco", action="store_true", default=False)
    parser.add_argument("--use_ucb", action="store_true", default=False)
    parser.add_argument("--threshold", type=int, help="Threshold determining size of termination set", default=0.1)
    parser.add_argument("--use_smdp_replay_buffer", action="store_true", help="Whether to use a replay buffer that has options", default=False)
    parser.add_argument("--allow_backward_options", action="store_true", default=False)
    parser.add_argument("--use_her", action="store_true", default=False)
    parser.add_argument("--use_her_locally", action="store_true", help="HER for local options", default=False)
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
                                    render=args.render,
                                    use_hard_coded_events=args.use_hard_coded_events)
        state_dim = overall_mdp.state_space_size()
        action_dim = overall_mdp.action_space_size()
    elif args.env == "d4rl-ant-maze":
        from simple_rl.tasks.d4rl_ant_maze.D4RLAntMazeMDPClass import D4RLAntMazeMDP
        overall_mdp = D4RLAntMazeMDP(maze_size="medium", seed=args.seed, render=args.render)
        state_dim = overall_mdp.state_space_size()
        action_dim = overall_mdp.action_space_size()
    elif args.env == "d4rl-medium-point-maze":
        from simple_rl.tasks.d4rl_point_maze.D4RLPointMazeMDPClass import D4RLPointMazeMDP
        overall_mdp = D4RLPointMazeMDP(seed=args.seed,
                                       render=args.render,
                                       use_hard_coded_events=args.use_hard_coded_events,
                                       difficulty="medium")
        state_dim = overall_mdp.state_space_size()
        action_dim = overall_mdp.action_space_size()
    elif args.env == "d4rl-hard-point-maze":
        from simple_rl.tasks.d4rl_point_maze.D4RLPointMazeMDPClass import D4RLPointMazeMDP
        overall_mdp = D4RLPointMazeMDP(seed=args.seed,
                                       render=args.render,
                                       use_hard_coded_events=args.use_hard_coded_events,
                                       difficulty="hard")
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
                            create_backward_options=args.create_backward_options,
                            learn_backward_options_offline=args.learn_backward_options_offline,
                            dense_reward=args.dense_reward,
                            update_global_solver=args.update_global_solver,
                            use_warmup_phase=args.use_warmup_phase,
                            experiment_name=args.experiment_name,
                            use_her=args.use_her,
                            use_her_locally=args.use_her_locally)

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
                                    use_hard_coded_events=args.use_hard_coded_events,
                                    use_dco=args.use_dco,
                                    dco_use_xy_prior=args.dco_use_xy_prior,
                                    allow_backward_options=args.allow_backward_options,
                                    experiment_name=args.experiment_name,
                                    seed=args.seed,
                                    threshold=args.threshold,
                                    use_smdp_replay_buffer=args.use_smdp_replay_buffer)
    num_successes = dsg_agent.dsg_run_loop(episodes=args.episodes, num_steps=args.steps)
