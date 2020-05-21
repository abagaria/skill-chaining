import ipdb
import argparse
import random
from copy import deepcopy
from simple_rl.agents.func_approx.dsc.SalientEventClass import SalientEvent
from simple_rl.agents.func_approx.dsc.SkillChainingAgentClass import SkillChaining
from simple_rl.agents.func_approx.dsc.SkillGraphPlanningAgentClass import SkillGraphPlanningAgent
from simple_rl.agents.func_approx.dsc.utils import *
from simple_rl.mdp import State
from simple_rl.mdp.GoalDirectedMDPClass import GoalDirectedMDP


class DeepSkillGraphAgent(object):
    def __init__(self, mdp, dsc_agent, planning_agent, experiment_name, seed):
        """
        This agent will interleave planning with the `planning_agent` and chaining with
        the `dsc_agent`.
        Args:
            mdp (GoalDirectedMDP)
            dsc_agent (SkillChaining)
            planning_agent (SkillGraphPlanningAgent)
            experiment_name (str)
            seed (int)
        """
        self.mdp = mdp
        self.dsc_agent = dsc_agent
        self.planning_agent = planning_agent
        self.experiment_name = experiment_name
        self.seed = seed

        self.known_options = []

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

        while target_event is None and num_tries < 100:
            target_event = random.sample(candidate_salient_events, k=1)[0]

            # If you are already at the target_event, then re-sample
            if target_event(state):
                target_event = None

            num_tries += 1


        print(f"Deep skill graphs target event: {target_event}")
        return target_event

    def dsg_run_loop(self, episodes, num_steps):
        successes = []

        for episode in range(episodes):
            step_number = 0
            self.mdp.reset()
            print(f"[DeepSkillGraphAgentClass] Episode {episode}: Resetting MDP to start state")
            state = deepcopy(self.mdp.cur_state)

            while step_number < num_steps:
                goal_salient_event = self.select_goal_salient_event(state)

                step_number, success = self.planning_agent.planning_run_loop(start_episode=episode,
                                                                             goal_state=goal_salient_event.target_state,
                                                                             goal_salient_event=goal_salient_event,
                                                                             step_number=step_number,
                                                                             to_reset=False)

                state = deepcopy(self.mdp.cur_state)

                if success:
                    print(f"[DeepSkillGraphAgentClass] successfully reached {goal_salient_event}")

                successes.append(success)

        return successes

    def generate_candidate_salient_events(self):
        # TODO: This doesn't work yet because you also have to add the forward options back to the list `untrained_options`
        if self.are_all_original_salient_events_forward_chained():
            self.dsc_agent.create_backward_options = True
            self.dsc_agent.learn_backward_options_offline = True
            return self.mdp.get_all_target_events_ever() + [self.mdp.get_start_state_salient_event()]

        self.dsc_agent.create_backward_options = False
        self.dsc_agent.learn_backward_options_offline = False
        return self.mdp.get_all_target_events_ever()

    def are_all_original_salient_events_forward_chained(self):
        original_salient_events = self.mdp.get_original_target_events()
        for event in original_salient_events:  # type: SalientEvent
            forward_chains = [chain for chain in self.dsc_agent.chains if chain.target_salient_event == event and not chain.is_backward_chain]
            if any([not chain.is_chain_completed(self.dsc_agent.chains) for chain in forward_chains]):
                return False
        return True


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
    args = parser.parse_args()

    if args.env == "point-reacher":
        from simple_rl.tasks.point_reacher.PointReacherMDPClass import PointReacherMDP

        overall_mdp = PointReacherMDP(seed=args.seed, dense_reward=args.dense_reward, render=args.render)
        state_dim = 6
        action_dim = 2
    elif args.env == "d4rl-ant-maze":
        from simple_rl.tasks.d4rl_ant_maze.D4RLAntMazeMDPClass import D4RLAntMazeMDP
        overall_mdp = D4RLAntMazeMDP(maze_size="medium", seed=args.seed, render=args.render)
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
    else:
        from simple_rl.tasks.gym.GymMDPClass import GymMDP

        overall_mdp = GymMDP(args.env, render=args.render)
        state_dim = overall_mdp.env.observation_space.shape[0]
        action_dim = overall_mdp.env.action_space.shape[0]
        overall_mdp.env.seed(args.seed)

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
                            experiment_name=args.experiment_name)

    assert any([args.use_start_state_salience, args.use_option_intersection_salience, args.use_event_intersection_salience])
    # assert args.use_option_intersection_salience ^ args.use_event_intersection_salience

    planner = SkillGraphPlanningAgent(mdp=overall_mdp, chainer=chainer, experiment_name=args.experiment_name, seed=args.seed, initialize_graph=False)
    dsg_agent = DeepSkillGraphAgent(mdp=overall_mdp, dsc_agent=chainer, planning_agent=planner, experiment_name=args.experiment_name, seed=args.seed)
    num_successes = dsg_agent.dsg_run_loop(episodes=args.episodes, num_steps=args.steps)
