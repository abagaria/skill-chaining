import argparse
import torch
from copy import deepcopy
from collections import deque
import numpy as np
import pickle
import os
import ipdb

from simple_rl.agents.func_approx.dsc.CoveringOptionsAgentClass import CoveringOptions
from simple_rl.agents.func_approx.dqn.DQNAgentClass import DQNAgent
from simple_rl.mdp.StateClass import State
from simple_rl.agents.func_approx.dsc.OptionClass import Option


class DeepCoveringOptionsAgent(object):
    def __init__(self, mdp, max_steps, lr_actor, lr_critic, ddpg_batch_size, device, max_num_options=5,
                 subgoal_reward=0.,
                 buffer_length=20,
                 init_q=None,
                 generate_plots=False,
                 pretrain_option_policies=False,
                 update_global_solver=False,
                 dense_reward=False,
                 log_dir="",
                 seed=0,
                 tensor_log=False,
                 experiment_name=""):

        self.mdp = mdp
        self.original_actions = deepcopy(mdp.actions)
        self.max_steps = max_steps
        self.subgoal_reward = subgoal_reward
        self.init_q = init_q
        self.generate_plots = generate_plots
        self.log_dir = log_dir
        self.seed = seed
        self.device = torch.device(device)
        self.max_num_options = max_num_options
        self.dense_reward = dense_reward
        self.pretrain_option_policies = pretrain_option_policies
        self.update_global_solver = update_global_solver
        self.experiment_name = experiment_name
        self.global_option = Option(overall_mdp=self.mdp, name="global_option", global_solver=None,
                                    lr_actor=lr_actor, lr_critic=lr_critic, buffer_length=buffer_length,
                                    ddpg_batch_size=ddpg_batch_size,
                                    subgoal_reward=self.subgoal_reward,
                                    seed=self.seed,
                                    max_steps=self.max_steps,
                                    enable_timeout=True,
                                    generate_plots=self.generate_plots,
                                    writer=None,
                                    device=self.device,
                                    use_warmup_phase=False,
                                    update_global_solver=self.update_global_solver,
                                    dense_reward=self.dense_reward,
                                    is_backward_option=False,
                                    option_idx=0)
        self.trained_options = [self.global_option]

        self.agent_over_options = DQNAgent(self.mdp.state_space_size(), 1,
                                           trained_options=self.trained_options,
                                           seed=seed, lr=1e-4, name="GlobalDQN",
                                           eps_start=0.1, tensor_log=tensor_log,
                                           use_double_dqn=True, writer=None, device=self.device)

    def augment_agent_with_new_option(self, newly_trained_option, init_q_value):
        """
        Train the current untrained option and initialize a new one to target.
        Add the newly_trained_option as a new node to the Q-function over options
        Args:
            newly_trained_option (Option)
            init_q_value (float): if given use this, else compute init_q optimistically
        """
        # Add the trained option to the action set of the global solver
        if newly_trained_option not in self.trained_options:
            self.trained_options.append(newly_trained_option)
            self.global_option.solver.trained_options.append(newly_trained_option)

        # Augment the global DQN with the newly trained option
        num_actions = len(self.trained_options)
        new_global_agent = DQNAgent(self.agent_over_options.state_size, num_actions, self.trained_options,
                                    seed=self.seed, name=self.agent_over_options.name,
                                    eps_start=self.agent_over_options.epsilon,
                                    tensor_log=self.agent_over_options.tensor_log,
                                    use_double_dqn=self.agent_over_options.use_ddqn,
                                    lr=self.agent_over_options.learning_rate,
                                    writer=None,
                                    device=self.device)
        new_global_agent.replay_buffer = self.agent_over_options.replay_buffer

        init_q = init_q_value
        print("Initializing new option node with q value {}".format(init_q))
        new_global_agent.policy_network.initialize_with_smaller_network(self.agent_over_options.policy_network, init_q)
        new_global_agent.target_network.initialize_with_smaller_network(self.agent_over_options.target_network, init_q)

        self.agent_over_options = new_global_agent


    def discover_new_option(self):
        option_idx = len(self.trained_options)
        c_option = CoveringOptions(replay_buffer=self.agent_over_options.replay_buffer,
                                   obs_dim=self.mdp.state_space_size(),
                                   feature=None,
                                   threshold=0.1,
                                   beta=0.1,
                                   use_xy_prior=True,
                                   name=f"covering-option-{option_idx}",
                                   option_idx=option_idx,
                                   overall_mdp=self.mdp,
                                   global_solver=self.global_option.solver,
                                   lr_actor=self.global_option.lr_actor,
                                   lr_critic=self.global_option.lr_critic,
                                   ddpg_batch_size=self.global_option.ddpg_batch_size,
                                   subgoal_reward=self.subgoal_reward,
                                   seed=self.seed,
                                   max_steps=self.max_steps,
                                   device=self.device,
                                   dense_reward=self.dense_reward)
        self.augment_agent_with_new_option(c_option, 0.)

    def make_smdp_update(self, *, action, next_state, option_transitions):
        """
        Use Intra-Option Learning for sample efficient learning of the option-value function Q(s, o)
        Args:
            action (int): option taken by the global solver
            next_state (State): state we landed in after executing the option
            option_transitions (list): list of (s, a, r, s') tuples representing the trajectory during option execution
        """

        def get_reward(transitions):
            gamma = self.global_option.solver.gamma
            raw_rewards = [tt[2] for tt in transitions]
            return sum([(gamma ** idx) * rr for idx, rr in enumerate(raw_rewards)])

        selected_option = self.trained_options[action]  # type: Option

        for i, transition in enumerate(option_transitions):
            start_state = transition[0]
            if selected_option.is_init_true(start_state):
                sub_transitions = option_transitions[i:]
                option_reward = get_reward(sub_transitions)
                self.agent_over_options.step(start_state.features(), action, option_reward, next_state.features(),
                                             next_state.is_terminal(), num_steps=len(sub_transitions))

    def act(self, state, train_mode=True):
        # Query the global Q-function to determine which option to take in the current state
        option_idx = self.agent_over_options.act(state.features(), train_mode=train_mode)

        if train_mode:
            self.agent_over_options.update_epsilon()

        # Selected option
        selected_option = self.trained_options[option_idx]  # type: Option
        return selected_option

    def dco_rollout(self, *, episode_number, step_number):
        score = 0
        state = deepcopy(self.mdp.cur_state)
        while step_number < self.max_steps:
            selected_option = self.act(state)
            option_transitions, reward = selected_option.execute_option_in_mdp(self.mdp,
                                                                               episode_number,
                                                                               step_number)
            score += reward
            state = deepcopy(self.mdp.cur_state)
            step_number += len(option_transitions)

            self.make_smdp_update(action=selected_option.option_idx,
                                  next_state=state,
                                  option_transitions=option_transitions)

            if state.is_terminal():
                break
        return score, step_number

    def dco_run_loop(self, *, num_episodes):
        per_episode_scores = []
        per_episode_durations = []
        last_10_scores = deque(maxlen=10)
        last_10_durations = deque(maxlen=10)

        for episode in range(num_episodes):

            step_number = 0
            self.mdp.reset()

            if episode > 0 and episode % 30 == 0 and len(self.trained_options) < self.max_num_options:
                self.discover_new_option()

            score, step_number = self.dco_rollout(episode_number=episode,
                                                  step_number=step_number)

            last_10_scores.append(score)
            last_10_durations.append(step_number)
            per_episode_scores.append(score)
            per_episode_durations.append(step_number)

            print('\rEpisode {}\tAverage Score: {:.2f}\tDuration: {:.2f} steps\tOP Eps: {:.2f}'.format(
                episode, np.mean(last_10_scores), np.mean(last_10_durations), self.agent_over_options.epsilon)
            )

        return per_episode_scores, per_episode_durations


def test_single_goal(agent, mdp, episodes, steps, start_state, goal_position):
    success_rates = []
    mdp.set_current_goal(goal_position)

    is_goal_state = lambda s: np.linalg.norm(s.position - goal_position) <= 0.6

    for episode in range(episodes):
        step = 0
        success = 0.
        reset_to_state(mdp, start_state)
        state = deepcopy(mdp.cur_state)

        while step < steps:
            selected_option = agent.act(state)  # type: Option
            option_transitions, reward = selected_option.execute_option_in_mdp(mdp,
                                                                               episode,
                                                                               step)
            if len(option_transitions) == 0:
                ipdb.set_trace()

            state = deepcopy(mdp.cur_state)
            step += len(option_transitions)

            option_visited_states = [transition[-1] for transition in option_transitions]

            if any([is_goal_state(s) for s in option_visited_states]):
                success = 1.
                break

        success_rates.append(success)

        print("*" * 80)
        print(f"[{start_state} -> {goal_position}] Episode: {episode} \t Success: {success}")
        print("*" * 80)

    return success_rates


def test(agent, mdp, episodes, steps, num_runs):
    lc = []
    start_states = [mdp.sample_random_state() for _ in range(num_runs)]
    goal_states = [mdp.sample_random_state() for _ in range(num_runs)]

    for start_state, goal_state in zip(start_states, goal_states):
        reset_to_state(mdp, start_state)
        successes = test_single_goal(agent, mdp, episodes, steps, start_state, goal_state)
        lc.append(successes)

    return lc


def reset_to_state(mdp, start_state):
    """ Reset the MDP to either the default start state or to the specified one. """
    mdp.reset()

    if start_state is not None:
        start_position = start_state.position if isinstance(start_state, State) else start_state[:2]
        mdp.set_xy(start_position)


def create_log_dir(experiment_name):
    path = os.path.join(os.getcwd(), experiment_name)
    try:
        os.mkdir(path)
    except OSError:
        print("Creation of the directory %s failed" % path)
    else:
        print("Successfully created the directory %s " % path)
    return path

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment_name", type=str, help="Experiment Name")
    parser.add_argument("--dense_reward", type=bool, help="Whether to use dense/sparse rewards", default=False)
    parser.add_argument("--env", type=str, help="name of gym environment", default="point-env")
    parser.add_argument("--difficulty", type=str, help="Control suite env difficulty", default="easy")
    parser.add_argument("--render", type=bool, help="render environment training", default=False)
    parser.add_argument("--log", type=bool, help="enable tensorboard logging", default=False)
    parser.add_argument("--episodes", type=int, help="number of training episodes", default=200)
    parser.add_argument("--steps", type=int, help="number of steps per episode", default=200)
    parser.add_argument("--device", type=str, help="cuda/cpu", default="cpu")
    parser.add_argument("--seed", type=int, help="random seed", default=0)
    parser.add_argument("--goal_directed", action="store_true", default=False)
    args = parser.parse_args()

    log_dir = create_log_dir(args.experiment_name)
    create_log_dir("saved_runs")
    create_log_dir("value_function_plots")
    create_log_dir("initiation_set_plots")
    create_log_dir("value_function_plots/{}".format(args.experiment_name))
    create_log_dir("initiation_set_plots/{}".format(args.experiment_name))

    if args.env == "point-reacher":
        from simple_rl.tasks.point_reacher.PointReacherMDPClass import PointReacherMDP

        overall_mdp = PointReacherMDP(seed=args.seed,
                                      dense_reward=args.dense_reward,
                                      render=args.render,
                                      use_hard_coded_events=False,
                                      goal_directed=args.goal_directed,
                                      init_goal_pos=np.array((9, 9)))
        state_dim = overall_mdp.state_space_size()
        action_dim = overall_mdp.action_space_size()
    elif args.env == "d4rl-medium-point-maze":
        from simple_rl.tasks.d4rl_point_maze.D4RLPointMazeMDPClass import D4RLPointMazeMDP
        overall_mdp = D4RLPointMazeMDP(seed=args.seed,
                                       render=args.render,
                                       goal_directed=args.goal_directed,
                                       init_goal_pos=np.array((9, 9)),
                                       difficulty="medium")
        state_dim = overall_mdp.state_space_size()
        action_dim = overall_mdp.action_space_size()
    elif args.env == "d4rl-hard-point-maze":
        from simple_rl.tasks.d4rl_point_maze.D4RLPointMazeMDPClass import D4RLPointMazeMDP
        overall_mdp = D4RLPointMazeMDP(seed=args.seed,
                                       render=args.render,
                                       goal_directed=args.goal_directed,
                                       init_goal_pos=np.array((9, 9)),
                                       difficulty="hard")
        state_dim = overall_mdp.state_space_size()
        action_dim = overall_mdp.action_space_size()
    elif args.env == "d4rl-ant-maze":
        from simple_rl.tasks.d4rl_ant_maze.D4RLAntMazeMDPClass import D4RLAntMazeMDP
        overall_mdp = D4RLAntMazeMDP(maze_size="medium",
                                     seed=args.seed,
                                     goal_directed=args.goal_directed,
                                     init_goal_pos=np.array((0, 9)),
                                     render=args.render)
        state_dim = overall_mdp.state_space_size()
        action_dim = overall_mdp.action_space_size()
    else:
        from simple_rl.tasks.gym.GymMDPClass import GymMDP

        overall_mdp = GymMDP(args.env, render=args.render)
        state_dim = overall_mdp.env.observation_space.shape[0]
        action_dim = overall_mdp.env.action_space.shape[0]
        overall_mdp.env.seed(args.seed)

    dco_agent = DeepCoveringOptionsAgent(
        mdp=overall_mdp,
        max_steps=args.steps,
        lr_actor=1e-4,
        lr_critic=1e-3,
        ddpg_batch_size=64,
        device=args.device,
        max_num_options=5,
        pretrain_option_policies=False,
        dense_reward=args.dense_reward,
        log_dir="",
        seed=args.seed,
        tensor_log=False,
        experiment_name=args.experiment_name)

    print("{}: State dim: {}, Action dim: {}".format(overall_mdp.env_name, state_dim, action_dim))

    agent_name = overall_mdp.env_name + "_dco_agent"
    scores, durations = dco_agent.dco_run_loop(num_episodes=args.episodes)

    success_curve = test(dco_agent, overall_mdp, 60, args.steps, 10)
    with open(f"{args.experiment_name}/lc_randomized_start_states_True_scores.pkl", "wb+") as f:
        pickle.dump(success_curve, f)
