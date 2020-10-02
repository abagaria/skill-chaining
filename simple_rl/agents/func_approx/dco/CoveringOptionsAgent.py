import argparse
import torch
from copy import deepcopy
from collections import deque
import numpy as np
import pickle
import os
import ipdb
import itertools
import random

from simple_rl.agents.func_approx.dco.CoveringOptions import CoveringOptions
from simple_rl.agents.func_approx.dqn.DQNAgentClass import DQNAgent
from simple_rl.mdp.StateClass import State
from simple_rl.agents.func_approx.dsc.OptionClass import Option
from simple_rl.agents.func_approx.dsc.SalientEventClass import SalientEvent


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
        self.target_events = []

        self.agent_over_options = DQNAgent(self.mdp.state_space_size() + 2, 1,
                                           trained_options=self.trained_options,
                                           seed=seed, lr=1e-4, name="GlobalDQN",
                                           eps_start=0.1, tensor_log=tensor_log,
                                           use_double_dqn=True, writer=None, device=self.device)

        self.analyzer = CoveringOptions(2, device, True, True, loss_type="corrected")

    def should_reject_target_event(self, salient_event):
        return any([event(salient_event.target_state) for event in self.target_events])

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
        self.analyzer.train(self.global_option.solver.replay_buffer)
        max_state = self.analyzer.get_max_f_value_state()
        min_state = self.analyzer.get_min_f_value_state()

        min_event = SalientEvent(min_state, len(self.target_events))
        if not self.should_reject_target_event(min_event):
            self.target_events.append(min_event)
            self.create_option(min_event)

        max_event = SalientEvent(max_state, len(self.target_events))
        if not self.should_reject_target_event(max_event):
            self.target_events.append(max_event)
            self.create_option(max_event)

    def create_option(self, target_event):
        i = len(self.trained_options)
        goal_option = Option(overall_mdp=self.mdp, name=f'goal_option_{i}', global_solver=self.global_option.solver,
                             ddpg_batch_size=64, num_subgoal_hits_required=np.inf,
                             subgoal_reward=self.subgoal_reward, seed=self.seed, max_steps=self.max_steps,
                             enable_timeout=True,
                             device=self.device,
                             lr_actor=self.global_option.lr_actor,
                             lr_critic=self.global_option.lr_critic,
                             dense_reward=self.dense_reward, chain_id=i, max_num_children=0,
                             target_salient_event=target_event,
                             option_idx=i,
                             use_her=False)
        self.augment_agent_with_new_option(goal_option, 0)

    def get_augmented_state(self, state, goal):
        if goal is not None:
            return np.concatenate((state.features(), goal))
        return state.features()

    def perform_experience_replay(self, state, episodic_trajectory, option_trajectory, overall_goal=None):
        """
        Perform experience replay for the global-option and the policy over options.

        Args:
            state (State): state at the end of the trajectory
            episodic_trajectory (list): sequence of (s, a, r, s') tuples
            option_trajectory (list): sequence of (option, trajectory) tuples
            overall_goal (np.ndarray): Goal DSC was pursuing for that episode

        """
        # Experience replay for the global-option
        reached_goal = self.mdp.get_position(state.position)
        overall_goal = self.mdp.get_position(self.mdp.goal_state) if overall_goal is None else overall_goal
        episodic_trajectory = list(itertools.chain.from_iterable(episodic_trajectory))

        self.global_option_experience_replay(episodic_trajectory, goal_state=overall_goal)

        reached_overall_goal = self.mdp.sparse_gc_reward_function(reached_goal, overall_goal, {})[1]
        if len(episodic_trajectory) > 0 and not reached_overall_goal:
            self.global_option_experience_replay(episodic_trajectory, goal_state=reached_goal)
        if len(option_trajectory) > 1:
            self.make_goal_conditioned_smdp_update(goal=overall_goal, final_state=state,
                                                   option_trajectories=option_trajectory)

    def global_option_experience_replay(self, trajectory, goal_state):
        for state, action, _, next_state in trajectory:
            augmented_state = self.get_augmented_state(state, goal=goal_state)
            augmented_next_state = self.get_augmented_state(next_state, goal=goal_state)
            reward_func = self.mdp.dense_gc_reward_function if self.dense_reward else self.mdp.sparse_gc_reward_function
            reward, done = reward_func(next_state, goal_state, info={})
            self.global_option.solver.step(augmented_state, action, reward, augmented_next_state, done)

    def make_goal_conditioned_smdp_update(self, *, goal, final_state, option_trajectories):
        def get_goal_conditioned_reward(g, transitions):
            gamma = self.global_option.solver.gamma
            reward_func = self.mdp.dense_gc_reward_function if self.dense_reward else self.mdp.sparse_gc_reward_function
            raw_rewards = [reward_func(trans[-1], g, info={})[0] for trans in transitions]
            return sum([(gamma ** idx) * rr for idx, rr in enumerate(raw_rewards)])

        def perform_gc_experience_replay(g, o, transitions):
            if len(transitions) == 0: return

            reward_func = self.mdp.dense_gc_reward_function if self.dense_reward \
                else self.mdp.sparse_gc_reward_function

            s0 = transitions[0][0]
            option_final_state = transitions[-1][-1]
            s0_augmented = np.concatenate((s0.features(), g), axis=0)
            sp_augmented = np.concatenate((option_final_state.features(), g), axis=0)
            option_reward = get_goal_conditioned_reward(g, transitions)
            done = reward_func(option_final_state, g, info={})[1]
            self.agent_over_options.step(s0_augmented, o.option_idx, option_reward, sp_augmented,
                                         done=done, num_steps=len(transitions))

        goal_position = self.mdp.get_position(goal)
        reached_goal = self.mdp.get_position(final_state)

        for option, trajectory in option_trajectories:  # type: Option, list
            perform_gc_experience_replay(goal_position, option, transitions=trajectory)

        # Only perform the hindsight update if you didn't reach the goal
        if not self.mdp.sparse_gc_reward_function(reached_goal, goal_position, {})[1]:
            for option, trajectory in option_trajectories:  # type: Option, list
                perform_gc_experience_replay(reached_goal, option, transitions=trajectory)

    def act(self, state, goal, train_mode=True):
        # Query the global Q-function to determine which option to take in the current state
        option_idx = self.agent_over_options.act(self.get_augmented_state(state, goal), train_mode=train_mode)

        if train_mode:
            self.agent_over_options.update_epsilon()

        # Selected option
        selected_option = self.trained_options[option_idx]  # type: Option
        return selected_option

    def dco_rollout(self, *, goal, episode_number, step_number):
        score = 0
        option_trajectory = []
        episodic_trajectory = []
        state = deepcopy(self.mdp.cur_state)
        goal_interrupt = lambda x: self.mdp.dense_gc_reward_function(x, goal, {})[1]

        while step_number < self.max_steps:
            selected_option = self.act(state, goal)
            option_transitions, reward = selected_option.execute_option_in_mdp(self.mdp,
                                                                               episode_number,
                                                                               step_number,
                                                                               poo_goal=goal,
                                                                               goal_salient_event=goal_interrupt)
            score += reward
            state = deepcopy(self.mdp.cur_state)
            step_number += len(option_transitions)
            episodic_trajectory.append(option_transitions)
            option_trajectory.append((selected_option, option_transitions))

            if state.is_terminal() or self.mdp.dense_gc_reward_function(state, goal, {})[1]:
                break

        self.perform_experience_replay(state, episodic_trajectory, option_trajectory, overall_goal=goal)

        return score, step_number

    def dco_run_loop(self, *, num_episodes):
        per_episode_scores = []
        per_episode_durations = []
        last_10_scores = deque(maxlen=10)
        last_10_durations = deque(maxlen=10)

        for episode in range(num_episodes):

            step_number = 0
            self.mdp.reset()
            goal = self.mdp.sample_random_state()[:2] if len(self.target_events) == 0 \
                else random.sample(self.target_events, k=1)[0].get_target_position()

            if (episode > 0 and episode % 50 == 0) or episode == 5:
                self.discover_new_option()

            score, step_number = self.dco_rollout(episode_number=episode,
                                                  step_number=step_number,
                                                  goal=goal)

            last_10_scores.append(score)
            last_10_durations.append(step_number)
            per_episode_scores.append(score)
            per_episode_durations.append(step_number)

            print('\rEpisode {}\tAverage Score: {:.2f}\tDuration: {:.2f} steps\tOP Eps: {:.2f}\tGoal: {}'.format(
                episode, np.mean(last_10_scores), np.mean(last_10_durations), self.agent_over_options.epsilon, goal)
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
            selected_option = agent.act(state, goal=goal_position)  # type: Option
            option_transitions, reward = selected_option.execute_option_in_mdp(mdp,
                                                                               episode,
                                                                               step,
                                                                               poo_goal=goal_position,
                                                                               goal_salient_event=is_goal_state)
            if len(option_transitions) == 0:
                ipdb.set_trace()

            state = deepcopy(mdp.cur_state)
            step += len(option_transitions)

            option_visited_states = [transition[-1] for transition in option_transitions]

            if any([is_goal_state(s) for s in option_visited_states]):
                success = 1.
                break

        success_rates.append(success)

        print("*"* 80)
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
                                      )
        state_dim = overall_mdp.state_space_size()
        action_dim = overall_mdp.action_space_size()
    elif args.env == "d4rl-medium-point-maze":
        from simple_rl.tasks.d4rl_point_maze.D4RLPointMazeMDPClass import D4RLPointMazeMDP
        overall_mdp = D4RLPointMazeMDP(seed=args.seed,
                                       render=args.render,
                                       goal_directed=args.goal_directed,
                                       difficulty="medium")
        state_dim = overall_mdp.state_space_size()
        action_dim = overall_mdp.action_space_size()
    elif args.env == "d4rl-hard-point-maze":
        from simple_rl.tasks.d4rl_point_maze.D4RLPointMazeMDPClass import D4RLPointMazeMDP
        overall_mdp = D4RLPointMazeMDP(seed=args.seed,
                                       render=args.render,
                                       goal_directed=args.goal_directed,
                                       difficulty="hard")
        state_dim = overall_mdp.state_space_size()
        action_dim = overall_mdp.action_space_size()
    elif args.env == "d4rl-ant-maze":
        from simple_rl.tasks.d4rl_ant_maze.D4RLAntMazeMDPClass import D4RLAntMazeMDP
        overall_mdp = D4RLAntMazeMDP(maze_size="medium",
                                     seed=args.seed,
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
