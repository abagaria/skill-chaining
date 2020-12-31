import os
import ipdb
import time
import torch
import pickle
import argparse
import numpy as np
from copy import deepcopy
from functools import reduce
from collections import deque
from simple_rl.agents.func_approx.dsc.experiments.utils import *
from simple_rl.agents.func_approx.dsc.MFOptionClass import ModelFreeOption
from simple_rl.tasks.d4rl_ant_maze.D4RLAntMazeMDPClass import D4RLAntMazeMDP

class BaselineModelFreeDSC(object):
    def __init__(self, max_steps, gestation_period, initiation_period, buffer_length,
                 use_diverse_starts, use_dense_rewards, lr_c, lr_a,
                 maze_type, experiment_name, device,
                 logging_freq, generate_init_gif, evaluation_freq, seed):
        assert maze_type in ("umaze", "medium")

        self.lr_c = lr_c
        self.lr_a = lr_a

        self.device = device
        self.experiment_name = experiment_name
        self.max_steps = max_steps
        self.use_diverse_starts = use_diverse_starts
        self.use_dense_rewards = use_dense_rewards

        self.seed = seed
        self.logging_freq = logging_freq
        self.evaluation_freq = evaluation_freq
        self.generate_init_gif = generate_init_gif

        self.buffer_length = buffer_length
        self.gestation_period = gestation_period
        self.initiation_period = initiation_period

        goal_state = np.array((8, 0)) if maze_type == "umaze" else np.array((20, 20))
        self.mdp = D4RLAntMazeMDP(maze_type, goal_state=goal_state, seed=seed)
        self.target_salient_event = self.mdp.get_original_target_events()[0]

        self.global_option = self.create_global_option()
        self.goal_option = self.create_local_option('goal-option', parent=None)
        self.chain = [self.goal_option]
        self.new_options = [self.goal_option]
        self.mature_options = []

        self.log = {}

    @staticmethod
    def _pick_earliest_option(state, options):
        for option in options:
            if option.is_init_true(state) and not option.is_term_true(state):
                return option

    def act(self, state):
        for option in self.chain:
            if option.is_init_true(state) and not option.is_term_true(state):
                return option
        return self.global_option

    def dsc_rollout(self, num_steps):
        step_number = 0
        while step_number < num_steps and not self.mdp.cur_state.is_terminal():
            state = deepcopy(self.mdp.cur_state)
            selected_option = self.act(state)
            transitions, reward = selected_option.rollout(step_number=step_number)

            if len(transitions) == 0:
                break

            self.manage_chain_after_rollout(selected_option)
            step_number += len(transitions)
        return step_number

    def run_loop(self, num_episodes, num_steps, start_episode=0):
        per_episode_durations = []
        last_10_durations = deque(maxlen=10)

        for episode in range(start_episode, start_episode + num_episodes):
            self.reset()

            step = self.dsc_rollout(num_steps)

            last_10_durations.append(step)
            per_episode_durations.append(step)
            self.log_status(episode, last_10_durations)
            self.log_success_metrics(episode)

        return per_episode_durations

    def log_success_metrics(self, episode):
        individual_option_data = {option.name: option.get_option_success_rate() for option in self.chain}
        overall_success = reduce(lambda x,y: x*y, individual_option_data.values())
        self.log[episode] = {"individual_option_data": individual_option_data, "success_rate": overall_success}

        if episode % self.evaluation_freq == 0 and episode > 0:
            success, step_count = test_agent(self, 1, self.max_steps)

            self.log[episode]["success"] = success
            self.log[episode]["step-count"] = step_count[0]

            with open(f"{self.experiment_name}/log_file_{self.seed}.pkl", "wb+") as log_file:
                pickle.dump(self.log, log_file)

    def log_status(self, episode, last_10_durations):
        print(f"Episode {episode} \t Mean Duration: {np.mean(last_10_durations)}")

        if episode % self.logging_freq == 0 and episode != 0:
            options = self.mature_options + self.new_options

            for option in self.mature_options:
                episode_label = episode if self.generate_init_gif else -1
                plot_two_class_classifier(option, episode_label, self.experiment_name, plot_examples=True)

            for option in options:
                make_chunked_value_function_plot(option.solver, episode, self.seed, self.experiment_name)

    def manage_chain_after_rollout(self, executed_option):

        if executed_option in self.new_options and executed_option.get_training_phase() != "gestation":
            self.new_options.remove(executed_option)
            self.mature_options.append(executed_option)

        if self.should_create_new_option():
            name = f"option-{len(self.mature_options)}"
            new_option = self.create_local_option(name, parent=self.mature_options[-1])
            print(f"Creating {name}, parent {new_option.parent}, new_options = {self.new_options}, mature_options = {self.mature_options}")
            self.new_options.append(new_option)
            self.chain.append(new_option)

    def should_create_new_option(self):  # TODO: Cleanup
        if len(self.mature_options) > 0 and len(self.new_options) == 0:
            return self.mature_options[-1].get_training_phase() == "initiation_done" and \
                not self.contains_init_state()
        return False

    def contains_init_state(self):
        for option in self.mature_options:
            if option.is_init_true(np.array([0,0])):  # TODO: Get test-time start state automatically
                return True
        return False

    def reset(self):
        self.mdp.reset()

        if self.use_diverse_starts:
            cond = lambda s: s[0] < 7.4 and s[1] < 2  # TODO: Hardcoded for bottom corridor
            random_state = self.mdp.sample_random_state(cond=cond)
            random_position = self.mdp.get_position(random_state)
            self.mdp.set_xy(random_position)

    def create_local_option(self, name, parent=None):
        option_idx = len(self.chain) + 1 if parent is not None else 1
        option = ModelFreeOption(name=name,
                                 parent=parent,
                                 mdp=self.mdp,
                                 global_solver=self.global_option.solver,
                                 buffer_length=self.buffer_length,
                                 global_init=False,
                                 gestation_period=self.gestation_period,
                                 initiation_period=self.initiation_period,
                                 timeout=200,
                                 max_steps=self.max_steps,
                                 device=self.device,
                                 dense_reward=self.use_dense_rewards,
                                 option_idx=option_idx,
                                 lr_c=self.lr_c,
                                 lr_a=self.lr_a,
                                 target_salient_event=self.target_salient_event
                                 )
        return option

    def create_global_option(self):
        option = ModelFreeOption(name="global-option",
                                 parent=None,
                                 mdp=self.mdp,
                                 global_solver=None,
                                 buffer_length=self.buffer_length,
                                 global_init=True,
                                 gestation_period=self.gestation_period,
                                 initiation_period=self.initiation_period,
                                 timeout=1,
                                 max_steps=self.max_steps,
                                 device=self.device,
                                 dense_reward=self.use_dense_rewards,
                                 option_idx=0,
                                 lr_c=self.lr_c,
                                 lr_a=self.lr_a,
                                 target_salient_event=self.target_salient_event
                                 )
        return option


def test_agent(exp, num_experiments, num_steps):
    def rollout():
        step_number = 0
        while step_number < num_steps and not \
        exp.mdp.sparse_gc_reward_function(exp.mdp.cur_state, exp.mdp.goal_state, {})[1]:
            state = deepcopy(exp.mdp.cur_state)
            selected_option = exp.act(state)
            transitions, reward = selected_option.rollout(step_number=step_number, eval_mode=True)
            step_number += len(transitions)
        return step_number

    success = 0
    step_counts = []

    for _ in tqdm(range(num_experiments), desc="Performing test rollout"):
        exp.mdp.reset()
        steps_taken = rollout()
        if steps_taken != num_steps:
            success += 1
        step_counts.append(steps_taken)

    print("*" * 80)
    print(f"Test Rollout Success Rate: {success / num_experiments}, Duration: {np.mean(step_counts)}")
    print("*" * 80)

    return success / num_experiments, step_counts


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment_name", type=str, help="Experiment Name")
    parser.add_argument("--device", type=str, help="cpu/cuda:0/cuda:1")
    parser.add_argument("--seed", type=int, help="Random seed")
    parser.add_argument("--gestation_period", type=int, default=3)
    parser.add_argument("--initiation_period", type=int, default=np.inf)
    parser.add_argument("--buffer_length", type=int, default=50)
    parser.add_argument("--episodes", type=int, default=150)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--use_diverse_starts", action="store_true", default=False)
    parser.add_argument("--use_dense_rewards", action="store_true", default=False)
    parser.add_argument("--logging_frequency", type=int, default=50, help="Draw init sets, etc after every _ episodes")
    parser.add_argument("--generate_init_gif", action="store_true", default=False)
    parser.add_argument("--evaluation_frequency", type=int, default=10)
    parser.add_argument("--maze_type", type=str)
    parser.add_argument("--lr_c", type=float, help="critic learning rate", default=3e-4)
    parser.add_argument("--lr_a", type=float, help="actor learning rate", default=3e-4)
    args = parser.parse_args()

    exp = BaselineModelFreeDSC(max_steps=args.steps,
                               gestation_period=args.gestation_period,
                               initiation_period=args.initiation_period,
                               buffer_length=args.buffer_length,
                               use_diverse_starts=args.use_diverse_starts,
                               use_dense_rewards=args.use_dense_rewards,
                               lr_c=args.lr_c,
                               lr_a=args.lr_a,
                               maze_type=args.maze_type,
                               experiment_name=args.experiment_name,
                               device=torch.device(args.device),
                               logging_freq=args.logging_frequency,
                               generate_init_gif=args.generate_init_gif,
                               evaluation_freq=args.evaluation_frequency,
                               seed=args.seed
                               )

    create_log_dir(args.experiment_name)
    create_log_dir(f"initiation_set_plots/{args.experiment_name}")
    create_log_dir(f"value_function_plots/{args.experiment_name}")

    start_time = time.time()
    durations = exp.run_loop(args.episodes, args.steps)
    end_time = time.time()

    print("exporting graphs!")
    exp.log_status(2000, [1000])

    print("TIME: ", end_time - start_time)
