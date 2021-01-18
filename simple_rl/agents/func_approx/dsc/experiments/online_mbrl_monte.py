import os
import ipdb
import time
import torch
import random
import pickle
import argparse
import itertools
import numpy as np
from copy import deepcopy
from functools import reduce
from collections import deque
from simple_rl.tasks.gym.GymMDPClass import GymMDP
from simple_rl.agents.func_approx.dsc.experiments.utils import *
from simple_rl.agents.func_approx.dsc.MBOptionClassMonte import ModelBasedOption

class OnlineModelBasedSkillChaining(object):
    def __init__(self, mdp, max_steps, gestation_period, buffer_length, clear_replay_buffer,
                 diverse_starts, experiment_name, device, evaluation_freq, seed, preprocessing,
                 logging_frequency, svm_gamma, freeze_init_sets):

        self.device = device
        self.experiment_name = experiment_name
        self.max_steps = max_steps
        self.clear_replay_buffer = clear_replay_buffer
        self.diverse_starts = diverse_starts
        self.svm_gamma = svm_gamma
        self.freeze_init_sets = freeze_init_sets

        self.seed = seed
        self.evaluation_freq = evaluation_freq
        self.logging_frequency = logging_frequency
        self.preprocessing = preprocessing

        self.buffer_length = buffer_length
        self.gestation_period = gestation_period

        self.mdp = mdp

        self.global_option = self.create_global_model_based_option()
        self.goal_option = self.create_model_based_option(name="goal-option", parent=None)

        self.chain = [self.goal_option]
        self.new_options = [self.goal_option]
        self.mature_options = []

        self.optimal_subgoal_selector = None

        self.log = {}

    def act(self, state):
        for option in self.chain:
            if option.is_init_true(state) and not option.is_term_true(state):
                return option
        return self.global_option

    def collect_data(self):
        def random_rollout(num_steps):
            step_number = 0
            self.mdp.reset()
            while step_number < num_steps and not self.mdp.cur_state.is_terminal():
                state = deepcopy(self.mdp.cur_state)      
                history.append(state)
                action = self.mdp.sample_random_action()
                self.mdp.execute_agent_action(action)
                step_number += 1
        
        history = deque(maxlen=10000)
        for _ in range(1000):
            random_rollout(100)     

        return history  

    def collect_full_data(self):
        def helper(num_steps):
            step_number = 0
            while step_number < num_steps and not self.mdp.cur_state.is_terminal():
                state = deepcopy(self.mdp.cur_state)   
                history.append(state)         
                selected_option = self.act(state)
                transitions, reward = selected_option.rollout(step_number=step_number, eval_mode=True)
                if len(transitions) == 0:
                    break
                step_number += len(transitions)
            return step_number
        
        history = deque(maxlen=10000)
        for i in range(100):
            print(f"Episode {i}")
            self.reset(i)
            helper(1000)
        return history

    def dsc_rollout(self, num_steps):
        step_number = 0
        
        while step_number < num_steps and not self.mdp.cur_state.is_terminal():
            state = deepcopy(self.mdp.cur_state)            
            selected_option = self.act(state)

            transitions, reward = selected_option.rollout(step_number=step_number)

            if len(transitions) == 0:
                break

            step_number += len(transitions)
            
            self.manage_chain_after_rollout(selected_option)

        return step_number

    def run_loop(self, num_episodes, num_steps, start_episode=0):
        per_episode_durations = []
        last_10_durations = deque(maxlen=10)

        for episode in range(start_episode, start_episode + num_episodes):
            self.reset(episode)

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

        if episode % self.evaluation_freq == 0 and episode != 0:
            success, step_count, traj = test_agent(self, 1, self.max_steps)

            self.log[episode]["success"] = success
            self.log[episode]["step-count"] = step_count[0]
            self.log[episode]["traj"] = traj

            with open(f"{self.experiment_name}/log_file_{self.seed}.pkl", "wb+") as log_file:
                pickle.dump(self.log, log_file)

    def is_chain_complete(self):
        return all([option.get_training_phase() == "initiation_done" for option in self.chain]) and self.contains_init_state()

    def should_create_new_option(self):
        if len(self.mature_options) > 0 and len(self.new_options) == 0:
            return self.mature_options[-1].get_training_phase() == "initiation_done" and not self.contains_init_state()
        return False

    def contains_init_state(self):
        for option in self.mature_options:
            if option.is_init_true(self.mdp.init_state):
                return True
        return False

    def manage_chain_after_rollout(self, executed_option):
        if executed_option in self.new_options and executed_option.get_training_phase() != "gestation":
            self.new_options.remove(executed_option)
            self.mature_options.append(executed_option)

            if self.clear_replay_buffer:
                executed_option.solver.replay_buffer.clear()
                print(f"{executed_option.name} replay buffer has been cleared!")
        
        if self.clear_replay_buffer:
            pass

        if self.should_create_new_option():
            name = f"option-{len(self.mature_options)}"
            new_option = self.create_model_based_option(name, parent=self.mature_options[-1])
            print(f"Creating {name}, parent {new_option.parent}, new_options = {self.new_options}, mature_options = {self.mature_options}")
            self.new_options.append(new_option)
            self.chain.append(new_option)

    def filter_replay_buffer(self, option):
        assert isinstance(option, ModelBasedOption)
        print(f"Clearing the replay buffer for {option.name}")
        option.solver.replay_buffer.clear()

    def log_status(self, episode, last_10_durations):
        print(f"Episode {episode} \t Mean Duration: {np.mean(last_10_durations)}")
        if episode % self.logging_frequency == 0 and episode != 0:
            """Collect data"""
            data = []
            for option in self.chain:
                data += option.positive_examples
                data += option.negative_examples
            buffer = list(itertools.chain.from_iterable(data))

            """Plot value function"""
            for option in self.chain:
                make_chunked_value_function_plot(option.solver, episode, self.seed, self.experiment_name, buffer)

            """Plot init sets"""
            random.shuffle(buffer)
            for option in self.mature_options:
                if self.preprocessing == "position":
                    plot_two_class_classifier(option, episode, self.experiment_name)
                else: 
                    monte_plot_option_init_set(self, option, buffer)

    def create_model_based_option(self, name, parent=None):
        option_idx = len(self.chain) + 1 if parent is not None else 1
        option = ModelBasedOption(parent=parent, mdp=self.mdp,
                                  buffer_length=self.buffer_length,
                                  global_init=False,
                                  gestation_period=self.gestation_period,
                                  timeout=200, max_steps=self.max_steps, device=self.device,
                                  name=name,
                                  global_value_learner=self.global_option.solver,
                                  option_idx=option_idx,
                                  preprocessing=self.preprocessing,
                                  gamma=self.svm_gamma,
                                  freeze_init_sets=self.freeze_init_sets)
        return option

    def create_global_model_based_option(self):  # TODO: what should the timeout be for this option?
        option = ModelBasedOption(parent=None, mdp=self.mdp,
                                  buffer_length=self.buffer_length,
                                  global_init=True,
                                  gestation_period=self.gestation_period,
                                  timeout=200, max_steps=self.max_steps, device=self.device,
                                  name="global-option",
                                  global_value_learner=None,
                                  option_idx=0,
                                  preprocessing=self.preprocessing,
                                  gamma=self.svm_gamma,
                                  freeze_init_sets=self.freeze_init_sets)
        return option

    def reset(self, episode):
        self.mdp.reset()

        if self.diverse_starts:
            x, y = random.choice(self.mdp.spawn_states)
            self.mdp.set_player_position(x, y)


def create_log_dir(experiment_name):
    path = os.path.join(os.getcwd(), experiment_name)
    try:
        os.mkdir(path)
    except OSError:
        print("Creation of the directory %s failed" % path)
    else:
        print("Successfully created the directory %s " % path)
    return path

def test_agent(exp, num_experiments, num_steps):
    def rollout():
        step_number = 0
            
        while step_number < num_steps and not exp.mdp.cur_state.is_terminal():
            state = deepcopy(exp.mdp.cur_state)
            selected_option = exp.act(state)
            transitions, _ = selected_option.rollout(step_number=step_number, eval_mode=True)
            if len(transitions) == 0:
                break
            traj.append((selected_option.name, transitions))
            step_number += len(transitions)
        return step_number, exp.mdp.is_goal_state(exp.mdp.cur_state)

    success = 0
    traj = []
    step_counts = []

    for _ in tqdm(range(num_experiments), desc="Performing test rollout"):
        exp.mdp.reset()
        steps_taken, succeeded = rollout()
        if succeeded:
            success += 1
        step_counts.append(steps_taken)

    print("*" * 80)
    print(f"Test Rollout Success Rate: {success / num_experiments}, Duration: {np.mean(step_counts)}")
    print("*" * 80)

    return success / num_experiments, step_counts, traj

def get_trajectory(exp, num_steps):
    exp.mdp.reset()
    
    traj = []
    step_number = 0
    
    while step_number < num_steps and not exp.mdp.cur_state.is_terminal():
        state = deepcopy(exp.mdp.cur_state)
        selected_option = exp.act(state)
        transitions, reward = selected_option.rollout(step_number=step_number, eval_mode=True)
        if len(transitions) == 0:
            break
        step_number += len(transitions)
        traj.append((selected_option.name, transitions))
    return traj, step_number

def filter_traj(trajectory):
    traj = []
    for option, transitions in trajectory:
        for state, action, reward, next_state in transitions:
            traj.append(next_state[-1,:,:])
    return traj

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment_name", type=str, help="Experiment Name")
    parser.add_argument("--device", type=str, help="cpu/cuda:0/cuda:1")
    parser.add_argument("--seed", type=int, help="Random seed")
    parser.add_argument("--gestation_period", type=int, default=3)
    parser.add_argument("--buffer_length", type=int, default=50)
    parser.add_argument("--episodes", type=int, default=150)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--evaluation_frequency", type=int, default=10)
    parser.add_argument("--logging_frequency", type=int, default=1000)
    parser.add_argument("--clear_replay_buffer", action="store_true", default=False)
    parser.add_argument("--diverse_starts", action="store_true", default=False)
    parser.add_argument("--freeze_init_sets", action="store_true", default=False)
    parser.add_argument("--preprocessing", type=str, help="go-explore/position")
    parser.add_argument("--svm_gamma", type=str, help="auto/scale")

    args = parser.parse_args()

    assert args.svm_gamma == "auto" or args.svm_gamma == "scale"

    mdp = GymMDP(env_name="MontezumaRevenge-v0", pixel_observation=True, seed=args.seed)

    exp = OnlineModelBasedSkillChaining(mdp=mdp,
                                        gestation_period=args.gestation_period,
                                        experiment_name=args.experiment_name,
                                        device=torch.device(args.device),
                                        max_steps=args.steps,
                                        evaluation_freq=args.evaluation_frequency,
                                        logging_frequency=args.logging_frequency,
                                        buffer_length=args.buffer_length,
                                        seed=args.seed,
                                        clear_replay_buffer=args.clear_replay_buffer,
                                        diverse_starts=args.diverse_starts,
                                        preprocessing=args.preprocessing,
                                        svm_gamma=args.svm_gamma,
                                        freeze_init_sets=args.freeze_init_sets)

    create_log_dir(args.experiment_name)
    create_log_dir(f"initiation_set_plots/{args.experiment_name}")
    create_log_dir(f"value_function_plots/{args.experiment_name}")

    start_time = time.time()
    durations = exp.run_loop(args.episodes, args.steps)
    history = exp.collect_data() # TODO remove after experimentation with init sets
    end_time = time.time()

    print("TIME: ", end_time - start_time)
