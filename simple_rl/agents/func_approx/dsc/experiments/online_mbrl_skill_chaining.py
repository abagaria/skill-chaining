import os
import ipdb
import torch
import pickle
import argparse
import numpy as np
from copy import deepcopy
from collections import deque
from simple_rl.agents.func_approx.dsc.experiments.utils import *
from simple_rl.agents.func_approx.dsc.MBOptionClass import ModelBasedOption
from simple_rl.tasks.d4rl_ant_maze.D4RLAntMazeMDPClass import D4RLAntMazeMDP


class OnlineModelBasedSkillChaining(object):
    def __init__(self, warmup_episodes, gestation_period, initiation_period, use_vf,
                 use_diverse_starts, experiment_name, device):
        self.device = device
        self.use_vf = use_vf
        self.experiment_name = experiment_name
        self.warmup_episodes = warmup_episodes
        self.use_diverse_starts = use_diverse_starts

        self.gestation_period = gestation_period
        self.initiation_period = initiation_period

        self.mdp = D4RLAntMazeMDP("umaze", goal_state=np.array((8, 0)))
        self.target_salient_event = self.mdp.get_original_target_events()[0]

        self.global_option = self.create_global_model_based_option()
        self.goal_option = self.create_model_based_option(name="goal-option", parent=None)

        self.chain = [self.goal_option]
        self.new_options = [self.goal_option]
        self.mature_options = []

    def act(self, state):
        for option in self.chain:
            if option.is_init_true(state) and not option.is_term_true(state):
                return option
        return self.global_option

    def random_rollout(self, num_steps):
        step_number = 0
        while step_number < num_steps and not self.mdp.cur_state.is_terminal():
            state = deepcopy(self.mdp.cur_state)
            action = self.mdp.sample_random_action()
            reward, next_state = self.mdp.execute_agent_action(action)
            self.global_option.update_model(state, action, reward, next_state)
            step_number += 1
        return step_number

    def dsc_rollout(self, num_steps):
        step_number = 0
        while step_number < num_steps and not self.mdp.cur_state.is_terminal():
            state = deepcopy(self.mdp.cur_state)
            selected_option = self.act(state)

            transitions, reward = selected_option.rollout(step_number=step_number)
            self.manage_chain_after_rollout(selected_option)

            step_number += len(transitions)
        return step_number

    def run_loop(self, num_episodes=300, num_steps=150):
        per_episode_durations = []
        last_10_durations = deque(maxlen=10)

        for episode in range(num_episodes):
            self.reset()

            step = self.dsc_rollout(num_steps) if episode > self.warmup_episodes else self.random_rollout(num_steps)

            last_10_durations.append(step)
            per_episode_durations.append(step)
            self.log_status(episode, last_10_durations)

            if episode >= self.warmup_episodes:
                self.learn_dynamics_model()

        return per_episode_durations

    def learn_dynamics_model(self):
        self.global_option.solver.load_data()
        self.global_option.solver.train(epochs=50, batch_size=1024)
        for option in self.chain:
            option.solver.model = self.global_option.solver.model

    def should_create_new_option(self):  # TODO: Cleanup
        if len(self.mature_options) > 0 and len(self.new_options) == 0:
            return self.mature_options[-1].get_training_phase() == "initiation_done" and \
                not self.mature_options[-1].pessimistic_is_init_true(self.mdp.init_state)
        return False

    def manage_chain_after_rollout(self, executed_option):

        if executed_option in self.new_options and executed_option.get_training_phase() != "gestation":
            self.new_options.remove(executed_option)
            self.mature_options.append(executed_option)

        if self.should_create_new_option():
            name = f"option-{len(self.mature_options)}"
            new_option = self.create_model_based_option(name, parent=self.mature_options[-1])
            print(f"Creating {name}, parent {new_option.parent}, new_options = {self.new_options}, mature_options = {self.mature_options}")
            self.new_options.append(new_option)
            self.chain.append(new_option)

    def log_status(self, episode, last_10_durations):
        print(f"Episode {episode} \t Mean Duration: {np.mean(last_10_durations)}")

        for option in self.mature_options:
            plot_two_class_classifier(option, episode, self.experiment_name, plot_examples=True)
            make_chunked_goal_conditioned_value_function_plot(option.value_learner,
                                                              goal=option.get_goal_for_rollout(),
                                                              episode=episode, seed=0,
                                                              experiment_name=self.experiment_name)
            # should_change = option.should_change_negative_examples()
            # print(f"{option.name} should-change: {sum(should_change)}/{len(should_change)}")

    def create_model_based_option(self, name, parent=None):
        option = ModelBasedOption(parent=parent, mdp=self.mdp,
                                  buffer_length=50, global_init=False,
                                  gestation_period=self.gestation_period,
                                  initiation_period=self.initiation_period,
                                  timeout=200, max_steps=1000, device=self.device,
                                  target_salient_event=self.target_salient_event,
                                  name=name,
                                  path_to_model="",
                                  global_solver=self.global_option.solver,
                                  use_vf=self.use_vf)
        return option

    def create_global_model_based_option(self):  # TODO: what should the timeout be for this option?
        option = ModelBasedOption(parent=None, mdp=self.mdp,
                                  buffer_length=50, global_init=True,
                                  gestation_period=self.gestation_period,
                                  initiation_period=self.initiation_period,
                                  timeout=100, max_steps=1000, device=self.device,
                                  target_salient_event=self.target_salient_event,
                                  name="global-option",
                                  path_to_model="",
                                  global_solver=None,
                                  use_vf=self.use_vf)
        return option

    def reset(self):
        self.mdp.reset()

        if self.use_diverse_starts:
            random_state = self.mdp.sample_random_state()
            random_position = self.mdp.get_position(random_state)
            self.mdp.set_xy(random_position)

    def save_option_dynamics_data(self):
        """ Save data that can be used to learn a model of the option dynamics. """
        for option in self.mature_options:
            states = np.array([pair[0] for pair in option.in_out_pairs])
            next_states = np.array([pair[1] for pair in option.in_out_pairs])
            with open(f"{self.experiment_name}/{option.name}-dynamics-data.pkl", "wb+") as f:
                pickle.dump((states, next_states), f)


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
    parser.add_argument("--device", type=str, help="cpu/cuda:0/cuda:1")
    parser.add_argument("--gestation_period", type=int, default=3)
    parser.add_argument("--initiation_period", type=float, default=3)
    parser.add_argument("--episodes", type=int, default=150)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--save_option_data", action="store_true", default=False)
    parser.add_argument("--warmup_episodes", type=int, default=5)
    parser.add_argument("--use_value_function", action="store_true", default=False)
    parser.add_argument("--use_diverse_starts", action="store_true", default=False)
    args = parser.parse_args()

    exp = OnlineModelBasedSkillChaining(gestation_period=args.gestation_period,
                                        initiation_period=args.initiation_period,
                                        experiment_name=args.experiment_name,
                                        device=torch.device(args.device),
                                        warmup_episodes=args.warmup_episodes,
                                        use_vf=args.use_value_function,
                                        use_diverse_starts=args.use_diverse_starts)

    create_log_dir(args.experiment_name)
    create_log_dir(f"initiation_set_plots/{args.experiment_name}")
    create_log_dir(f"value_function_plots/{args.experiment_name}")

    durations = exp.run_loop(args.episodes, args.steps)

    if args.save_option_data:
        exp.save_option_dynamics_data()
