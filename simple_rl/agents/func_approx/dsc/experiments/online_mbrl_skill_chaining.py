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
from simple_rl.agents.func_approx.dsc.SubgoalSelectionClass import OptimalSubgoalSelector


class OnlineModelBasedSkillChaining(object):
    def __init__(self, warmup_episodes, gestation_period, use_vf,
                 use_diverse_starts, use_dense_rewards, use_optimal_sampler, experiment_name, device):
        self.device = device
        self.use_vf = use_vf
        self.experiment_name = experiment_name
        self.warmup_episodes = warmup_episodes
        self.use_diverse_starts = use_diverse_starts
        self.use_dense_rewards = use_dense_rewards
        self.use_optimal_sampler = use_optimal_sampler

        self.gestation_period = gestation_period

        self.mdp = D4RLAntMazeMDP("umaze", goal_state=np.array((8, 0)))
        self.target_salient_event = self.mdp.get_original_target_events()[0]

        self.global_option = self.create_global_model_based_option()
        self.goal_option = self.create_model_based_option(name="goal-option", parent=None)

        self.chain = [self.goal_option]
        self.new_options = [self.goal_option]
        self.mature_options = []

        self.optimal_subgoal_selector = None

    @staticmethod
    def _pick_earliest_option(state, options):
        for option in options:
            if option.is_init_true(state) and not option.is_term_true(state):
                return option

    def act(self, state):
        current_option = self._pick_earliest_option(state, self.chain)
        return current_option if current_option is not None else self.global_option

    def pick_optimal_subgoal(self, state):
        current_option = self._pick_earliest_option(state, self.mature_options)

        if current_option is not None and self.optimal_subgoal_selector is not None:
            sg = self.optimal_subgoal_selector.pick_subgoal(state, current_option)
            if sg is None: print(f"Did not find a subgoal for option {current_option}")
            return sg

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
            subgoal = self.pick_optimal_subgoal(state) if self.use_optimal_sampler else None
            selected_option = self.act(state)
            transitions, reward = selected_option.rollout(step_number=step_number, rollout_goal=subgoal)
            self.manage_chain_after_rollout(selected_option)

            step_number += len(transitions)
        return step_number

    def run_loop(self, num_episodes=300, num_steps=150):
        per_episode_durations = []
        last_10_durations = deque(maxlen=10)

        for episode in range(num_episodes):
            self.reset(episode)

            step = self.dsc_rollout(num_steps) if episode > self.warmup_episodes else self.random_rollout(num_steps)

            last_10_durations.append(step)
            per_episode_durations.append(step)
            self.log_status(episode, last_10_durations)

            if episode >= self.warmup_episodes:
                self.learn_dynamics_model(episode)

            if len(self.mature_options) > 0 and self.use_optimal_sampler:
                self.optimal_subgoal_selector = OptimalSubgoalSelector(self.mature_options,
                                                                       self.mdp.goal_state,
                                                                       self.mdp.state_space_size())

        return per_episode_durations

    def learn_dynamics_model(self, episode):
        num_epochs = 50 if episode < 100 else 10  # TODO
        self.global_option.solver.load_data()
        self.global_option.solver.train(epochs=10, batch_size=1024)
        for option in self.chain:
            option.solver.model = self.global_option.solver.model

    def should_create_new_option(self):  # TODO: Cleanup
        if len(self.mature_options) > 0 and len(self.new_options) == 0:
            return self.mature_options[-1].get_training_phase() == "initiation_done" and \
                not self.contains_init_state()
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

        if self.should_create_new_option():
            name = f"option-{len(self.mature_options)}"
            new_option = self.create_model_based_option(name, parent=self.mature_options[-1])
            print(f"Creating {name}, parent {new_option.parent}, new_options = {self.new_options}, mature_options = {self.mature_options}")
            self.new_options.append(new_option)
            self.chain.append(new_option)

    def log_status(self, episode, last_10_durations):
        print(f"Episode {episode} \t Mean Duration: {np.mean(last_10_durations)}")
        options = self.mature_options + self.new_options

        for option in self.mature_options:
            plot_two_class_classifier(option, episode, self.experiment_name, plot_examples=True)

        for option in options:
            make_chunked_goal_conditioned_value_function_plot(option.value_learner,
                                                              goal=option.get_goal_for_rollout(),
                                                              episode=episode, seed=0,
                                                              experiment_name=self.experiment_name)

    def create_model_based_option(self, name, parent=None):
        option_idx = len(self.chain) + 1 if parent is not None else 1
        option = ModelBasedOption(parent=parent, mdp=self.mdp,
                                  buffer_length=50, global_init=False,
                                  gestation_period=self.gestation_period,
                                  timeout=200, max_steps=1000, device=self.device,
                                  target_salient_event=self.target_salient_event,
                                  name=name,
                                  path_to_model="",
                                  global_solver=self.global_option.solver,
                                  use_vf=self.use_vf,
                                  dense_reward=self.use_dense_rewards,
                                  global_value_learner=self.global_option.value_learner,
                                  option_idx=option_idx)
        return option

    def create_global_model_based_option(self):  # TODO: what should the timeout be for this option?
        option = ModelBasedOption(parent=None, mdp=self.mdp,
                                  buffer_length=50, global_init=True,
                                  gestation_period=self.gestation_period,
                                  timeout=100, max_steps=1000, device=self.device,
                                  target_salient_event=self.target_salient_event,
                                  name="global-option",
                                  path_to_model="",
                                  global_solver=None,
                                  use_vf=self.use_vf,
                                  dense_reward=self.use_dense_rewards,
                                  global_value_learner=None,
                                  option_idx=0)
        return option

    def reset(self, episode):
        self.mdp.reset()

        if self.use_diverse_starts and episode > self.warmup_episodes:
            cond = lambda s: s[0] < 7.4 and s[1] < 2  # TODO: Hardcoded for bottom corridor
            random_state = self.mdp.sample_random_state(cond=cond)
            random_position = self.mdp.get_position(random_state)
            self.mdp.set_xy(random_position)


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
    parser.add_argument("--episodes", type=int, default=150)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--warmup_episodes", type=int, default=5)
    parser.add_argument("--use_value_function", action="store_true", default=False)
    parser.add_argument("--use_diverse_starts", action="store_true", default=False)
    parser.add_argument("--use_dense_rewards", action="store_true", default=False)
    parser.add_argument("--use_optimal_sampler", action="store_true", default=False)
    args = parser.parse_args()

    exp = OnlineModelBasedSkillChaining(gestation_period=args.gestation_period,
                                        experiment_name=args.experiment_name,
                                        device=torch.device(args.device),
                                        warmup_episodes=args.warmup_episodes,
                                        use_vf=args.use_value_function,
                                        use_diverse_starts=args.use_diverse_starts,
                                        use_dense_rewards=args.use_dense_rewards,
                                        use_optimal_sampler=args.use_optimal_sampler)

    create_log_dir(args.experiment_name)
    create_log_dir(f"initiation_set_plots/{args.experiment_name}")
    create_log_dir(f"value_function_plots/{args.experiment_name}")

    durations = exp.run_loop(args.episodes, args.steps)

    with open(f"{args.experiment_name}/durations.pkl", "wb+") as f:
        pickle.dump(durations, f)
