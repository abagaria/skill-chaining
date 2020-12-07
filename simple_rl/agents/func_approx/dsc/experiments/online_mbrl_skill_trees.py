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
from simple_rl.agents.func_approx.dsc.experiments.utils import SkillTree


class OnlineModelBasedSkillTrees(object):
    def __init__(self, warmup_episodes, gestation_period, use_vf,
                 use_diverse_starts, use_dense_rewards, experiment_name, device):
        self.device = device
        self.use_vf = use_vf
        self.experiment_name = experiment_name
        self.warmup_episodes = warmup_episodes
        self.use_diverse_starts = use_diverse_starts
        self.use_dense_rewards = use_dense_rewards

        self.gestation_period = gestation_period

        self.mdp = D4RLAntMazeMDP("medium", goal_state=np.array((12, 12)))
        self.target_salient_event = self.mdp.get_original_target_events()[0]

        self.global_option = self.create_global_model_based_option()
        self.goal_option = self.create_model_based_option(name="goal-option", parent=None)

        self.skill_tree = SkillTree(options=[self.goal_option])
        self.new_options = [self.goal_option]
        self.mature_options = []

    def act(self, state):
        option_names = self.skill_tree.traverse()
        options = [self.skill_tree.get_option(name) for name in option_names]
        for option in options:
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
            self.reset(episode)

            step = self.dsc_rollout(num_steps) if episode > self.warmup_episodes else self.random_rollout(num_steps)

            last_10_durations.append(step)
            per_episode_durations.append(step)
            self.log_status(episode, last_10_durations)

            if episode >= self.warmup_episodes:
                self.learn_dynamics_model(episode)

        return per_episode_durations

    def learn_dynamics_model(self, episode):
        num_epochs = 50 if episode < 100 else 10
        self.global_option.solver.load_data()
        self.global_option.solver.train(epochs=num_epochs, batch_size=1024)
        for option in self.skill_tree.options:
            option.solver.model = self.global_option.solver.model

    def should_create_child_option(self, parent_option):
        assert isinstance(parent_option, ModelBasedOption)

        global_condition = not parent_option.pessimistic_is_init_true(self.mdp.init_state)
        local_condition = len(parent_option.children) < parent_option.max_num_children and \
                            parent_option.get_training_phase() == "initiation_done"

        return local_condition and global_condition

    def create_children_options(self, option):
        o1 = self.create_child_option(option)
        children = [o1]
        current_parent = option.parent
        while current_parent is not None:
            child_option = self.create_child_option(current_parent)
            children.append(child_option)
            current_parent = current_parent.parent
        return children

    def create_child_option(self, parent_option):
        assert isinstance(parent_option, ModelBasedOption)

        if not self.should_create_child_option(parent_option):
            return None

        prefix = "option_1" if parent_option.parent is None else parent_option.name
        name = prefix + f"_{len(parent_option.children)}"

        new_option = self.create_model_based_option(name, parent=parent_option)
        print(f"Creating {name}, parent {new_option.parent}, new_options = {self.new_options}, mature_options = {self.mature_options}")
        parent_option.children.append(new_option)

        return new_option

    def manage_chain_after_rollout(self, executed_option):

        if executed_option in self.new_options and executed_option.get_training_phase() != "gestation":
            self.new_options.remove(executed_option)
            self.mature_options.append(executed_option)

            if self.should_create_child_option(executed_option):
                new_options = self.create_children_options(executed_option)
                self.add_new_options(new_options)

    def add_new_options(self, new_options):
        for new_option in new_options:  # type: ModelBasedOption
            if new_option is not None:
                self.new_options.append(new_option)
                self.skill_tree.add_node(new_option)

    def log_status(self, episode, last_10_durations):
        print(f"Episode {episode} \t Mean Duration: {np.mean(last_10_durations)}")
        options = self.mature_options + self.new_options

        for option in self.mature_options:
            plot_two_class_classifier(option, episode, self.experiment_name, plot_examples=True)

        # for option in options:
        #     make_chunked_goal_conditioned_value_function_plot(option.value_learner,
        #                                                       goal=option.get_goal_for_rollout(),
        #                                                       episode=episode, seed=0,
        #                                                       experiment_name=self.experiment_name)

    def create_model_based_option(self, name, parent=None):
        option_idx = len(self.skill_tree.options) + 1 if parent is not None else 1
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
            # cond = lambda s: s[0] < 7.4 and s[1] < 2  # TODO: Hardcoded for bottom corridor
            random_state = self.mdp.sample_random_state()
            random_position = self.mdp.get_position(random_state)
            self.mdp.set_xy(random_position)

        print(f"[Episode {episode}] Reset state to {self.mdp.cur_state.position}")

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
    parser.add_argument("--episodes", type=int, default=150)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--save_option_data", action="store_true", default=False)
    parser.add_argument("--warmup_episodes", type=int, default=5)
    parser.add_argument("--use_value_function", action="store_true", default=False)
    parser.add_argument("--use_diverse_starts", action="store_true", default=False)
    parser.add_argument("--use_dense_rewards", action="store_true", default=False)
    args = parser.parse_args()

    exp = OnlineModelBasedSkillTrees(   gestation_period=args.gestation_period,
                                        experiment_name=args.experiment_name,
                                        device=torch.device(args.device),
                                        warmup_episodes=args.warmup_episodes,
                                        use_vf=args.use_value_function,
                                        use_diverse_starts=args.use_diverse_starts,
                                        use_dense_rewards=args.use_dense_rewards)

    create_log_dir(args.experiment_name)
    create_log_dir(f"initiation_set_plots/{args.experiment_name}")
    create_log_dir(f"value_function_plots/{args.experiment_name}")

    durations = exp.run_loop(args.episodes, args.steps)

    if args.save_option_data:
        exp.save_option_dynamics_data()
