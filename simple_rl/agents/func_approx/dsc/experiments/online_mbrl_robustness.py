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
from simple_rl.agents.func_approx.dsc.MBOptionClass import ModelBasedOption
from simple_rl.tasks.d4rl_ant_maze.D4RLAntMazeMDPClass import D4RLAntMazeMDP
from simple_rl.agents.func_approx.dsc.SubgoalSelectionClass import OptimalSubgoalSelector


class OnlineModelBasedSkillChaining(object):
    def __init__(self, warmup_episodes, max_steps, gestation_period, buffer_length, use_vf, use_global_vf, use_model,
                 use_diverse_starts, use_dense_rewards, use_optimal_sampler, lr_c, lr_a, clear_option_buffers,
                 experiment_name, device, logging_freq, generate_init_gif, evaluation_freq, seed):

        self.lr_c = lr_c
        self.lr_a = lr_a

        self.device = device
        self.use_vf = use_vf
        self.use_global_vf = use_global_vf
        self.use_model = use_model
        self.experiment_name = experiment_name
        self.warmup_episodes = warmup_episodes
        self.max_steps = max_steps
        self.use_diverse_starts = use_diverse_starts
        self.use_dense_rewards = use_dense_rewards
        self.use_optimal_sampler = use_optimal_sampler
        self.clear_option_buffers = clear_option_buffers

        self.seed = seed
        self.logging_freq = logging_freq
        self.evaluation_freq = evaluation_freq
        self.generate_init_gif = generate_init_gif

        self.buffer_length = buffer_length
        self.gestation_period = gestation_period

        self.mdp = D4RLAntMazeMDP("umaze", goal_state=np.array((8,0)), seed=seed)
        self.target_salient_event = self.mdp.get_original_target_events()[0]

        self.global_option = self.create_global_model_based_option()
        self.goal_option = self.create_model_based_option(name="goal-option", parent=None)

        self.chain = [self.goal_option]
        self.new_options = [self.goal_option]
        self.mature_options = []

        self.optimal_subgoal_selector = None

        self.freeze_chain = False
        self.log = {}

    @staticmethod
    def _pick_earliest_option(state, options):
        for option in options:
            if option.is_init_true(state) and not option.is_term_true(state):
                return option

    def act(self, state):
        for option in self.chain:
            if option.is_init_true(state):
                subgoal = self.pick_optimal_subgoal(state, option) if self.use_optimal_sampler else None
                if subgoal is None:  # can be none if use_optimal_sampler is False or if pick_optimal_subgoal returned none
                    subgoal = option.get_goal_for_rollout()
                if not option.is_at_local_goal(state, subgoal):
                    return option, subgoal
        return self.global_option, self.global_option.get_goal_for_rollout()

    def pick_optimal_subgoal(self, state, current_option=None):

        if current_option is None:
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

            selected_option, subgoal = self.act(state)
            transitions, reward = selected_option.rollout(step_number=step_number, rollout_goal=subgoal)

            if len(transitions) == 0:
                break

            if not self.freeze_chain:
                self.manage_chain_after_rollout(selected_option)
            
            step_number += len(transitions)
        return step_number

    def run_loop(self, num_episodes, num_steps, start_episode=0):
        per_episode_durations = []
        last_10_durations = deque(maxlen=10)

        for episode in range(start_episode, start_episode + num_episodes):
            self.reset(episode)

            step = self.dsc_rollout(num_steps) if episode > self.warmup_episodes else self.random_rollout(num_steps)

            last_10_durations.append(step)
            per_episode_durations.append(step)
            self.log_status(episode, last_10_durations)

            if episode == self.warmup_episodes - 1 and self.use_model:
                self.learn_dynamics_model(epochs=50)
            elif episode >= self.warmup_episodes and self.use_model:
                self.learn_dynamics_model(epochs=5)

            if len(self.mature_options) > 0 and self.use_optimal_sampler:
                condition = lambda o: len(o.positive_examples) > 0 and len(o.effect_set) > 0
                options = [option for option in self.chain if condition(option)]
                self.optimal_subgoal_selector = OptimalSubgoalSelector(options,
                                                                       self.mdp.goal_state,
                                                                       self.mdp.state_space_size())

            self.log_success_metrics(episode)

        return per_episode_durations

    def save_log(self):
        with open(f"{self.experiment_name}/log_file_{self.seed}.pkl", "wb+") as log_file:
            pickle.dump(self.log, log_file)

    def log_success_metrics(self, episode):
        individual_option_data = {option.name: option.get_option_success_rate() for option in self.chain}
        overall_success = reduce(lambda x,y: x*y, individual_option_data.values())
        self.log[episode] = {}
        self.log[episode]["individual-option-rates"] = individual_option_data
        self.log[episode]["overall-success-rate"] = overall_success
        self.log[episode]["option_stats"] = [f"{option.name}: {option.num_goal_hits}/{option.num_executions}" for option in self.chain]

        if episode % self.evaluation_freq == 0 and episode > self.warmup_episodes:
            self.save_log()

    def learn_dynamics_model(self, epochs=50, batch_size=1024):
        self.global_option.solver.load_data()
        self.global_option.solver.train(epochs=epochs, batch_size=batch_size)
        for option in self.chain:
            option.solver.model = self.global_option.solver.model

    def is_chain_complete(self):
        return all([option.get_training_phase() == "initiation_done" for option in self.chain]) and self.mature_options[-1].is_init_true(np.array([0,0]))

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

    def manage_chain_after_rollout(self, executed_option):

        if executed_option in self.new_options and executed_option.get_training_phase() != "gestation":
            self.new_options.remove(executed_option)
            self.mature_options.append(executed_option)

            if self.clear_option_buffers:
                self.filter_replay_buffer(executed_option)

        if executed_option.num_goal_hits == 2 * executed_option.gestation_period and self.clear_option_buffers:
            self.filter_replay_buffer(executed_option)

        if self.should_create_new_option():
            name = f"option-{len(self.mature_options)}"
            new_option = self.create_model_based_option(name, parent=self.mature_options[-1])
            print(f"Creating {name}, parent {new_option.parent}, new_options = {self.new_options}, mature_options = {self.mature_options}")
            self.new_options.append(new_option)
            self.chain.append(new_option)

    def filter_replay_buffer(self, option):
        assert isinstance(option, ModelBasedOption)
        print(f"Clearing the replay buffer for {option.name}")
        option.value_learner.replay_buffer.clear()

    def log_status(self, episode, last_10_durations):
        print(f"Episode {episode} \t Mean Duration: {np.mean(last_10_durations)}")

        if episode % self.logging_freq == 0 and episode != 0:
            options = self.mature_options + self.new_options

            for option in self.mature_options:
                episode_label = episode if self.generate_init_gif else -1
                plot_two_class_classifier(option, episode_label, self.experiment_name, plot_examples=True, seed=self.seed)

            for option in options:
                if self.use_global_vf:
                    make_chunked_goal_conditioned_value_function_plot(option.global_value_learner,
                                                                    goal=option.get_goal_for_rollout(),
                                                                    episode=episode, seed=self.seed,
                                                                    experiment_name=self.experiment_name,
                                                                    option_idx=option.option_idx)
                else:
                    make_chunked_goal_conditioned_value_function_plot(option.value_learner,
                                                                    goal=option.get_goal_for_rollout(),
                                                                    episode=episode, seed=self.seed,
                                                                    experiment_name=self.experiment_name)

    def create_model_based_option(self, name, parent=None):
        option_idx = len(self.chain) + 1 if parent is not None else 1
        option = ModelBasedOption(parent=parent, mdp=self.mdp,
                                  buffer_length=self.buffer_length,
                                  global_init=False,
                                  gestation_period=self.gestation_period,
                                  timeout=200, max_steps=self.max_steps, device=self.device,
                                  target_salient_event=self.target_salient_event,
                                  name=name,
                                  path_to_model="",
                                  global_solver=self.global_option.solver,
                                  use_vf=self.use_vf,
                                  use_global_vf=self.use_global_vf,
                                  use_model=self.use_model,
                                  dense_reward=self.use_dense_rewards,
                                  global_value_learner=self.global_option.value_learner,
                                  option_idx=option_idx,
                                  lr_c=self.lr_c, lr_a=self.lr_a)
        return option

    def create_global_model_based_option(self):  # TODO: what should the timeout be for this option?
        option = ModelBasedOption(parent=None, mdp=self.mdp,
                                  buffer_length=self.buffer_length,
                                  global_init=True,
                                  gestation_period=self.gestation_period,
                                  timeout=200, max_steps=self.max_steps, device=self.device,
                                  target_salient_event=self.target_salient_event,
                                  name="global-option",
                                  path_to_model="",
                                  global_solver=None,
                                  use_vf=self.use_vf,
                                  use_global_vf=self.use_global_vf,
                                  use_model=self.use_model,
                                  dense_reward=self.use_dense_rewards,
                                  global_value_learner=None,
                                  option_idx=0,
                                  lr_c=self.lr_c, lr_a=self.lr_a)
        return option

    def reset(self, episode):
        self.mdp.reset()
        self.mdp.set_xy((0,0))

        if not self.is_chain_complete():
            if self.use_diverse_starts and episode > self.warmup_episodes:
                cond = lambda s: s[0] < 7.4 and s[1] < 2  # Hardcoded for bottom corridor
                random_state = self.mdp.sample_random_state(cond=cond)
                random_position = self.mdp.get_position(random_state)
                self.mdp.set_xy(random_position)
        elif not self.freeze_chain:
            # sets up variables to ensure no new options will ever be added
            # and that the last option has the start state always in its init set
            self.freeze_chain = True
            self.mature_options[-1].is_last_option = True
            self.log["chain-freeze-episode"] = episode


def create_log_dir(experiment_name):
    path = os.path.join(os.getcwd(), experiment_name)
    try:
        os.mkdir(path)
    except OSError:
        print("Creation of the directory %s failed" % path)
    else:
        print("Successfully created the directory %s " % path)
    return path

def plot_success_curve(agent, filepath):
    last_epoch = max(list(exp.log.keys()))
    num_options = len(exp.log[last_epoch]["individual-option-rates"])
    x = []
    y = []
    for i in range(0, last_epoch + 1):
        if len(exp.log[i]["individual-option-rates"]) == num_options:
            x.append(i)
            y.append(exp.log[i]["overall-success-rate"])

    import matplotlib.pyplot as plt
    plt.figure()
    plt.plot(x,y)
    plt.savefig(filepath)

def plot_success_curve2(file, filepath):
    last_epoch = 1000
    log = pickle.load(open(file, "rb"))
    num_options = len(log[last_epoch]["individual-option-rates"])
    x = []
    y = []
    for i in range(0, last_epoch + 1):
        if len(log[i]["individual-option-rates"]) == num_options:
            x.append(i)
            y.append(log[i]["overall-success-rate"])
    plt.figure()
    plt.plot(x,y)
    plt.savefig(filepath)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment_name", type=str, help="Experiment Name")
    parser.add_argument("--device", type=str, help="cpu/cuda:0/cuda:1")
    parser.add_argument("--seed", type=int, help="Random seed")
    parser.add_argument("--gestation_period", type=int, default=3)
    parser.add_argument("--buffer_length", type=int, default=50)
    parser.add_argument("--episodes", type=int, default=150)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--warmup_episodes", type=int, default=5)
    parser.add_argument("--use_value_function", action="store_true", default=False)
    parser.add_argument("--use_global_value_function", action="store_true", default=False)
    parser.add_argument("--use_model", action="store_true", default=False)
    parser.add_argument("--use_diverse_starts", action="store_true", default=False)
    parser.add_argument("--use_dense_rewards", action="store_true", default=False)
    parser.add_argument("--use_optimal_sampler", action="store_true", default=False)
    parser.add_argument("--logging_frequency", type=int, default=50, help="Draw init sets, etc after every _ episodes")
    parser.add_argument("--generate_init_gif", action="store_true", default=False)
    parser.add_argument("--evaluation_frequency", type=int, default=10)

    parser.add_argument("--clear_option_buffers", action="store_true", default=False)
    parser.add_argument("--lr_c", type=float, help="critic learning rate")
    parser.add_argument("--lr_a", type=float, help="actor learning rate")
    args = parser.parse_args()

    assert args.use_model or args.use_value_function

    if not args.use_value_function:
        assert not args.use_global_value_function

    if args.clear_option_buffers:
        assert not args.use_global_value_function

    exp = OnlineModelBasedSkillChaining(gestation_period=args.gestation_period,
                                        experiment_name=args.experiment_name,
                                        device=torch.device(args.device),
                                        warmup_episodes=args.warmup_episodes,
                                        max_steps=args.steps,
                                        use_model=args.use_model,
                                        use_vf=args.use_value_function,
                                        use_global_vf=args.use_global_value_function,
                                        use_diverse_starts=args.use_diverse_starts,
                                        use_dense_rewards=args.use_dense_rewards,
                                        use_optimal_sampler=args.use_optimal_sampler,
                                        logging_freq=args.logging_frequency,
                                        evaluation_freq=args.evaluation_frequency,
                                        buffer_length=args.buffer_length,
                                        generate_init_gif=args.generate_init_gif,
                                        seed=args.seed,
                                        lr_c=args.lr_c,
                                        lr_a=args.lr_a,
                                        clear_option_buffers=args.clear_option_buffers)

    create_log_dir(args.experiment_name)
    create_log_dir(f"initiation_set_plots/{args.experiment_name}")
    create_log_dir(f"value_function_plots/{args.experiment_name}")

    start_time = time.time()
    durations = exp.run_loop(args.episodes, args.steps)
    end_time = time.time()

    print("TIME: ", end_time - start_time)

    exp.save_log()
    plot_success_curve(exp, f"{args.experiment_name}_{args.seed}_success_curve")

    # with open(f"{args.experiment_name}/training_durations.pkl", "wb+") as f:
    #     pickle.dump(durations, f)
