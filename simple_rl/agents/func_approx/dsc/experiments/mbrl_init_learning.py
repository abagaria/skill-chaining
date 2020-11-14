import os
import ipdb
import torch
import argparse
import numpy as np
from copy import deepcopy
from collections import deque
from simple_rl.agents.func_approx.dsc.experiments.utils import *
from simple_rl.agents.func_approx.dsc.MBOptionClass import ModelBasedOption
from simple_rl.tasks.d4rl_ant_maze.D4RLAntMazeMDPClass import D4RLAntMazeMDP


class LearnInitClassifierForMBOption(object):
    def __init__(self, gestation_period, initiation_period, experiment_name, device):
        self.device = device
        self.experiment_name = experiment_name
        self.gestation_period = gestation_period
        self.initiation_period = initiation_period

        self.mdp = D4RLAntMazeMDP("umaze", goal_state=np.array((8, 0)))
        self.target_salient_event = self.mdp.get_original_target_events()[0]
        self.option = self.create_model_based_option()

    def act(self, state):
        return self.option

    def run_loop(self, num_episodes=1000):
        per_episode_durations = []
        last_10_durations = deque(maxlen=10)

        for episode in range(num_episodes):
            self.reset()
            state = deepcopy(self.mdp.cur_state)

            selected_option = self.act(state)
            transitions, reward = selected_option.rollout(step_number=0)

            duration = len(transitions)
            last_10_durations.append(duration)
            per_episode_durations.append(duration)

            self.log_status(episode, last_10_durations)

    def log_status(self, episode, last_10_durations):
        print(f"Episode {episode} \t Mean Duration: {np.mean(last_10_durations)}")
        plot_two_class_classifier(self.option, episode, self.experiment_name, plot_examples=True)

    def create_model_based_option(self):
        option = ModelBasedOption(parent=None, mdp=self.mdp,
                                  buffer_length=50, global_init=False,
                                  gestation_period=self.gestation_period,
                                  initiation_period=self.initiation_period,
                                  timeout=100, max_steps=1000, device=self.device,
                                  target_salient_event=self.target_salient_event,
                                  name="model-based-option",
                                  path_to_model="mpc-ant-umaze-model.pkl")
        return option

    def reset(self):
        """ Special reset function -- start the episode inside the option's initiation region. """

        start_state = None
        while start_state is None:
            s = self.mdp.sample_random_state()
            start_state = s if self.option.is_init_true(s) else None
        start_position = self.mdp.get_position(start_state)
        self.mdp.reset()
        self.mdp.set_xy(start_position)


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
    parser.add_argument("--initiation_period", type=float, default=np.inf)
    parser.add_argument("--episodes", type=int, default=150)
    args = parser.parse_args()

    exp = LearnInitClassifierForMBOption(gestation_period=args.gestation_period,
                                         initiation_period=args.initiation_period,
                                         experiment_name=args.experiment_name,
                                         device=torch.device(args.device))

    create_log_dir(args.experiment_name)
    create_log_dir(f"initiation_set_plots/{args.experiment_name}")
    exp.run_loop(args.episodes)
