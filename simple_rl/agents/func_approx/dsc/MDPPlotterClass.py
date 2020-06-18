import os
import abc
import pickle
import sys
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import ipdb


class MDPPlotter(metaclass=abc.ABCMeta):
    def __init__(self, task_name, experiment_name, subdirectories):
        """
        Args:
            task_name (str): The name of the current task, so we know where to save plots
            experiment_name (str): The name of the current experiment, so we know where to save plots
            subdirectories (List[str]): List of subdirectories to make where plots will be saved
        """
        self.path = rotate_file_name(os.path.join("plots", task_name, experiment_name))
        subdirectories.append("event_graphs")
        for subdirectory in subdirectories:
            self._create_log_dir(os.path.join(self.path, subdirectory))

        self.kGraphIterationNumber = 0

    @abc.abstractmethod
    def generate_episode_plots(self, chainer, episode):
        """
        Args:
            chainer (SkillChainingAgent): the skill chaining agent we want to plot
            episode (int): the current episode
        """
        # initiation set
        # value function
        # low level shaped rewards
        # high level shaped rewards
        pass

    def generate_experiment_plots(self, chainer, pretrained, scores, durations):
        """
        Args:
            durations:
            scores:
            pretrained:
            chainer (SkillChainingAgent): the skill chaining agent we want to plot
        """
        print("\rSaving training and validation scores..")

        base_file_name = f"_pretrained_{pretrained}_seed_{chainer.seed}.pkl"
        training_scores_file_name = os.path.join(self.path, "training_scores", base_file_name)
        training_durations_file_name = os.path.join(self.path, "training_durations", base_file_name)
        validation_scores_file_name = os.path.join(self.path, "validation_scores", base_file_name)
        num_option_history_file_name = os.path.join(self.path, "num_options_per_episode", base_file_name)

        with open(training_scores_file_name, "wb+") as _f:
            pickle.dump(scores, _f)
        with open(training_durations_file_name, "wb+") as _f:
            pickle.dump(durations, _f)
        with open(validation_scores_file_name, "wb+") as _f:
            pickle.dump(chainer.validation_scores, _f)
        with open(num_option_history_file_name, "wb+") as _f:
            pickle.dump(chainer.num_options_history, _f)

    @staticmethod
    def _get_qvalues(solver, replay_buffer):
        CHUNK_SIZE = 1000
        replay_buffer = replay_buffer if replay_buffer is not None else solver.replay_buffer
        states = np.array([exp[0] for exp in replay_buffer])
        actions = np.array([exp[1] for exp in replay_buffer])

        # Chunk up the inputs so as to conserve GPU memory
        num_chunks = int(np.ceil(states.shape[0] / CHUNK_SIZE))
        state_chunks = np.array_split(states, num_chunks, axis=0)
        action_chunks = np.array_split(actions, num_chunks, axis=0)
        qvalues = np.zeros((states.shape[0],))
        current_idx = 0

        # get qvalues for taking action in each state
        for chunk_number, (state_chunk, action_chunk) in enumerate(zip(state_chunks, action_chunks)):
            state_chunk = torch.from_numpy(state_chunk).float().to(solver.device)
            action_chunk = torch.from_numpy(action_chunk).float().to(solver.device)
            chunk_qvalues = solver.get_qvalues(state_chunk, action_chunk).cpu().numpy().squeeze(1)
            current_chunk_size = len(state_chunk)
            qvalues[current_idx:current_idx + current_chunk_size] = chunk_qvalues
            current_idx += current_chunk_size

        return states, qvalues

    @staticmethod
    def _create_log_dir(directory_path):
        try:
            os.makedirs(directory_path)
        except OSError:
            print("Creation of the directory %s failed" % os.path.join(os.getcwd(), directory_path))
        else:
            print("Successfully created the directory %s " % os.path.join(os.getcwd(), directory_path))

    def visualize_graph(self, chains, plot_completed_events):
        def _completed(chain):
            return chain.is_chain_completed(chains) if plot_completed_events else True

        def _plot_event_pair(event1, event2):
            x = [event1.target_state[0], event2.target_state[0]]
            y = [event1.target_state[1], event2.target_state[1]]
            plt.plot(x, y, "o-", c="black")

        plt.figure()
        forward_chains = [chain for chain in chains if not chain.is_backward_chain and _completed(chain)]
        with sns.axes_style("white"):
            for chain in forward_chains:
                _plot_event_pair(chain.init_salient_event, chain.target_salient_event)

            plt.xticks([])
            plt.yticks([])

        file_name = f"event_graphs_episode_{self.kGraphIterationNumber}.png"
        plt.savefig(os.path.join(self.path, "event_graphs", file_name))
        plt.close()

        self.kGraphIterationNumber += 1

    def save_args(self):
        f = open(os.path.join(self.path, "run_command.txt"), 'w')
        f.write(' '.join(str(arg) for arg in sys.argv))
        f.close()


def rotate_file_name(file_path):
    if os.path.isdir(file_path):
        suffix = 1
        file_path += "_" + str(suffix)
        while os.path.isdir(file_path):
            suffix += 1
            file_path = file_path.rsplit("_", 1)[0] + "_" + str(suffix)
        return file_path
    else:
        return file_path
