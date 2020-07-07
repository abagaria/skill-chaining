import csv
import os
import abc
import pickle
import sys
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import seaborn as sns
import torch
import ipdb


class MDPPlotter(metaclass=abc.ABCMeta):
    def __init__(self, task_name, experiment_name, subdirectories, mdp):
        """
        Args:
            task_name (str): The name of the current task, so we know where to save plots
            experiment_name (str): The name of the current experiment, so we know where to save plots
            subdirectories (List[str]): List of subdirectories to make where plots will be saved
        """
        self.path = rotate_file_name(os.path.join("plots", task_name, experiment_name))
        subdirectories.extend(["final_results", "event_graphs"])
        for subdirectory in subdirectories:
            self._create_log_dir(os.path.join(self.path, subdirectory))
        self.save_args()
        self.kGraphIterationNumber = 0
        self.mdp = mdp

    @abc.abstractmethod
    def generate_episode_plots(self, dsc_agent, episode):
        """
        Args:
            dsc_agent (SkillChainingAgent): the skill chaining agent we want to plot
            episode (int): the current episode
        """
        # initiation set
        # value function
        # low level shaped rewards
        # high level shaped rewards
        pass

    def generate_final_experiment_plots(self, dsg_agent, episode):
        """
        Args:
            dsg_agent (DeepSkillGraphAgent): the skill chaining agent we want to plot
            episode (int): the current episode
        """
        self.save_option_success_rate(dsg_agent.dsc_agent)
        self.plot_learning_curve(dsg_agent, episode)

    def plot_learning_curve(self, dsg_agent, episode):
        learning_curves = self.learning_curve(dsg_agent, episode, 1, True)
        ipdb.set_trace()

    def learning_curve(self, dsc_agent, episodes, episode_interval, randomize_start_states=False):
        start_states = self.generate_start_states(num_states=20)
        goal_salient_events = self.generate_goal_salient_events(num_states=20)
        all_runs = []
        for start_state, goal_salient_event in zip(start_states, goal_salient_events):
            start_state = start_state if randomize_start_states else None
            single_run = self.success_curve(dsc_agent, start_state, goal_salient_event, episodes, episode_interval)
            all_runs.append(single_run)
        return all_runs

    @abc.abstractmethod
    def generate_start_states(self, num_states):
        pass

    @abc.abstractmethod
    def generate_goal_salient_events(self, num_states):
        pass

    @staticmethod
    def success_curve(dsg_agent, start_state, goal_salient_event, episodes, episode_interval):
        success_rates_over_time = []
        for episode in range(episodes):
            success_rate = dsg_agent.planning_agent.measure_success(goal_salient_event=goal_salient_event,
                                                                    start_state=start_state,
                                                                    starting_episode=episode,
                                                                    num_episodes=episode_interval)

            print("*" * 80)
            print(f"[{goal_salient_event}]: Episode {episode}: Success Rate: {success_rate}")
            print("*" * 80)

            success_rates_over_time.append(success_rate)
        return success_rates_over_time

    def generate_random_states(self, num_states):
        generated_states = []
        for i in range(num_states):
            state = self.mdp.sample_random_state()
            generated_states.append(state)
        return generated_states

    def save_option_success_rate(self, dsc_agent):
        def write_options_csv():
            fields = ['option', 'salient event', 'num_goal_hits', 'num_executions', 'success_rate']
            rows = [(o.name,
                     o.target_salient_event.name if o.target_salient_event is not None else None,
                     o.num_goal_hits,
                     o.num_executions,
                     o.get_option_success_rate()) for o in dsc_agent.trained_options]
            with open(os.path.join(self.path, "final_results", "option_results.csv"), "w") as csv_file:
                csv_writer = csv.writer(csv_file)
                csv_writer.writerow(fields)
                csv_writer.writerows(rows)

        def write_chains_csv():
            fields = ['chain', 'is_completed']
            rows = [(str(chain), chain.is_chain_completed(dsc_agent.chains)) for chain in dsc_agent.chains]
            with open(os.path.join(self.path, "final_results", "chain_results.csv"), "w") as csv_file:
                csv_writer = csv.writer(csv_file)
                csv_writer.writerow(fields)
                csv_writer.writerows(rows)

        write_options_csv()
        write_chains_csv()

    def save_pickled_data(self, dsc_agent, pretrained, scores, durations):
        """
        Args:
            durations:
            scores:
            pretrained:
            dsc_agent (SkillChainingAgent): the skill chaining agent we want to plot
        """
        print("\rSaving training and validation scores.")

        base_file_name = f"_pretrained_{pretrained}_seed_{dsc_agent.seed}.pkl"
        training_scores_file_name = os.path.join(self.path, "final_results", "training_scores" + base_file_name)
        training_durations_file_name = os.path.join(self.path, "final_results", "training_durations" + base_file_name)
        validation_scores_file_name = os.path.join(self.path, "final_results", "validation_scores" + base_file_name)
        num_option_history_file_name = os.path.join(self.path, "final_results", "num_options_per_episode" + base_file_name)

        with open(training_scores_file_name, "wb+") as _f:
            pickle.dump(scores, _f)
        with open(training_durations_file_name, "wb+") as _f:
            pickle.dump(durations, _f)
        with open(validation_scores_file_name, "wb+") as _f:
            pickle.dump(dsc_agent.validation_scores, _f)
        with open(num_option_history_file_name, "wb+") as _f:
            pickle.dump(dsc_agent.num_options_history, _f)

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
        for chain in forward_chains:
            _plot_event_pair(chain.init_salient_event, chain.target_salient_event)

            plt.xticks([])
            plt.yticks([])

        file_name = f"event_graphs_episode_{self.kGraphIterationNumber}.png"
        plt.savefig(os.path.join(self.path, "event_graphs", file_name))
        plt.close()

        self.kGraphIterationNumber += 1

    def visualize_plan_graph(self, plan_graph, seed, episode=None):
        ipdb.set_trace()
        pos = nx.planar_layout(plan_graph)
        labels = nx.get_edge_attributes(plan_graph, "weight")

        # Truncate the labels to 2 decimal places
        for key in labels:
            labels[key] = np.round(labels[key], 2)

        plt.figure(figsize=(16, 10))

        nx.draw_networkx(plan_graph, pos)
        nx.draw_networkx_edge_labels(plan_graph, pos, edge_labels=labels)

        file_name = f"plan_graph_seed_{seed}_episode_{episode}.png" if episode is not None else f"plan_graph_seed_{seed}.png"
        plt.savefig(os.path.join(self.path, "event_graphs", file_name))
        plt.close()

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
