# Python imports.
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from collections import defaultdict, Counter
import itertools
import ipdb
import random
import argparse
from copy import deepcopy
from collections import deque
import matplotlib
matplotlib.use('Agg') # non-interactive
import time
from math import ceil

# Other imports.
from simple_rl.tasks.gridworld.gridworld import GridWorld
from simple_rl.tasks.gridworld.sensors import SensorChain, ResampleSensor, ImageSensor, NoisySensor
from simple_rl.agents.func_approx.exploration.optimism.latent.CountingLatentSpaceClass import CountingLatentSpace
from simple_rl.agents.func_approx.dqn.DQNAgentClass import DQNAgent
from simple_rl.tasks.gym.GymMDPClass import GymMDP
from simple_rl.agents.func_approx.exploration.optimism.latent.utils import create_log_dir, save_scores


class Experiment12:
    """ RL on mountain car using the learned exploration bonus. """

    def __init__(self, seed, *, pixel_observation, optimization_quantity, count_train_mode,
                 eval_eps, exploration_method, num_episodes, num_steps, device, experiment_name,
                 bonus_scaling_term, lam_scaling_term, lam_c1, lam_c2,
                 no_novelty_during_regression, tensor_log,
                 phi_type, bonus_from_position, max_to_plot, mcar_win_reward, bonus_form,
                 prioritize_positive_terminal_transitions,
                 use_opiq, opiq_regression_exponent, action_selection_exponent, cls_latent_dim):
        self.mdp = GymMDP("MountainCar-v0", pixel_observation=pixel_observation,
                          seed=seed, control_problem=True)
        state_dim = self.mdp.state_dim
        self.cls_latent_dim = cls_latent_dim

        self.novelty_during_regression = not no_novelty_during_regression
        self.bonus_from_position = bonus_from_position
        normalize_states = not pixel_observation or bonus_from_position

        self.agent = DQNAgent(state_size=state_dim, action_size=len(self.mdp.actions),
                              trained_options=[], seed=seed, device=device,
                              name="GlobalDDQN", lr=1e-3, use_double_dqn=False,
                              exploration_method=exploration_method, pixel_observation=pixel_observation,
                              evaluation_epsilon=eval_eps, tensor_log=tensor_log, experiment_name=experiment_name,
                              bonus_scaling_term=bonus_scaling_term,
                              lam_scaling_term=lam_scaling_term, lam_c1=lam_c1, lam_c2=lam_c2,
                              novelty_during_regression=self.novelty_during_regression,
                              normalize_states=normalize_states,
                              optimization_quantity=optimization_quantity,
                              phi_type=phi_type, bonus_from_position=bonus_from_position, bonus_form=bonus_form,
                              prioritize_positive_terminal_transitions=prioritize_positive_terminal_transitions,
                              use_opiq=use_opiq, opiq_regression_exponent=opiq_regression_exponent,
                              action_selection_exponent=action_selection_exponent,
                              cls_latent_dim=cls_latent_dim)
        self.exploration_method = exploration_method
        self.episodes = num_episodes
        self.num_steps = num_steps
        self.seed = seed
        self.experiment_name = experiment_name
        self.count_train_mode = count_train_mode
        self.max_to_plot = max_to_plot
        self.mcar_win_reward = mcar_win_reward
        self.prioritize_positive_terminal_transitions = prioritize_positive_terminal_transitions

    def run_experiment(self):
        training_scores, training_durations = self.train_dqn_agent(self.agent, self.mdp, self.episodes, self.num_steps)
        return training_scores, training_durations

    def make_latent_plot(self, agent, episode, chunk_size=1000, max_to_plot=10000):

        # Normalize the data before asking for its embeddings
        normalized_sns_buffer = agent.novelty_tracker.get_sns_buffer(normalized=(not self.mdp.pixel_observation))

        # phi(s)
        states = np.array([sns[0] for sns in normalized_sns_buffer])
        max_to_plot = min(max_to_plot, len(states))
        shuffler = np.random.permutation(len(states))[:max_to_plot]
        states = states[shuffler]


        # Chunk up the inputs so as to conserve GPU memory
        num_chunks = int(np.ceil(states.shape[0] / chunk_size))
        input_chunks = np.array_split(states, num_chunks, axis=0)
        states_repr = np.zeros((states.shape[0], agent.novelty_tracker.counting_space.latent_dim))
        current_idx = 0

        for chunk_number, input_chunk in tqdm(enumerate(input_chunks), desc="Making latent plot"):  # type: (int, np.ndarray)
            chunk_repr = self.agent.novelty_tracker.counting_space.extract_features(input_chunk)
            current_chunk_size = len(chunk_repr)
            states_repr[current_idx:current_idx+current_chunk_size] = chunk_repr
            current_idx += current_chunk_size

        assert states_repr.shape[1] == self.cls_latent_dim, states_repr.shape

        # Create all pair-wise combinations of the possible indices in the latent space
        latent_dimensions = list(range(self.cls_latent_dim))
        pairwise_combinations = list(itertools.combinations(latent_dimensions, 2))

        plt.figure(figsize=(16, 12))
        n_rows = ceil(len(pairwise_combinations) / 3)
        n_cols = 3

        for i, (latent_dim_1, latent_dim_2) in enumerate(pairwise_combinations):
            plt.subplot(n_rows, n_cols, i + 1)
            plt.scatter(states_repr[:, latent_dim_1], states_repr[:, latent_dim_2], alpha=0.3)
            plt.xlabel(f"Latent dimension {latent_dim_1}")
            plt.ylabel(f"Latent dimension {latent_dim_2}")

        plt.suptitle("Latent Space after Episode {}".format(episode))
        plt.savefig(f"{self.experiment_name}/latent_plots/latents_{episode}_seed_{self.seed}.png")
        plt.close()

    def make_bonus_plot(self, agent, episode, chunk_size=1000, max_to_plot=10000):

        states = np.array([transition.state for transition in agent.replay_buffer])
        positions = np.array([transition.position for transition in agent.replay_buffer])

        bonus_inputs = positions if self.bonus_from_position else states

        max_to_plot = min(max_to_plot, len(states))
        shuffler = np.random.permutation(len(states))[:max_to_plot]

        states = states[shuffler]
        positions = positions[shuffler]
        bonus_inputs = bonus_inputs[shuffler]

        # Chunk up the inputs so as to conserve GPU memory
        num_chunks = int(np.ceil(bonus_inputs.shape[0] / chunk_size))
        input_chunks = np.array_split(bonus_inputs, num_chunks, axis=0)
        bonuses = np.zeros((bonus_inputs.shape[0], len(agent.actions)))
        current_idx = 0

        for chunk_number, input_chunk in tqdm(enumerate(input_chunks), desc="Making bonus plot"):  # type: (int, np.ndarray)
            chunk_bonuses = agent.novelty_tracker.get_batched_exploration_bonus(input_chunk)
            current_chunk_size = len(chunk_bonuses)
            bonuses[current_idx:current_idx+current_chunk_size] = chunk_bonuses
            current_idx += current_chunk_size

        # bonuses = agent.novelty_tracker.get_batched_exploration_bonus(bonus_inputs)

        plt.figure(figsize=(14, 10))
        for i, action in enumerate(self.mdp.actions):
            plt.subplot(1, len(self.mdp.actions), i + 1)
            plt.scatter(positions[:, 0], positions[:, 1], c=bonuses[:, action], norm=matplotlib.colors.LogNorm())
            plt.colorbar()
            plt.yticks([])

        plt.suptitle("Exploration Bonuses after Episode {}".format(episode))
        plt.savefig(f"{self.experiment_name}/bonus_plots/bonuses_{episode}_seed_{self.seed}.png")
        plt.close()

    def make_value_plot(self, agent, episode, chunk_size=1000, max_to_plot=10000):
        states = np.array([transition.state for transition in agent.replay_buffer])
        positions = np.array([transition.position for transition in agent.replay_buffer])

        max_to_plot = min(max_to_plot, len(states))
        shuffler = np.random.permutation(len(states))[:max_to_plot]
        states = states[shuffler]
        positions = positions[shuffler]


        # Chunk up the inputs so as to conserve GPU memory
        num_chunks = int(np.ceil(states.shape[0] / chunk_size))
        input_chunks = np.array_split(states, num_chunks, axis=0)
        qvalues = np.zeros((states.shape[0], len(agent.actions)))
        current_idx = 0

        for chunk_number, input_chunk in tqdm(enumerate(input_chunks), desc="Making VF plot"):  # type: (int, np.ndarray)
            states_chunk = torch.from_numpy(input_chunk).float().to(agent.device)
            chunk_qvalues = agent.get_batched_qvalues(states_chunk, None).cpu().numpy()
            current_chunk_size = len(states_chunk)
            qvalues[current_idx:current_idx + current_chunk_size] = chunk_qvalues
            current_idx += current_chunk_size

        plt.figure(figsize=(14, 10))
        for action in self.agent.actions:
            plt.subplot(1, len(self.agent.actions), action + 1)
            plt.scatter(positions[:, 0], positions[:, 1], c=qvalues[:, action])#, norm=matplotlib.colors.LogNorm())
            plt.colorbar()

        plt.suptitle("Q-functions (no novelty bonus) after episode {}".format(episode))
        plt.savefig(f"{self.experiment_name}/qf_plots/vf_{episode}_seed_{self.seed}.png")
        plt.close()

    # def make_advantage_plot(self, agent, episode):
    #     states = np.array([transition.state for transition in agent.replay_buffer])
    #     positions = np.array([transition.position for transition in agent.replay_buffer])

    #     states_tensor = torch.from_numpy(states).float().to(agent.device)

    #     qvalues = agent.get_batched_qvalues(states_tensor, None).cpu().numpy()
    #     av_qs = qvalues.mean(axis=1, keepdims=True)
    #     q_advantage = qvalues - av_qs

    #     plt.figure(figsize=(14, 10))
    #     for action in self.agent.actions:
    #         plt.subplot(1, len(self.agent.actions), action + 1)
    #         plt.scatter(positions[:, 0], positions[:, 1], c=q_advantage[:, action])#, norm=matplotlib.colors.LogNorm())
    #         plt.colorbar()

    #     plt.suptitle("Q-Advantage (no novelty bonus) after episode {}".format(episode))
    #     plt.savefig(f"{self.experiment_name}/q_advantage_plots/qadv_{episode}_seed_{self.seed}.png")
    #     plt.close()

    def make_which_actions_plot(self, agent, episode, max_to_plot=10000, allow_novelty=False):
        # Allow novelty determines whether novelty is used to make the plots.
        # Sort of simulates how the agent would do greedily.
        evaluation_epsilon = agent.evaluation_epsilon
        agent.evaluation_epsilon = 0.

        states = np.array([transition.state for transition in agent.replay_buffer])
        positions = np.array([transition.position for transition in agent.replay_buffer])

        max_to_plot = min(max_to_plot, len(states))
        shuffler = np.random.permutation(len(states))[:max_to_plot]
        states = states[shuffler]
        positions = positions[shuffler]

        # Doesn't need chunking cause we do one state at a time...
        actions = []
        for s, p in tqdm(zip(states, positions), desc="Making Which-Action plot"):
            action = agent.act(s, p, train_mode=True, use_novelty=allow_novelty)
            actions.append(action)

        plt.figure(figsize=(14, 10))
        for action in self.agent.actions:
            plt.subplot(1, len(self.agent.actions), action + 1)
            take_this_action_from = np.array([p for p,a in zip(positions, actions) if a == action])
            take_other_action_from = np.array([p for p,a in zip(positions, actions) if a != action])
            if len(take_other_action_from) != 0:
                plt.scatter(take_other_action_from[:, 0], take_other_action_from[:, 1], c='#000000')#, norm=matplotlib.colors.LogNorm())
            if len(take_this_action_from) != 0:
                plt.scatter(take_this_action_from[:, 0], take_this_action_from[:, 1], c='r')#, norm=matplotlib.colors.LogNorm())

        plt.suptitle("Action chosen after episode {}".format(episode))
        plt.savefig(f"{self.experiment_name}/which_action/which_action_{episode}_seed_{self.seed}.png")
        plt.close()
        agent.evaluation_epsilon = evaluation_epsilon

    def train_dqn_agent(self, agent, mdp, episodes, steps):
        per_episode_scores = []
        per_episode_durations = []

        last_10_scores = deque(maxlen=10)
        last_10_durations = deque(maxlen=10)
        iteration_counter = 0

        for episode in range(episodes):
            episode_start_time = time.time()
            mdp.reset()
            state = deepcopy(mdp.cur_state)

            score = 0.

            for step in range(steps):
                iteration_counter += 1

                position = mdp.get_position()
                action = agent.act(state.features(), train_mode=True, position=position, use_novelty=True)
                reward, next_state = mdp.execute_agent_action(action)

                reward = self.mcar_win_reward if next_state.is_terminal() else 0.
                next_position = mdp.get_position()

                agent.step(state.features(), position, action, reward, next_state.features(), next_position,
                           next_state.is_terminal(), num_steps=1)
                agent.update_epsilon()

                state = next_state
                score += reward

                if state.is_terminal():
                    break

            total_episode_time = time.time() - episode_start_time

            if agent.tensor_log:
                agent.writer.add_scalar("Score", score, iteration_counter)

            if agent.exploration_method == "count-phi" and episode >= 1:
                agent.train_novelty_detector(mode=self.count_train_mode)

            if args.make_plots:
                if self.exploration_method == "count-phi":
                    self.make_latent_plot(agent, episode, max_to_plot=self.max_to_plot)
                    self.make_bonus_plot(agent, episode, max_to_plot=self.max_to_plot)
                    self.make_value_plot(agent, episode, max_to_plot=self.max_to_plot)
                    # self.make_advantage_plot(agent, episode)
                    self.make_which_actions_plot(agent, episode, max_to_plot=self.max_to_plot, allow_novelty=False)

            last_10_scores.append(score)
            last_10_durations.append(step + 1)
            per_episode_scores.append(score)
            per_episode_durations.append(step + 1)

            sns_size = len(agent.novelty_tracker.un_normalized_sns_buffer) if self.exploration_method == "count-phi" else 0
            lam = agent.novelty_tracker.counting_space.lam if self.exploration_method == "count-phi" else 0

            if self.prioritize_positive_terminal_transitions:
                log_str = '\rEpisode {}\tAverage Score: {:.2f}\tAverage Duration: {:.2f}\tEpsilon: {:.2f}\tSNS Size: {}\tLam: {}\tNum Pos Transitions: {}\tTime: {:.2f}'.format(episode,
                                                                                                                             np.mean(last_10_scores),
                                                                                                                             np.mean(last_10_durations),
                                                                                                                             agent.epsilon, sns_size, lam,
                                                                                                                             len(self.agent.replay_buffer.positive_transitions),
                                                                                                                             total_episode_time)
            else:
                log_str = '\rEpisode {}\tAverage Score: {:.2f}\tAverage Duration: {:.2f}\tEpsilon: {:.2f}\tSNS Size: {}\tLam: {}\tTime: {:.2f}'.format(episode,
                                                                                                                             np.mean(last_10_scores),
                                                                                                                             np.mean(last_10_durations),
                                                                                                                             agent.epsilon, sns_size, lam, total_episode_time)
            print(log_str)
            save_scores(scores=per_episode_durations, experiment_name=self.experiment_name, seed=self.seed)

        return per_episode_scores, per_episode_durations


if __name__ == "__main__":
    """
    Example run (from root dir):
        python simple_rl/agents/func_approx/exploration/optimism/latent/experiments/experiment12.py --experiment_name testing --run_title testing --episodes 100 --pixel_observation --exploration_method chunked-log --use_bonus_during_action_selection --make_plots --bonus_scaling_term none --lam_scaling_term fit --optimization_quantity count-phi --count_train_mode partial
    """

    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment_name", type=str, help="Experiment Name", required=True)
    parser.add_argument("--run_title", type=str, required=True) # This is the subdir that we'll be saving in.

    parser.add_argument("--seed", type=int, help="Random seed for this run (default=0)", default=0)
    parser.add_argument("--episodes", type=int, help="# episodes", default=100)
    parser.add_argument("--steps", type=int, help="# steps", default=200)
    parser.add_argument("--render", type=bool, help="Render the mdp env", default=False)
    parser.add_argument("--pixel_observation", action="store_true", help="Images / Dense input", default=False)
    parser.add_argument("--exploration_method", type=str, default="eps-decay")
    parser.add_argument("--use_bonus_during_action_selection", action="store_true", default=False)
    parser.add_argument("--eval_eps", type=float, default=0.05)
    parser.add_argument("--make_plots", action="store_true", default=False)
    parser.add_argument("--bonus_scaling_term", type=str, default="none")
    parser.add_argument("--lam_scaling_term", type=str, default="fit")
    parser.add_argument("--no_novelty_during_regression", action="store_true", default=False)
    parser.add_argument("--tensor_log", action="store_true", default=False)
    parser.add_argument("--optimization_quantity", type=str)
    parser.add_argument("--count_train_mode", type=str, default="entire")
    parser.add_argument("--phi_type", type=str, default="function")
    parser.add_argument("--bonus_from_position", action="store_true", default=False)
    parser.add_argument("--mcar_win_reward", type=float, default=20.)
    parser.add_argument("--bonus_form", type=str, default="sqrt")
    parser.add_argument("--prioritize_positive_terminal_transitions", action="store_true", default=False)
    parser.add_argument("--use_opiq", action="store_true", default=False)
    parser.add_argument("--opiq_regression_exponent", type=float, default=-0.5)
    parser.add_argument("--action_selection_exponent", type=float, default=-2.0)
    parser.add_argument("--cls_latent_dim", type=int, default=2)
    parser.add_argument("--lam_c1", type=float, default=None)
    parser.add_argument("--lam_c2", type=float, default=None)

    args = parser.parse_args()

    full_experiment_name = f"{args.experiment_name}/{args.run_title}"

    create_log_dir(args.experiment_name)
    create_log_dir(full_experiment_name)
    create_log_dir(f"{full_experiment_name}/bonus_plots")
    create_log_dir(f"{full_experiment_name}/latent_plots")
    create_log_dir(f"{full_experiment_name}/qf_plots")
    create_log_dir(f"{full_experiment_name}/q_advantage_plots")
    create_log_dir(f"{full_experiment_name}/which_action")
    create_log_dir(f"{full_experiment_name}/scores")


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    exp = Experiment12(args.seed, pixel_observation=args.pixel_observation,
                      eval_eps=args.eval_eps, exploration_method=args.exploration_method, num_episodes=args.episodes, tensor_log=args.tensor_log,
                      num_steps=args.steps, device=device, experiment_name=full_experiment_name, bonus_scaling_term=args.bonus_scaling_term,
                      no_novelty_during_regression=args.no_novelty_during_regression,
                      lam_scaling_term=args.lam_scaling_term, lam_c1=args.lam_c1, lam_c2=args.lam_c2,
                      optimization_quantity=args.optimization_quantity, count_train_mode=args.count_train_mode,
                      phi_type=args.phi_type, bonus_from_position=args.bonus_from_position,
                      max_to_plot=10000, mcar_win_reward=args.mcar_win_reward, bonus_form=args.bonus_form,
                      prioritize_positive_terminal_transitions=args.prioritize_positive_terminal_transitions,
                      use_opiq=args.use_opiq, opiq_regression_exponent=args.opiq_regression_exponent,
                      action_selection_exponent=args.action_selection_exponent, cls_latent_dim=args.cls_latent_dim
                    )

    episodic_scores, episodic_durations = exp.run_experiment()

    save_scores(episodic_durations, args.experiment_name, args.seed, run_title=args.run_title)
