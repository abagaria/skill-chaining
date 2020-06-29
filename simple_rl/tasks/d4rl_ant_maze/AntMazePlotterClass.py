import os
from simple_rl.mdp.MDPPlotterClass import MDPPlotter
import numpy as np
import torch
import matplotlib.pyplot as plt


# import simple_rl.agents.func_approx.dsc.BaseSalientEventClass
class AntMazePlotter(MDPPlotter):
    def __init__(self, experiment_name):
        MDPPlotter.__init__(self, "ant_maze", experiment_name)

    def visualize_best_option_to_take(self, policy_over_options_dqn, episode, seed):
        states = np.array([exp.state for exp in policy_over_options_dqn.replay_buffer.memory])
        states_tensor = torch.from_numpy(states).float().to(policy_over_options_dqn.device)
        options = policy_over_options_dqn.get_best_actions_batched(states_tensor).cpu().numpy()
        x = [s[0] for s in states]
        y = [s[1] for s in states]
        plt.scatter(x, y, c=options, cmap=plt.cm.Dark2)
        plt.colorbar()
        file_name = f"{policy_over_options_dqn.name}_best_options_seed_{seed}_episode_{episode}.png"
        plt.savefig(os.path.join(self.path, file_name))
        plt.close()

    def visualize_ddpg_shaped_rewards(self, global_option, other_option, episode, seed):
        ddpg_agent = global_option.solver
        next_states = np.array([exp[-2] for exp in ddpg_agent.replay_buffer])
        if other_option.should_target_with_bonus():
            shaped_rewards = 1. * other_option.batched_is_init_true(next_states)
            plt.scatter(next_states[:, 0], next_states[:, 1], c=shaped_rewards)
            plt.colorbar()
            file_name = f"{other_option.name}_low_level_shaped_rewards_seed_{seed}_episode_{episode}.png"
            plt.savefig(os.path.join(self.path, file_name))
            plt.close()

    def visualize_dqn_shaped_rewards(self, dqn_agent, option, episode, seed):
        next_states = np.array([exp.next_state for exp in dqn_agent.replay_buffer.memory])
        if option.should_target_with_bonus():
            shaped_rewards = 1.0 * option.batched_is_init_true(next_states)
            plt.scatter(next_states[:, 0], next_states[:, 1], c=shaped_rewards)
            plt.colorbar()
            file_name = f"{option.name}_high_level_shaped_rewards_seed_{seed}_episode_{episode}.png"
            plt.savefig(os.path.join(self.path, file_name))
            plt.close()

    def visualize_dqn_replay_buffer(self, solver):
        goal_transitions = list(filter(lambda e: e[2] >= 0 and e[4] == 1, solver.replay_buffer.memory))
        cliff_transitions = list(filter(lambda e: e[2] < 0 and e[4] == 1, solver.replay_buffer.memory))
        non_terminals = list(filter(lambda e: e[4] == 0, solver.replay_buffer.memory))

        goal_x = [e[3][0] for e in goal_transitions]
        goal_y = [e[3][1] for e in goal_transitions]
        cliff_x = [e[3][0] for e in cliff_transitions]
        cliff_y = [e[3][1] for e in cliff_transitions]
        non_term_x = [e[3][0] for e in non_terminals]
        non_term_y = [e[3][1] for e in non_terminals]

        # background_image = imread("pinball_domain.png")

        plt.figure()
        plt.scatter(cliff_x, cliff_y, alpha=0.67, label="cliff")
        plt.scatter(non_term_x, non_term_y, alpha=0.2, label="non_terminal")
        plt.scatter(goal_x, goal_y, alpha=0.67, label="goal")
        # plt.imshow(background_image, zorder=0, alpha=0.5, extent=[0., 1., 1., 0.])

        plt.legend()
        plt.title(f"# transitions = {len(solver.replay_buffer)}")
        file_name = f"{solver.name}_replay_buffer_analysis.png"
        plt.savefig(os.path.join(self.path, file_name))
        plt.close()

    def visualize_next_state_reward_heat_map(self, option, episode=None):
        solver = option.solver
        next_states = [experience[3] for experience in solver.replay_buffer]
        rewards = np.array([experience[2] for experience in solver.replay_buffer])
        x = np.array([state[0] for state in next_states])
        y = np.array([state[1] for state in next_states])

        plt.scatter(x, y, None, c=rewards.squeeze(), cmap=plt.cm.coolwarm)
        plt.colorbar()
        plt.xlabel("x")
        plt.ylabel("y")
        plt.title("Replay Buffer Reward Heat Map")

        x_low_lim, y_low_lim = option.overall_mdp.get_x_y_low_lims()
        x_high_lim, y_high_lim = option.overall_mdp.get_x_y_high_lims()

        plt.xlim((x_low_lim, x_high_lim))
        plt.ylim((y_low_lim, y_high_lim))

        name = option.name if episode is None else f"{option.name}_{episode}"
        file_name = f"{name}_replay_buffer_reward_map.png"
        plt.savefig(os.path.join(self.path, file_name))
        plt.close()

    def generate_episode_plots(self, chainer, episode):
        self.visualize_best_option_to_take(chainer.agent_over_options, episode, chainer.seed)

        for option in chainer.trained_options:
            self.visualize_ddpg_shaped_rewards(chainer.global_option, option, episode, chainer.seed)
            self.visualize_dqn_shaped_rewards(chainer.agent_over_options, option, episode, chainer.seed)

    def generate_experiment_plots(self, chainer):
        for option in chainer.trained_options:
            self.visualize_dqn_replay_buffer(option.solver)

        for i, o in enumerate(chainer.trained_options):
            plt.subplot(1, len(chainer.trained_options), i + 1)
            plt.plot(chainer.option_qvalues[o.name])
            plt.title(o.name)
        file_name = "sampled_q_so_{}.png".format(chainer.seed)
        plt.savefig(os.path.join(self.path, file_name))
        plt.close()

        for option in chainer.trained_options:
            self.visualize_next_state_reward_heat_map(option.solver, -1)
