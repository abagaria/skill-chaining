import ipdb
import torch
import argparse
import numpy as np
from tqdm import tqdm
from copy import deepcopy
import torch.nn.functional as F
from simple_rl.agents.func_approx.dsc.dynamics.mpc import MPC
from simple_rl.agents.func_approx.td3.utils import create_log_dir
from simple_rl.tasks.d4rl_ant_maze.D4RLAntMazeMDPClass import D4RLAntMazeMDP
from simple_rl.agents.func_approx.polo.ensemble import ValueFunctionEnsemble
from simple_rl.agents.func_approx.polo.utils import visualize_value_function, visualize_value_targets


class POLOAgent:
    def __init__(self, mdp, goal, ensemble_size, var_scaling, use_bootstrapping, device):
        self.mdp = mdp
        self.goal = goal
        self.device = device
        self.var_scaling = var_scaling
        self.ensemble_size = ensemble_size
        self.use_bootstrapping = use_bootstrapping
        self.state_size = mdp.state_space_size()
        self.action_size = mdp.action_space_size()

        self.update_number = 0
        self.value_function_ensemble = ValueFunctionEnsemble(self.state_size, ensemble_size, device)
        self.mpc = MPC(mdp, self.state_size, self.action_size, dense_reward=True, device=device)

        # Hyperparameters (taken from POLO paper)
        self.gamma = 0.95
        self.horizon = 7
        self.value_func_batch_size = 32
        self.dynamics_batch_size = 1024
        self.num_rollouts = 5000
        self.update_interval = 1000
        self.num_gradient_steps = 64

    def act(self, state):
        return self.mpc.act(state, self.goal, vf=self.value)

    def value(self, states, goals=None):
        means, var = self.value_function_ensemble(states)
        assert isinstance(means, np.ndarray), type(means)
        assert isinstance(var, np.ndarray), type(var)
        return means + (self.var_scaling * var)

    def step(self, state, action, reward, next_state, done):
        self.update_number += 1
        self.mpc.step(state, action, reward, next_state, done)
        self.value_function_ensemble.add_data(state)

        if self.update_number % self.update_interval == 0:
            self.update_dynamics_model()
            self.update_value_function()

    def update_dynamics_model(self, epochs=5):
        self.mpc.load_data()
        self.mpc.train(epochs=epochs, batch_size=1024)

    def update_value_function(self):
        for i in range(self.ensemble_size):
            vf_member = self.value_function_ensemble.members[i]
            optimizer = self.value_function_ensemble.optimizers[i]

            for _ in range(self.num_gradient_steps):
                sampled_states = self.value_function_ensemble.sample(i)
                assert sampled_states.shape == (self.value_func_batch_size, self.state_size)

                predictions = vf_member(sampled_states)

                with torch.no_grad():
                    targets = self.get_target_values(sampled_states, vf_member).detach()

                loss = F.mse_loss(predictions.squeeze(), targets)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            print(f"Ensemble {i} Final Loss: {loss.item()}")


    def get_target_values(self, states, vf):
        """ Given a state and a single VF, return the regression target for the next iteration of VF learning. """

        predicted_next_states, _, costs = self.mpc.batched_simulate(states,
                                                                    num_steps=self.horizon,
                                                                    num_rollouts=self.num_rollouts)

        discounted_reward = self.get_n_step_returns(costs)
        assert discounted_reward.shape == (states.shape[0], self.num_rollouts), discounted_reward.shape

        if self.use_bootstrapping:
            discounted_next_values = self.get_bootstrap_next_state_values(predicted_next_states, vf)
            assert discounted_next_values.shape == (states.shape[0], self.num_rollouts), discounted_next_values.shape

            return self.get_combined_target_values(discounted_reward, discounted_next_values)

        return discounted_reward.mean(dim=1).float()

    def get_n_step_returns(self, cost_tensor):
        gammas = np.power(self.gamma * np.ones(self.horizon), np.arange(0, self.horizon))
        cumulative_costs = torch.sum(cost_tensor * gammas, dim=-1)
        discounted_reward = -cumulative_costs.to(self.device)
        return discounted_reward

    def get_bootstrap_next_state_values(self, predicted_next_states, vf):
        next_state_values = vf.predict(predicted_next_states).squeeze()
        discounted_next_values = (self.gamma ** self.horizon) * next_state_values
        return discounted_next_values

    @staticmethod
    def get_combined_target_values(discounted_reward, discounted_next_values):
        assert discounted_next_values.shape == discounted_reward.shape
        target = discounted_next_values + discounted_reward
        return target.mean(dim=1).float()

def warm_up_period(agent, mdp, episodes, steps):
    for _ in tqdm(range(episodes), desc="Warm up run-loop"):
        mdp.reset()
        state = deepcopy(mdp.cur_state)
        for _ in range(steps):
            action = mdp.sample_random_action()
            reward, next_state = mdp.execute_agent_action(action)
            agent.mpc.step(state.features(), action, reward, next_state.features(), next_state.is_terminal())
            state = next_state

    agent.update_dynamics_model(epochs=50)


def train(agent, mdp, episodes, steps):

    episodic_rewards = []
    for episode in range(episodes):
        mdp.reset()
        score = 0.
        state = deepcopy(mdp.cur_state)
        for _ in range(steps):
            action = agent.act(state)
            reward, next_state = mdp.execute_agent_action(action)
            agent.step(state.features(), action, reward, next_state.features(), next_state.is_terminal())
            score += reward
            state = next_state

        episodic_rewards.append(score)
        print(f"Episode {episode} \t Reward: {score} \t Final pos: {np.round(state.position, decimals=2)}")

        visualize_value_function(agent, args.experiment_name, episode)
        visualize_value_targets(agent, args.experiment_name, episode)

    return episodic_rewards


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment_name", type=str, help="Experiment Name")
    parser.add_argument("--render", type=bool, help="render environment training", default=False)
    parser.add_argument("--episodes", type=int, help="number of training episodes", default=200)
    parser.add_argument("--steps", type=int, help="number of steps per episode", default=200)
    parser.add_argument("--device", type=str, help="cuda/cpu", default="cpu")
    parser.add_argument("--var_scaling", type=float, default=0.1)
    parser.add_argument("--use_bootstrapping", action="store_true", default=False)
    args = parser.parse_args()

    log_dir = create_log_dir(args.experiment_name)
    create_log_dir("value_function_plots")
    create_log_dir("value_function_plots/{}".format(args.experiment_name))

    g = np.array((8., 0.))
    overall_mdp = D4RLAntMazeMDP("umaze", goal_state=g, render=args.render)
    agent = POLOAgent(overall_mdp, goal=g, ensemble_size=4, device=args.device,
                      var_scaling=args.var_scaling, use_bootstrapping=args.use_bootstrapping)

    warm_up_period(agent, overall_mdp, 50, 1000)
    r = train(agent, overall_mdp, episodes=args.episodes, steps=args.steps)
