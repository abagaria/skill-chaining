import ipdb
import random
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt
from simple_rl.mdp.StateClass import State
from simple_rl.agents.func_approx.dsc.OptionClass import Option
from simple_rl.agents.func_approx.ddpg.DDPGAgentClass import DDPGAgent
from simple_rl.agents.func_approx.dsc.ChainClass import SkillChain


class UCBActionSelectionAgent(object):
    """ Goal directed Option selection when the goal-state lies outside the skill graph. """

    def __init__(self, goal_state, options, chains, use_option_vf=False):
        """

        Args:
            goal_state (State or np.ndarray)
            options (list): List of options in the known part of the graph
            chains (list): List of skill-chains in the MDP
            use_option_vf (bool): Whether to initialize the value of jumping off an option
                                  with that option's value for the `goal_state`.
        """
        assert isinstance(goal_state, (State, np.ndarray)), goal_state
        assert all([option.get_training_phase() == "initiation_done" for option in options])
        assert not any([option.name == "global_option" for option in options]), options

        print(f"Creating bandit targeting {goal_state} with options {options}")

        self.options = options
        self.chains = chains
        self.goal_state = goal_state
        self.use_option_vf = use_option_vf

        # Mapping from option to a floating number representing the fraction of
        # times we have been able to jump off that option to reach the `self.goal_state`
        self.monte_carlo_estimates = defaultdict(float)

        # Mapping from option to an integer representing the number of times
        # we have jumped off the graph from that option's initiation set
        self.option_visitation_counts = defaultdict(int)

        # The total number of times we have executed *any* option by querying
        # the current bandit algorithm. This will be used in place of the horizon in UCB1
        self.total_option_executions = 0

        # Trade-off between exploration and exploitation
        self.trade_off_constant = np.sqrt(2)
        self.max_bonus = 1000.

    def __str__(self):
        return f"Bandit algorithm targeting {str(self.goal_state)}"

    def __repr__(self):
        return str(self)

    def add_candidate_option(self, option):
        """
        If a new option is added to the graph, we want the UCB agent to be able
        to select that new option as the one that the DSG agent should jump off from.

        Args:
            option (Option)

        """
        assert option.get_training_phase() == "initiation_done"
        if option not in self.options:
            self.options.append(option)
            print(f"Adding {option} to the list of options for {self}. Now my options are {self.options}")

    def act(self):
        """
        Given the goal state, the available options in the skill-graph and the number of
        times each option has been executed, choose the option whose initiation set to jump
        off from.
        Returns:
            selected_option (Option)
        """
        candidate_options = self._filter_candidate_options()
        if len(candidate_options) > 0:
            option_values = [self._get_option_value(option, self.goal_state) for option in candidate_options]
            exploration_bonuses = [self._get_option_bonus(option) for option in candidate_options]
            selected_option = self._get_best_option(option_values, exploration_bonuses)
            return selected_option
        return None

    def update(self, option, success):
        """
        Given that we executed `option` in the MDP, update the visitation statistics.
        Args:
            option (Option)
            success (bool)

        """
        self.option_visitation_counts[option] += 1
        self.total_option_executions += 1

        self.monte_carlo_estimates[option] += (float(success) / self.option_visitation_counts[option])

    def _get_option_bonus(self, option):
        """
        Based on the UCB1 algorithm and the visitation counts, what is the
        exploration bonus associated with executing `option` to get to goal
        `state`.
        Args:
            option (Option)

        Returns:
            bonus (float)
        """
        num_option_executions = self.option_visitation_counts[option]

        if num_option_executions == 0:
            return self.max_bonus

        horizon_term = np.log(self.total_option_executions)
        bonus = np.sqrt(horizon_term/ num_option_executions)
        return bonus

    def _get_best_option(self, option_values, exploration_bonuses):
        """
        UCB1 option selection.
        Args:
            option_values (list)
            exploration_bonuses (list)

        Returns:
            selected_option (Option)
        """
        assert len(option_values) == len(exploration_bonuses)
        augmented_values = [v + (self.trade_off_constant*b) for v, b in zip(option_values, exploration_bonuses)]

        best_option_idx = np.argmax(augmented_values)

        if isinstance(best_option_idx, (list, np.ndarray)):
            best_option_idx = random.choice(best_option_idx)

        return self.options[best_option_idx]

    def _get_option_value(self, option, state):
        """
        What is the value of the `goal_state` under `option`'s value function and the
        monte-carlo rollouts performed under the current bandit algorithm.
        Args:
            option (Option)
            state (State or np.ndarray)

        Returns:
            value (float)
        """
        option_solver = option.solver  # type: DDPGAgent
        query_state = state.features() if isinstance(state, State) else state
        value = option_solver.get_value(query_state) if self.use_option_vf else 0.

        # To match the scaling between the MC-rollouts and the option value function,
        # we scale the MC-estimates by the sub-goal reward (unless its 0)
        scaling_factor = option.subgoal_reward if option.subgoal_reward > 0 and self.use_option_vf else 1.
        monte_carlo_estimate = self.monte_carlo_estimates[option]
        combined_value = value + (scaling_factor * monte_carlo_estimate)

        return combined_value

    def _filter_candidate_options(self):
        """ Given the full set of candidate options, we only want to pick those that are part of a complete chain. """
        filtered_options = []
        for option in self.options:  # type: Option
            chain = self.chains[option.chain_id - 1]  # type: SkillChain
            if chain.is_chain_completed(self.chains):
                filtered_options.append(option)
        return filtered_options

    def visualize_option_values(self, states, seed, experiment_name):
        for option in self.options:  # type: Option
            x = [state.position[0] for state in states]
            y = [state.position[1] for state in states]
            values = [self._get_option_value(option, state) for state in states]
            plt.scatter(x, y, c=values)
            plt.colorbar()
            file_name = f"{option.name}_value_function_seed_{seed}"
            plt.savefig(f"value_function_plots/{experiment_name}/{file_name}.png")
            plt.close()
