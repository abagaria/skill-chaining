import ipdb
import random
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt
from simple_rl.mdp.StateClass import State
from simple_rl.agents.func_approx.dsc.OptionClass import Option
from simple_rl.agents.func_approx.dsc.SalientEventClass import SalientEvent
from simple_rl.agents.func_approx.dsc.ChainClass import SkillChain


class UCBActionSelectionAgent(object):
    """ Goal directed Option selection when the goal-state lies outside the skill graph. """

    def __init__(self, goal_state, options, events, chains):
        """

        Args:
            goal_state (State or np.ndarray)
            options (list): List of options in the known part of the graph
            events (list): List of salient events in the known part of the graph
            chains (list): List of skill-chains in the MDP
        """
        assert isinstance(goal_state, (State, np.ndarray)), goal_state
        assert all([option.get_training_phase() == "initiation_done" for option in options])
        assert not any([option.name == "global_option" for option in options]), options

        print(f"Creating bandit targeting {goal_state} with options {options} and events {events}")

        self.options = options
        self.chains = chains
        self.events = events
        self.goal_state = goal_state

        # Mapping from option to a floating number representing the fraction of
        # times we have been able to jump off that option to reach the `self.goal_state`
        self.monte_carlo_estimates = defaultdict(float)

        # Mapping from option to an integer representing the number of times
        # we have jumped off the graph from that option's initiation set
        self.node_visitation_counts = defaultdict(int)

        # The total number of times we have executed *any* option by querying
        # the current bandit algorithm. This will be used in place of the horizon in UCB1
        self.total_executions = 0

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

    def add_candidate_event(self, salient_event):
        if salient_event not in self.events:
            self.events.append(salient_event)
            print(f"Adding {salient_event} to the list of options for {self}. Now my events are {self.events}")

    def act(self):
        """
        Given the goal state, the available options in the skill-graph and the number of
        times each option has been executed, choose the option whose initiation set to jump
        off from.

        Returns:
            selected_option (Option or SalientEvent)
        """
        # Only consider the option nodes that are in the known part of the graph
        candidate_nodes = self._get_candidate_nodes()

        if len(candidate_nodes) > 0:

            # Compute the UCB-augmented values for each of the candidate nodes
            values = []

            for node in candidate_nodes:
                augmented_value = self._get_node_value(node)
                values.append(augmented_value)

            # Now pick the node to jump off the graph from
            best_node_idx = np.argmax(values)

            # Tie break if there are multiple nodes with the same value
            if isinstance(best_node_idx, (list, np.ndarray)):
                best_node_idx = random.choice(best_node_idx)

            # Otherwise, pick the node with the highest value
            assert isinstance(best_node_idx, int), best_node_idx
            return candidate_nodes[best_node_idx]

        return None


    def update(self, node, success):
        """
        Given that we jumped off the graph from `node` in the MDP, update the visitation statistics.
        Args:
            node (Option or Salient Event)
            success (bool)

        """
        self.node_visitation_counts[node] += 1
        self.total_executions += 1

        self.monte_carlo_estimates[node] += (float(success) / self.node_visitation_counts[node])

    def _get_node_bonus(self, node):
        """
        Based on the UCB1 algorithm and the visitation counts, what is the
        exploration bonus associated with executing `option` to get to goal
        `state`.
        Args:
            node (Option or SalientEvent)

        Returns:
            bonus (float)
        """
        num_executions = self.node_visitation_counts[node]

        if num_executions == 0:
            return self.max_bonus

        horizon_term = np.log(self.total_executions)
        bonus = np.sqrt(horizon_term / num_executions)
        return bonus

    def _get_node_value(self, node):
        mc_value = self.monte_carlo_estimates[node]
        bonus = self._get_node_bonus(node)
        augmented_value = mc_value + (self.trade_off_constant * bonus)
        return augmented_value

    def _filter_candidate_options(self):
        """ Given the full set of candidate options, we only want to pick those that are part of a complete chain. """
        filtered_options = []
        for option in self.options:  # type: Option
            chain = self.chains[option.chain_id - 1]  # type: SkillChain
            if chain.is_chain_completed(self.chains):
                filtered_options.append(option)
        return filtered_options

    def _filter_candidate_events(self):
        """ Given the full set of candidate events in the MDP, we only want to pick those that have
            some chains targeting it. """
        def _is_chain_eligible(skill_chain, salient):
            return skill_chain.target_salient_event == salient and not skill_chain.is_backward_chain

        filtered_events = []
        for event in self.events:  # type: SalientEvent
            targeting_chains = [chain for chain in self.chains if _is_chain_eligible(chain, event)]
            if any([chain.is_chain_completed() for chain in targeting_chains]):
                filtered_events.append(event)
        return filtered_events

    def _get_candidate_nodes(self):
        candidate_options = self._filter_candidate_options()
        candidate_events = self._filter_candidate_events()
        candidate_nodes = candidate_options + candidate_events
        return candidate_nodes
