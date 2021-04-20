import ipdb
import random
import itertools
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt
from simple_rl.mdp.StateClass import State
from simple_rl.agents.func_approx.dsc.OptionClass import Option
from simple_rl.agents.func_approx.dsc.SalientEventClass import SalientEvent
from simple_rl.agents.func_approx.dsc.ChainClass import SkillChain


class UCBNodeSelectionAgent(object):
    """ Goal directed node selection when trying to connect sub-graphs in the skill-graph. """

    def __init__(self, root_event, descendant_events, target_events, alpha=0.1):
        """

        Args:
           root_event (SalientEvent): event corresponding to current sub-graph
           descendant_events (list): list of salient events reachable from the root_event
           target_events (list): list of events not currently reachable from the root_event 
           alpha (float)

        """
        assert isinstance(root_event, SalientEvent), f"{type(root_event)}"
        assert root_event not in target_events, root_event

        print(f"Creating bandit from {root_event} targetting {target_events}")

        self.root_event = root_event
        self.target_events = target_events
        self.descendant_events = descendant_events

        # Sum of EMDs between all u, v pairs in the graph
        self.norm_distance = self.determine_normalization_factor() if len(target_events) > 0 else 1.

        # Mapping from node-pair (descendant-ancestor pair) to a floating number representing the 
        # value of jumping from one node to the other
        self.q_table = {}  # TODO: Need to initialize with a defaultdict

        # Mapping from node pair to the number of times we have tried it
        self.node_visitation_counts = defaultdict(int)

        # The total number of times we have executed *any* option by querying
        # the current bandit algorithm. This will be used in place of the horizon in UCB1
        self.total_executions = 0

        # Trade-off between exploration and exploitation
        self.trade_off_constant = np.sqrt(2)  # TODO: Set this based on how many updates we expect before connecting two events.
        self.max_bonus = 1.

        # Q-value learning rate
        self.alpha = alpha
        
        # Debug
        self.log = {}

    def __str__(self):
        return f"Bandit rooted at {str(self.root_event)}"

    def __repr__(self):
        return str(self)

    def init_q(self, node):
        assert isinstance(node, tuple), node
        assert isinstance(node[0], SalientEvent), type(node[0])
        assert isinstance(node[1], SalientEvent), type(node[1])
        
        assert self.norm_distance > 0, self.norm_distance

        src_node, dest_node = node[0], node[1]
        return -src_node.distance_to_other_event(dest_node) / self.norm_distance

    def determine_normalization_factor(self):
        src_nodes = self._get_src_nodes()
        all_nodes = src_nodes + self.target_events
        all_node_pairs = itertools.combinations(all_nodes, 2)
        distances = [src.distance_to_other_event(dest) for src, dest in all_node_pairs]
        distance_sum = sum(distances)

        print(f"Summed distance between nodes in {self} = {distance_sum}")
        return distance_sum

    def add_descendant_event(self, salient_event):
        """ Add a newly added `salient_event` if there is a path to it from the `root_event`. """
        assert isinstance(salient_event, SalientEvent), f"{type(salient_event)}"

        if salient_event not in self.descendant_events:
            self.descendant_events.append(salient_event)
            print(f"Adding {salient_event} to the list of descedants for {self}. Now my descendants are {self.descendant_events}")

            self.norm_distance = self.determine_normalization_factor()

    def add_target_event(self, salient_event):
        """ Add a newly added `salient_event` if there is no path to it from the `root_event`. """
        assert isinstance(salient_event, SalientEvent), f"{type(salient_event)}"
        assert salient_event not in self.descendant_events, salient_event
        assert salient_event != self.root_event, salient_event
        
        if salient_event not in self.target_events:
            self.target_events.append(salient_event)
            print(f"Adding {salient_event} to the list of targets for {self}. Now my target nodes are {self.target_events}")

            self.norm_distance = self.determine_normalization_factor()

    def remove_descendant(self, salient_event):
        """ Remove the event from the list of descendants if there is no longer a path from the root to that event. """
        assert isinstance(salient_event, SalientEvent), f"{type(salient_event)}"

        if salient_event in self.descendant_events:
            self.descendant_events.remove(salient_event)
            print(f"Removing {salient_event} from list of descendants for {self}. Now descendants are {self.descendant_events}")

            self.norm_distance = self.determine_normalization_factor()

    def remove_target(self, salient_event):
        """ Remove the event from the list of targets if there is a path from the root to that event now. """
        assert isinstance(salient_event, SalientEvent), f"{type(salient_event)}"

        if salient_event in self.target_events:
            self.target_events.remove(salient_event)
            print(f"Removing {salient_event} from list of targets for {self}. Now targets are {self.target_events}")

            self.norm_distance = self.determine_normalization_factor()

    def act(self):
        """
        Choose the best node u to jump off the current sub-graph from. 
        Choose the best node v to target in some other unconnected sub-graph.

        Returns:
            src_node (SalientEvent): node to jump off from
            dest_node (SalientEvent): node to jump on to
        """
        # Get all feasible (u, v) pairs of nodes
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
            assert isinstance(best_node_idx, (int, np.int64)), best_node_idx
            return candidate_nodes[best_node_idx]

        return None

    def update(self, src_node, dest_node, success):
        """
        Given that we jumped off the graph from `src_node` targetting `dest_node`, update 
        the visitation counts and the Q-table. 

        Args:
            src_node (SalientEvent)
            dest_node (SalientEvent)
            success (bool)
        """
        assert isinstance(src_node, SalientEvent), f"{type(src_node)}"
        assert isinstance(dest_node, SalientEvent), f"{type(dest_node)}"
        assert isinstance(success, bool), f"{type(success)}"

        key = (src_node, dest_node)
        self.total_executions += 1
        self.node_visitation_counts[key] += 1

        reward = 0. if success else -1.
        prediction = self.get_q(key)
        error = reward - prediction
        self.q_table[key] = prediction + (self.alpha * error)

        self._log_progress(key, reward, success)

    def _log_progress(self, key, reward, success):
        """ Log progress of the UCB agent after every update() call. """ 

        if key not in self.log:
            self.log[key] = {}
            self.log[key]["average_rewards"] = []
            self.log[key]["q_values"] = []
            self.log[key]["successes"] = []
        
        self.log[key]["successes"].append(success)
        self.log[key]["q_values"].append(self.q_table[key])
        self.log[key]["average_rewards"].append(reward / self.node_visitation_counts[key])
        
    def get_q(self, node):
        """ Fetch the q-value of a node if it is in the table. 
        If it is not in the table, consult the init q-function and it. """
        assert isinstance(node, tuple)
        assert isinstance(node[0], SalientEvent)
        assert isinstance(node[1], SalientEvent)

        if node in self.q_table:
            return self.q_table[node]
        v = self.init_q(node)
        self.q_table[node] = v
        return v

    def _get_node_bonus(self, node):
        """
        Based on the UCB1 algorithm and the visitation counts, what is the
        exploration bonus associated with executing `option` to get to goal
        `state`.
        Args:
            node (tuple)

        Returns:
            bonus (float)
        """
        assert isinstance(node, tuple), node
        assert len(node) == 2, len(node)
        assert isinstance(node[0], SalientEvent), f"{type(node[0])}"
        assert isinstance(node[1], SalientEvent), f"{type(node[1])}"
        
        num_executions = self.node_visitation_counts[node]

        if num_executions == 0:
            return self.max_bonus

        horizon_term = np.log(self.total_executions)
        bonus = np.sqrt(horizon_term / num_executions)
        return bonus

    def _get_node_value(self, node):
        """" Get the UCB value of the input `node`. """

        value = self.get_q(node)
        bonus = self._get_node_bonus(node)
        augmented_value = value + (self.trade_off_constant * bonus)
        print(f"Value of {node} is {augmented_value}")
        return augmented_value

    def _get_candidate_nodes(self):
        """ Return all the u, v pairs where u is the set of events in the 
        current sub-graph and v is the set of events in all unconnected 
        sub-graphs. """

        nodes = []
        for src in self._get_src_nodes():
            for dest in self.target_events:
                nodes.append((src, dest))
        return nodes

    def _get_src_nodes(self):
        return [self.root_event] + self.descendant_events
