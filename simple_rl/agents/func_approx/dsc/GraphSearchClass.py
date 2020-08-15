import numpy as np
import networkx as nx
import ipdb
import matplotlib.pyplot as plt
import networkx.algorithms.shortest_paths as shortest_paths
from simple_rl.agents.func_approx.dsc.OptionClass import Option
from simple_rl.agents.func_approx.dsc.ChainClass import SkillChain
from simple_rl.agents.func_approx.dsc.SalientEventClass import SalientEvent


class GraphSearch(object):
    def __init__(self, pre_compute_shortest_paths=False):
        self.plan_graph = nx.DiGraph()
        self.pre_compute_shortest_paths = pre_compute_shortest_paths
        self.shortest_paths = {}
        self.option_nodes = []
        self.salient_nodes = []

    def add_node(self, node):
        if node not in self.plan_graph.nodes:
            self.plan_graph.add_node(node)

        #  Keep track of the options and the salient events separately in the graph
        if isinstance(node, Option) and (node not in self.option_nodes):
            self.option_nodes.append(node)
        elif isinstance(node, SalientEvent) and (node not in self.salient_nodes):
            self.salient_nodes.append(node)
        else:
            raise IOError(f"Got {node} of type {type(node)}, but expected either Option or SalientEvent")

    def add_edge(self, option1, option2, edge_weight=1.):
        self.plan_graph.add_edge(option1, option2)
        self.set_edge_weight(option1, option2, edge_weight)

    def set_edge_weight(self, option1, option2, weight):
        if self.plan_graph.has_edge(option1, option2):
            self.plan_graph[option1][option2]["weight"] = weight

    def add_skill_chain(self, skill_chain):
        def _get_option_success_rate(o):
            if o.num_test_executions > 0:
                return o.num_successful_test_executions / o.num_test_executions
            return 1.

        # Add the init salient event as a node to the plan-graph
        if skill_chain.init_salient_event not in list(self.plan_graph.nodes):
            self.add_node(skill_chain.init_salient_event)

        # Add the option nodes to the plan graph
        for option in skill_chain.options:  # type: Option
            self.add_node(option)

        # Add connections between the init salient event and all the leaf nodes in the current chain
        for leaf_node in skill_chain.get_leaf_nodes_from_skill_chain():
            self.add_edge(skill_chain.init_salient_event, leaf_node, edge_weight=0.)

        # Add the target salient event as a node to the plan-graph
        if skill_chain.target_salient_event not in list(self.plan_graph.nodes):
            self.add_node(skill_chain.target_salient_event)

        # Add connections between all the root nodes of the current chain and the target salient event
        for root_node in skill_chain.get_root_nodes_from_skill_chain():
            root_option_success_rate = _get_option_success_rate(root_node)
            self.add_edge(root_node, skill_chain.target_salient_event, edge_weight=1./root_option_success_rate)

        for option in skill_chain.options:  # type: Option
            if option.children:
                for child_option in option.children:  # type: Option
                    if child_option is not None:
                        child_option_success_rate = _get_option_success_rate(child_option)
                        self.add_edge(child_option, option, edge_weight=1./child_option_success_rate)

    def construct_graph(self, skill_chains):
        for skill_chain in skill_chains:  # type: SkillChain
            self.add_skill_chain(skill_chain)

        if self.pre_compute_shortest_paths:
            self.shortest_paths = shortest_paths.shortest_path(self.plan_graph)

    def visualize_plan_graph(self, file_name=None):
        try:
            pos = nx.planar_layout(self.plan_graph)
        except nx.NetworkXException:
            pos = nx.random_layout(self.plan_graph)
        labels = nx.get_edge_attributes(self.plan_graph, "weight")

        # Truncate the labels to 2 decimal places
        for key in labels:
            labels[key] = np.round(labels[key], 2)

        plt.figure(figsize=(16, 10))

        nx.draw_networkx(self.plan_graph, pos)
        nx.draw_networkx_edge_labels(self.plan_graph, pos, edge_labels=labels)

        plt.savefig(file_name) if file_name is not None else plt.show()
        plt.close()

    def does_path_exist_between_nodes(self, node1, node2):
        return shortest_paths.has_path(self.plan_graph, node1, node2)

    def does_path_exist_between_state_and_node(self, state, node):
        start_options = self._get_options_node_for_state(state)
        for start_node in start_options:
            if self.does_path_exist_between_nodes(start_node, node):
                return True
        return False

    def get_shortest_path_between_nodes(self, option1, option2):
        if self.pre_compute_shortest_paths:
            return self.shortest_paths[option1][option2], nx.dijkstra_path_length(self.plan_graph, option1, option2)
        return nx.dijkstra_path(self.plan_graph, option1, option2), nx.dijkstra_path_length(self.plan_graph, option1, option2)

    def does_path_exist(self, start_state, goal_state):

        start_options, goal_options = self._get_start_nodes_and_goal_nodes(start_state, goal_state)

        for start_option in start_options:  # type: Option or SalientEvent
            for goal_option in goal_options:  # type: Option or SalientEvent
                if self.does_path_exist_between_nodes(start_option, goal_option):
                    return True
        return False

    def get_shortest_paths(self, start_state, goal_state):
        """
        Return all the shortest paths that go from options that cover the start state
        and options that cover the goal state.
        Args:
            start_state (State): state where plan must start
            goal_state (State): state where plan must end

        Returns:
            paths (list): Shortest paths (in terms of graph with options as nodes) that get you from
                          `start_state` to `goal_state`.
            path_lengths (list): the cost associated with each of the paths in `paths`
        """

        paths = []
        path_lengths = []

        start_options, goal_options = self._get_start_nodes_and_goal_nodes(start_state, goal_state)

        # Get the shortest paths between all pairs of start and goal options
        for start_option in start_options:  # type: Option or SalientEvent
            for goal_option in goal_options:  # type: Option or SalientEvent
                if shortest_paths.has_path(self.plan_graph, start_option, goal_option):
                    path, path_length = self.get_shortest_path_between_nodes(start_option, goal_option)

                    paths.append(path)
                    path_lengths.append(path_length)

        return paths, path_lengths

    def get_path_to_execute(self, start_state, goal_state):
        paths, path_costs = self.get_shortest_paths(start_state, goal_state)

        if not paths:
            return paths

        # Sort all the paths in ascending order of path-costs
        min_cost = min(path_costs)
        paths_with_min_costs = [path for (path, path_cost) in zip(paths, path_costs) if path_cost == min_cost]

        # If there are multiple paths with the same cost, choose the one that has the fewest number of options
        paths_sorted_by_length = sorted(paths_with_min_costs, key=lambda x: len(x))
        path_to_execute = paths_sorted_by_length[0]

        # Filter out all the salient events from the path so that it is actually executable
        option_sequence_to_execute = list(filter(lambda node: isinstance(node, Option), path_to_execute))

        return option_sequence_to_execute

    def get_goal_option(self, start_state, goal_state):
        path_to_execute = self.get_path_to_execute(start_state, goal_state)

        if len(path_to_execute) > 0:
            return path_to_execute[-1]

        return None

    def _get_start_nodes_and_goal_nodes(self, start_state, goal_state):
        start_salient_event = self._get_salient_node_for_state(start_state)
        goal_salient_event = self._get_salient_node_for_state(goal_state)

        # Four possible cases:
        # Case 1: Both states are in some known salient event
        # Case 2: Only start state is in some known salient event
        # Case 3: Only goal state is in some known salient event
        # Case 4: Neither start nor goal states are in known salient events
        if start_salient_event is not None and goal_salient_event is not None:
            return [start_salient_event], [goal_salient_event]
        elif start_salient_event is not None and goal_salient_event is None:
            goal_options = [option for option in self.option_nodes if option.is_term_true(goal_state)]
            return [start_salient_event], goal_options
        elif start_salient_event is None and goal_salient_event is not None:
            start_options = [option for option in self.option_nodes if option.is_init_true(start_state)]
            return start_options, [goal_salient_event]
        else:
            assert start_salient_event is None and goal_salient_event is None
            start_options = [option for option in self.option_nodes if option.is_init_true(start_state)]
            goal_options = [option for option in self.option_nodes if option.is_term_true(goal_state)]
            return start_options, goal_options

    def _get_salient_node_for_state(self, state):
        for salient_node in self.salient_nodes:  # type: SalientEvent
            if salient_node(state):
                return salient_node
        return None

    def _get_options_node_for_state(self, state):
        return [option for option in self.option_nodes if option.is_init_true(state)]
