import networkx as nx
import matplotlib.pyplot as plt
import networkx.algorithms.shortest_paths as shortest_paths
from simple_rl.agents.func_approx.dsc.OptionClass import Option
from simple_rl.agents.func_approx.dsc.ChainClass import SkillChain


class GraphSearch(object):
    def __init__(self, pre_compute_shortest_paths=True):
        self.plan_graph = nx.DiGraph()
        self.pre_compute_shortest_paths = pre_compute_shortest_paths
        self.shortest_paths = {}

    def add_node(self, option):
        self.plan_graph.add_node(option)

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

        for option in skill_chain.options:  # type: Option
            self.add_node(option)

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
        pos = nx.planar_layout(self.plan_graph)
        labels = nx.get_edge_attributes(self.plan_graph, "weight")

        # Truncate the labels to 2 decimal places
        for key in labels:
            labels[key] = np.round(labels[key], 2)

        nx.draw_networkx(self.plan_graph, pos)
        nx.draw_networkx_edge_labels(self.plan_graph, pos, edge_labels=labels)

        plt.savefig(file_name) if file_name is not None else plt.show()

    def does_path_exist(self, start_state, goal_state, use_init_predicate=False):
        def _does_path_exist(option1, option2):
            return shortest_paths.has_path(self.plan_graph, option1, option2)

        if use_init_predicate:
            start_options = [option for option in self.plan_graph.nodes if option.init_predicate(start_state)]
            goal_options = [option for option in self.plan_graph.nodes if option.init_predicate(goal_state)]
        else:
            start_options = [option for option in self.plan_graph.nodes if option.is_init_true(start_state)]
            goal_options = [option for option in self.plan_graph.nodes if option.is_term_true(goal_state)]

        for start_option in start_options:  # type: Option
            for goal_option in goal_options:  # type: Option
                if _does_path_exist(start_option, goal_option):
                    return True
        return False

    def get_shortest_paths(self, start_state, goal_state, use_init_predicate=False):
        """
        Return all the shortest paths that go from options that cover the start state
        and options that cover the goal state.
        Args:
            start_state (State): state where plan must start
            goal_state (State): state where plan must end
            use_init_predicate (bool): Mainly set for testing, used if option doesn't have an is_init_true

        Returns:
            paths (list): Shortest paths (in terms of graph with options as nodes) that get you from
                          `start_state` to `goal_state`.
        """

        def _get_shortest_path(option1, option2):
            if self.pre_compute_shortest_paths:
                return self.shortest_paths[option1][option2]
            return shortest_paths.shortest_path(self.plan_graph, option1, option2)

        paths = []

        if use_init_predicate:
            start_options = [option for option in self.plan_graph.nodes if option.init_predicate(start_state)]
            goal_options = [option for option in self.plan_graph.nodes if option.init_predicate(goal_state)]
        else:
            start_options = [option for option in self.plan_graph.nodes if option.is_init_true(start_state)]
            goal_options = [option for option in self.plan_graph.nodes if option.is_term_true(goal_state)]

        # Get the shortest paths between all pairs of start and goal options
        for start_option in start_options:  # type: Option
            for goal_option in goal_options:  # type: Option
                if shortest_paths.has_path(self.plan_graph, start_option, goal_option):
                        paths.append(_get_shortest_path(start_option, goal_option))

        return paths

    def get_path_to_execute(self, start_state, goal_state):
        paths = self.get_shortest_paths(start_state, goal_state)

        if not paths:
            return paths

        # TODO: Hack - currently just picking the plan with the largest length
        # TODO: We need to come up with a method of picking the current plan in general
        paths_sorted_by_length = sorted(paths, key=lambda x: len(x))

        return paths_sorted_by_length[-1]

    def get_goal_option(self, start_state, goal_state):
        path_to_execute = self.get_path_to_execute(start_state, goal_state)

        if len(path_to_execute) > 0:
            return path_to_execute[-1]

        return None


if __name__ == "__main__":
    from simple_rl.tasks.point_reacher.PointReacherMDPClass import PointReacherMDP
    import numpy as np

    mdp = PointReacherMDP(0, render=False)
    o1 = Option(mdp, "o1", None, 0., 0., 0.)
    o2 = Option(mdp, "o2", None, 0., 0., 0.)
    o3 = Option(mdp, "o3", None, 0., 0., 0.)
    o4 = Option(mdp, "o4", None, 0., 0., 0.)
    o1.children = [o2, o3]
    o2.children = [o3]
    c1 = SkillChain([], [], None, [o1, o2, o3], 1)
    g = GraphSearch()
    g.add_skill_chain(c1)
    g.add_node(o4)
    g.visualize_plan_graph()
    plt.show()

    s0 = np.array((0., 0.))
    sg = np.array((10., 10.))
    sm = np.array((5., 5.))
    sn = np.array((7., 7.))

    o1.init_predicate = lambda s: np.linalg.norm(s - s0) <= 0.1
    o2.init_predicate = lambda s: np.linalg.norm(s - sm) <= 0.1
    o3.init_predicate = lambda s: np.linalg.norm(s - sg) <= 0.1
    o4.init_predicate = lambda s: np.linalg.norm(s - sn) <= 0.1

    f1 = g.does_path_exist(s0, sg, use_init_predicate=True)  # True: o1 -> o2 -> o3, o1 -> o3
    f2 = g.does_path_exist(s0, sm, use_init_predicate=True)  # True: o1 -> o2
    f3 = g.does_path_exist(s0, sn, use_init_predicate=True)  # False

    p1 = g.get_shortest_paths(s0, sg, use_init_predicate=True)  # o1 -> o3
    p2 = g.get_shortest_paths(s0, sm, use_init_predicate=True)  # o1 -> o2
    p3 = g.get_shortest_paths(s0, sn, use_init_predicate=True)  # []
