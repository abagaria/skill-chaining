import os
import pickle

import ipdb
import time
import numpy as np
import matplotlib.pyplot as plt
import torch
from matplotlib.lines import Line2D
import matplotlib.cm as cm
from matplotlib.colors import Normalize

from simple_rl.agents.func_approx.dsc.SkillChainingPlotterClass import SkillChainingPlotter


class LeapWrapperPlotter(SkillChainingPlotter):
    def __init__(self, task_name, experiment_name, mdp):
        print("Setting up Plotter")
        # The true start state is: [-0.007, 0.52], but the hand gets to [0.032, 0.409]
        # within a few steps of resetting the environment. This is probably because the
        # arm starts with an initial z position of 0.12, but it goes to 0.07 very quickly.
        # As a result, Reacher might have to slightly adjust the x and y position to drop
        # the z position.
        self.hand_start = (0.032, 0.409)
        self.puck_start = (0., 0.6)
        self.puck_goal = mdp.goal_state[3:]

        # bounds of possible endeff positions (smaller than possible puck positions)
        self.endeff_box = np.array([[-0.28, 0.3], [-0.28, 0.9], [0.28, 0.9], [0.28, 0.3], [-0.28, 0.3]])
        self.puck_x_range = (-0.4, 0.4)
        self.puck_y_range = (0.2, 1.)
        self.axis_labels = ['endeff_x', 'endeff_y', 'endeff_z', 'puck_x', 'puck_y']

        # Tolerance of being within goal state or salient events. This is used to plot the
        # radius of the goal and salient events
        self.goal_tolerance = mdp.goal_tolerance
        self.salient_tolerance = mdp.salient_tolerance

        # colors used for plotting sawyer features
        self.positive_color = "b"
        self.negative_color = "r"
        self.target_salient_event_color = "green"
        self.goal_color = "orange"

        # Used to plot pcolormesh for value function and initiation set plots. This is costly
        # to compute, so do it once when initializing the plotter.
        self.center_points, self.endeff_grid, self.puck_grid = self._get_endeff_puck_grids()

        # used when calculating average of value function when grouping by puck pos or endeff pos
        self.endeff_idx, self.endeff_cnt = self._setup_unique_weighted_average((0, 1))
        self.puck_idx, self.puck_cnt = self._setup_unique_weighted_average((3, 4))

        # We only want to plot the final initiation set of each option once. This array will
        # have one boolean entry for each option: True if that option's final initiation set
        # has already been plotted, false otherwise.
        self.final_initiation_set_has_been_plotted = []

        super().__init__(task_name, experiment_name, ["initiation_set_plots", "value_function_plots"])

    def generate_episode_plots(self, chainer, episode):
        """
        Args:
            chainer (SkillChainingAgent): the skill chaining agent we want to plot
            episode (int)
        """
        # only want to plot the final initiation set of each option once
        while len(self.final_initiation_set_has_been_plotted) < len(chainer.trained_options):
            self.final_initiation_set_has_been_plotted.append(False)

        for i, option in enumerate(chainer.trained_options):
            self._plot_value_function(option, chainer.seed, episode)
            if (option.get_training_phase() == "initiation" or option.get_training_phase() == "initiation_done") and \
                    option.name != "global_option" and not self.final_initiation_set_has_been_plotted[i]:
                self._plot_initiation_sets(option, episode)

                if option.get_training_phase() == "initiation_done":
                    self.final_initiation_set_has_been_plotted[i] = True

    def get_value_function_values(self, solver):
        CHUNK_SIZE = 250

        # Chunk up the inputs so as to conserve GPU memory
        num_chunks = int(np.ceil(self.center_points.shape[0] / CHUNK_SIZE))
        state_chunks = np.array_split(self.center_points, num_chunks, axis=0)
        values = np.zeros((self.center_points.shape[0],))
        current_idx = 0
        actions = [[-1.0, 0.], [1.0, 0.], [0., -1.0], [0., 1.0]]

        # get values for each state
        for chunk_number, state_chunk in enumerate(state_chunks):
            # To get from Q(s, a) to V(s), we take the argmax across actions. This is a continuous
            # state space, so we are just taking the argmax across going left, right, up, or down.
            current_chunk_size = len(state_chunk)
            repeated_states = np.repeat(state_chunk, 4, axis=0)
            repeated_actions = np.tile(actions, (current_chunk_size, 1))
            state_chunk = torch.from_numpy(repeated_states).float().to(solver.device)
            action_chunk = torch.from_numpy(repeated_actions).float().to(solver.device)
            # argmax across actions
            chunk_values = np.amax(solver.get_qvalues(state_chunk, action_chunk).cpu().numpy().squeeze(1).reshape(-1, 4), axis=1)
            values[current_idx:current_idx + current_chunk_size] = chunk_values
            current_idx += current_chunk_size

        return values

    def _plot_value_function(self, option, seed, episode):
        """
        It is difficult to plot our state space because it is 5-D. This method
        makes two separate graphs to plot value function: one for puck position and
        one for endeff position. In the puck position graph, each point is the
        average of all states with that puck position. For example, there are many
        endeff states in which the puck position is at (0, 0.6)).
        Args:
            option:
            seed:
            episode:

        Returns:
            None
        """
        print(f"Plotting {option.name} value function")
        solver = option.solver
        v = self.get_value_function_values(solver)
        endeff_z = self._average_groupby_puck_or_endeff_pos("endeff", v)
        puck_z = self._average_groupby_puck_or_endeff_pos("puck", v)
        vmin = min(np.amin(endeff_z), np.amin(puck_z))
        vmax = max(np.amax(endeff_z), np.amax(puck_z))
        norm = Normalize(vmin=vmin, vmax=vmax)
        cmap = "inferno"

        fig, (ax1, ax2) = self._setup_plot((1, 2))
        fig.suptitle(f"{solver.name} Value Function Plot")
        fig.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), ax=(ax1, ax2), aspect=40)
        self._add_legend(ax2, option)

        # plot value function with respect to endeff pos
        ax1.pcolormesh(self.endeff_grid[0], self.endeff_grid[1], endeff_z, norm=norm, cmap=cmap)
        ax1.set_title("Endeff Value Function", size=16)
        ax1.set_xlabel(self.axis_labels[0], size=14)
        ax1.set_ylabel(self.axis_labels[1], size=14)
        self._plot_sawyer_features(ax1, option)

        # plot value function with respect to puck pos
        ax2.pcolormesh(self.puck_grid[0], self.puck_grid[1], puck_z, norm=norm, cmap=cmap)
        ax2.set_title("Puck Value Function", size=16)
        ax2.set_xlabel(self.axis_labels[3], size=14)
        ax2.set_ylabel(self.axis_labels[4], size=14)
        self._plot_sawyer_features(ax2, option)

        file_name = f"{solver.name}_value_function_seed_{seed}_episode_{episode}.png"
        plt.savefig(os.path.join(self.path, "value_function_plots", file_name))
        plt.close()

    def _plot_initiation_sets(self, option, episode):
        def _plot_trajectories(axis):
            positive_trajectories = option.positive_examples
            negative_trajectories = option.negative_examples
            for positive_trajectory in positive_trajectories:
                positive_trajectory = np.array(positive_trajectory)
                axis.plot(positive_trajectory[:, x_idx], positive_trajectory[:, y_idx],
                          label="positive", c=self.positive_color, alpha=0.5, linewidth=1.5)
            for negative_trajectory in negative_trajectories:
                negative_trajectory = np.array(negative_trajectory)
                axis.plot(negative_trajectory[:, x_idx], negative_trajectory[:, y_idx],
                          label="negative", c=self.negative_color, alpha=0.5, linewidth=1.5)
            axis.set_title(f"{title} Trajectories", size=16)
            axis.set_xlabel(self.axis_labels[x_idx], size=14)
            axis.set_ylabel(self.axis_labels[y_idx], size=14)

        def _plot_initiation_classifier(axis, data):
            x_y, counts = np.unique(data[:, [x_idx, y_idx]], axis=0, return_counts=True)
            axis.scatter(x_y[:, 0], x_y[:, 1], c=counts, cmap=plt.cm.get_cmap("Blues"))
            axis.set_title(f"{title} Initiation Set Classifier", size=16)
            axis.set_xlabel(self.axis_labels[x_idx], size=14)
            axis.set_ylabel(self.axis_labels[y_idx], size=14)

        print(f"Plotting initiation set of {option.name}")

        # indices for end effector and puck
        indices = [(0, 1), (3, 4)]
        titles = ['Endeff', 'Puck']
        boolean_mesh = self.mesh[option.batched_is_init_true(self.mesh)]

        fig, axs = self._setup_plot((2, 2))
        fig.suptitle(f"{option.name} Initiation Set", size=24)
        mesh_axes = axs[0]
        trajectory_axes = axs[1]

        for (x_idx, y_idx), mesh_axis, trajectory_axis, title in zip(indices, mesh_axes, trajectory_axes, titles):
            # plot positive and negative examples
            _plot_trajectories(mesh_axis)

            # plot initiation classifier using mesh
            _plot_initiation_classifier(trajectory_axis, boolean_mesh)

        # plot end effector bounds, end effector start, puck start, puck goal, and option target
        for ax in axs.flatten():
            self._plot_sawyer_features(ax, option)

        # plot legend
        trajectories = "all" if len(option.negative_examples) > 0 else "positive"
        has_salient_target = option.target_salient_event is not None
        self._add_legend(axs[0, 1], option, trajectories=trajectories)
        self._add_legend(axs[1, 1], option)

        # save plot as png
        file_name = f"{option.name}_episode_{episode}_{option.seed}.png"
        plt.savefig(os.path.join(self.path, "initiation_set_plots", file_name))
        plt.close()

    def _setup_plot(self, shape):
        GRAPH_WIDTH = 6
        # set up figure and axes
        fig, axs = plt.subplots(shape[0], shape[1], sharex='all', sharey='all', constrained_layout=True)
        fig.set_size_inches(shape[1] * GRAPH_WIDTH, shape[0] * GRAPH_WIDTH)

        # doesn't matter which axis we set these for because sharey and sharex are true
        ax = axs.flat[0]
        ax.set_xlim(self.puck_x_range)
        ax.set_ylim(self.puck_y_range)
        ax.set_xticks(np.linspace(self.puck_x_range[0], self.puck_x_range[1], 9))
        ax.set_yticks(np.linspace(self.puck_y_range[0], self.puck_y_range[1], 9))

        for ax in axs.flatten():
            ax.set_aspect("equal")
        return fig, axs

    def _plot_sawyer_features(self, ax, option=None):
        """Plots the different Sawyer environment features.

        Plots the end effector box, puck goal (as a circle with radius = goal_tolerance),
        puck start, and end effector start states on `ax`.
        Args:
            ax: Axis generated by plt.subplots
            option (Option): if option is not None, will try to plot the target_salient_event
            of the option.

        Returns:
            None
        """
        # plot box showing the valid moves for the end effector because this is smaller than where
        # the puck can go
        ax.plot(self.endeff_box[:, 0], self.endeff_box[:, 1], color="k", label="endeff bounds")

        # using circle instead of scatter plot because we need to specify the radius of the
        # goal state (tolerance) in plot units
        puck_goal = plt.Circle(self.puck_goal, self.goal_tolerance, color=self.goal_color,
                               label="puck goal", alpha=0.3)
        ax.add_patch(puck_goal)

        # plot salient event that this option is targeting
        if option is not None and option.target_salient_event is not None:
            target_puck_pos = option.target_salient_event.get_target_position()
            salient_event = plt.Circle(target_puck_pos, self.salient_tolerance, alpha=0.3,
                                       color=self.target_salient_event_color, label="target salient event")
            ax.add_patch(salient_event)

        # plot the puck and endeff starting positions.
        ax.scatter(self.hand_start[0], self.hand_start[1], color="k", label="endeff start", marker="x", s=180)
        ax.scatter(self.puck_start[0], self.puck_start[1], color="k", label="puck start", marker="*", s=400)

    def _add_legend(self, ax, option, trajectories="none"):
        """
        Adds a legend to `ax`.
        Args:
            ax: axis created from plt.subplots() to add legend to
            option: if option.target_salient_event is not None, add a legend marker for target salient events
            trajectories:
                "all" - add legend marker for positive and negative trajectories.
                "positive" - add legend marker for positive trajectories only
                "none" - don't add legend marker for trajectories

        Returns:
            None
        """
        endeff_box_marker = Line2D([], [], marker="s", markerfacecolor="none", linestyle="none",
                                   color="k", markersize=12, label="end effector bounding box")
        puck_goal_marker = Line2D([], [], marker="o", linestyle="none", color=self.goal_color,
                                  markersize=12, label="puck goal")
        puck_start_marker = Line2D([], [], marker="*", linestyle="none", color="k",
                                   markersize=12, label="puck start")
        endeff_start_marker = Line2D([], [], marker="x", linestyle="none", color="k",
                                     markersize=12, label="end effector start")

        handles = [endeff_box_marker, puck_goal_marker, puck_start_marker, endeff_start_marker]

        if option.target_salient_event is not None:
            target_salient_marker = Line2D([], [], marker="o", linestyle="none", label="target salient event",
                                           markersize=12, color=self.target_salient_event_color)
            handles.append(target_salient_marker)

        if trajectories == "positive" or trajectories == "all":
            positive_trajectories_marker = Line2D([], [], color=self.positive_color, linewidth=2.5, label="successful trajectories")
            handles.append(positive_trajectories_marker)
            if trajectories == "all":
                negative_trajectories_marker = Line2D([], [], color=self.negative_color, linewidth=2.5, label="unsuccessful trajectories")
                handles.append(negative_trajectories_marker)
        elif trajectories != "none":
            raise NotImplementedError(f"Invalid trajectories type: {trajectories}")

        ax.legend(handles=handles, loc="upper right")

    def _setup_unique_weighted_average(self, indices):
        """
        These constants are useful for calculating the average for unique values in a list in
        `_average_groupby_puck_or_endeff_pos`
        Args:
            indices: (0, 1) for endeff position x-y position and (3, 4) for puck x-y position

        Returns:
            inverse_indices and number of times each unique value is repeated from np.unique
        """
        _, idx, cnt = np.unique(self.center_points[:, indices], return_inverse=True, return_counts=True, axis=0)
        # converting to int16 to save memory because this is a large array
        return idx.astype(np.int16), cnt

    def _average_groupby_puck_or_endeff_pos(self, puck_or_endeff, values):
        """
        Calculate the average of values along the grouped puck pos or endeff pos. From:
        https://stackoverflow.com/questions/29243982/average-using-grouping-value-in-another-vector-numpy-python
        Args:
            puck_or_endeff:
            values: values to be grouped by

        Returns:
            Average of `values` grouped by either puck pos or endeff pos. This will be the z-values
            plotted in plt.pcolormesh.
        """
        if puck_or_endeff.lower() == "endeff":
            avg = np.bincount(self.endeff_idx, weights=values) / self.endeff_cnt
            num_buckets_along_y = self.endeff_grid[0].shape[1] - 1
        elif puck_or_endeff.lower() == "puck":
            avg = np.bincount(self.puck_idx, weights=values) / self.puck_cnt
            num_buckets_along_y = self.puck_grid[0].shape[1] - 1
        else:
            raise NotImplementedError("Invalid input for `puck_or_endeff`.")

        # we need to take the transpose because pcolormesh traverses along x but our
        # idx and cnt traverse along y because np.unique sorts lexographically
        return avg.reshape((num_buckets_along_y, -1)).T

    @staticmethod
    def _get_endeff_puck_grids(step=0.02):
        """
        Make meshgrids (boundary rectangles) for plt.pcolormesh and get the center point of each rectangle.
        Args:
            step: making this smaller improves resolution but hurts runtime/memory

        Returns:
            center_points, endeff_grid, puck_grid
        """
        axes_low = [-0.28, 0.3, 0.07, -0.4, 0.2]
        axes_high = [0.28, 0.9, 0.07, 0.4, 1.]

        # use np.meshgrid to make mesh of state space. Only need to return 2D meshes for graphing endeff
        # pos and puck pos.
        mesh_axes = [np.arange(axis_min, axis_max, step) for axis_min, axis_max in zip(axes_low, axes_high)]
        endeff_grid = np.meshgrid(*mesh_axes[:2])
        puck_grid = np.meshgrid(*mesh_axes[3:])

        # we need to calculate the center point of each rectangle to get the color of each rectangle (Z dimension)
        center_points = np.array(mesh_axes) + step / 2
        center_points = [center_points[0][:-1], center_points[1][:-1], [0.07], center_points[3][:-1], center_points[4][:-1]]
        center_points = np.column_stack(list(map(np.ravel, np.meshgrid(*center_points))))

        return center_points, endeff_grid, puck_grid
