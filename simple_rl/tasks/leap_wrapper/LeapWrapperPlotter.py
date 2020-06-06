import os
import numpy as np
import ipdb
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from simple_rl.agents.func_approx.dsc.SkillChainingPlotterClass import SkillChainingPlotter


class LeapWrapperPlotter(SkillChainingPlotter):
    def __init__(self, task_name, experiment_name):
        # The true start state is: [-0.007, 0.52], but the hand gets to [0.032, 0.409]
        # within a few steps of resetting the environment. This is probably because the
        # arm starts with an initial z position of 0.12, but it goes to 0.07 very quickly.
        # As a result, Reacher might have to slightly adjust the x and y position to drop
        # the z position.
        self.hand_start = (0.032, 0.409)
        self.puck_start = (0., 0.6)
        self.puck_goal = (-0.15, 0.6)

        # for plotting axes
        self.endeff_box = np.array([[-0.28, 0.3], [-0.28, 0.9], [0.28, 0.9], [0.28, 0.3], [-0.28, 0.3]])
        self.puck_x_range = (-0.4, 0.4)
        self.puck_y_range = (0.2, 1.)
        self.axis_labels = ['endeff_x', 'endeff_y', 'endeff_z', 'puck_x', 'puck_y']

        # grid of points to plot decision classifiers
        axes_low = [-0.28, 0.3, 0.065, -0.4, 0.2]
        axes_high = [0.28, 0.9, 0.075, 0.4, 1.]
        meshgrid = np.meshgrid(*[np.arange(axis_min, axis_max, 0.01) for axis_min, axis_max in
                                 zip(axes_low, axes_high)], indexing="ij")
        self.mesh = np.column_stack(list(map(np.ravel, meshgrid)))

        # Tolerance of being within goal state or salient events. This is used to plot the
        # radius of the goal and salient events
        self.goal_tolerance = 0.03
        self.salient_tolerance = 0.03

        # only want to plot the final initiation set of each option once
        self.final_initiation_set_has_been_plotted = []

        self.positive_color = "b"
        self.negative_color = "r"
        self.target_salient_event_color = "green"
        self.goal_color = "orange"
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
            if (option.get_training_phase() == "initiation" or option.get_training_phase() == "initiation_done") and \
                    option.name != "global_option" and not self.final_initiation_set_has_been_plotted[i]:
                self._plot_initiation_sets(option, episode)

                if option.get_training_phase() == "initiation_done":
                    self.final_initiation_set_has_been_plotted[i] = True
        pass

    def generate_experiment_plots(self, chainer):
        pass

    def _plot_value_function(self, option, episode):
        x = 1

    def _plot_initiation_sets(self, option, episode):
        def _plot_trajectories(axis):
            positive_trajectories = option.positive_examples
            negative_trajectories = option.negative_examples
            for positive_trajectory in positive_trajectories:
                positive_trajectory = np.array(positive_trajectory)
                axis.plot(positive_trajectory[:, x_idx], positive_trajectory[:, y_idx], label="positive", c=self.positive_color)
            for negative_trajectory in negative_trajectories:
                negative_trajectory = np.array(negative_trajectory)
                axis.plot(negative_trajectory[:, x_idx], negative_trajectory[:, y_idx], label="negative", c=self.negative_color)
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

        fig, axs = self._setup_plot(option)
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
        self._add_legend(axs[0, 1], trajectories=trajectories, target_salient=has_salient_target)
        self._add_legend(axs[1, 1], trajectories=trajectories, target_salient=has_salient_target)

        # save plot as png
        file_name = f"{option.name}_episode_{episode}_{option.seed}.png"
        plt.savefig(os.path.join(self.path, "initiation_set_plots", file_name))

    def _setup_plot(self, option):
        # set up figure and axes
        fig, axs = plt.subplots(2, 2, sharex='all', sharey='all', constrained_layout=True)
        fig.set_size_inches(18, 15)
        fig.suptitle(f"{option.name} Initiation Set", size=24)

        # doesn't matter which axis we set these for because sharey and sharex are true
        axs[0, 0].set_xlim(self.puck_x_range)
        axs[0, 0].set_ylim(self.puck_y_range)
        axs[0, 0].set_xticks(np.linspace(self.puck_x_range[0], self.puck_x_range[1], 9))
        axs[0, 0].set_yticks(np.linspace(self.puck_y_range[0], self.puck_y_range[1], 9))

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
        ax.scatter(self.puck_start[0], self.puck_start[1], color="k", label="puck start", marker="*", s=100)

    def _add_legend(self, ax, target_salient=False, trajectories="none"):
        """
        Adds a legend to `ax`/
        Args:
            ax:
            target_salient:
            trajectories:

        Returns:

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

        if target_salient:
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
