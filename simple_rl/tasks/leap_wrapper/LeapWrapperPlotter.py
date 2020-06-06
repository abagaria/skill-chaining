import os

import ipdb
import numpy as np
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
        axes_low = [-0.28, 0.3, 0.06, -0.4, 0.2]
        axes_high = [0.28, 0.9, 0.08, 0.4, 1.]
        meshgrid = np.meshgrid(*[np.arange(axis_min, axis_max, 0.02) for axis_min, axis_max in
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
            if option.name == "goal_option_2":
                ipdb.set_trace()
            if (option.get_training_phase() == "initiation" or option.get_training_phase() == "initiation_done") and \
                    option.name != "global_option" and not self.final_initiation_set_has_been_plotted[i]:
                self._plot_initiation_sets(option, episode)

                if option.get_training_phase() == "initiation_done":
                    self.final_initiation_set_has_been_plotted[i] = True
        pass

    def generate_experiment_plots(self, chainer):
        pass

    def _plot_initiation_sets(self, option, episode):
        def _setup_plot():
            # set up figure and axes
            fig, axs = plt.subplots(2, 2, sharex='all', sharey='all', constrained_layout=True)
            fig.set_size_inches(18, 15)
            fig.suptitle(f"{option.name} Initiation Set", size=24)

            # doesn't matter which axis we set these for because sharey and sharex are true
            axs[0, 0].set_xlim(self.puck_x_range)
            axs[0, 0].set_ylim(self.puck_y_range)
            axs[0, 0].set_xticks(np.linspace(self.puck_x_range[0], self.puck_x_range[1], 9))
            axs[0, 0].set_yticks(np.linspace(self.puck_y_range[0], self.puck_y_range[1], 9))

            # plot end effector valid range, puck goal, and target salient event
            for ax in axs.flatten():
                ax.set_aspect("equal")
                # plot box showing the valid moves for the end effector because this is smaller than where
                # the puck can go
                ax.plot(self.endeff_box[:, 0], self.endeff_box[:, 1], color="k", label="endeff bounds")

                # using circle to specify the radius of the goal state (tolerance)
                puck_goal = plt.Circle(self.puck_goal, self.goal_tolerance, color=self.goal_color,
                                       label="puck goal", alpha=0.3)
                ax.add_patch(puck_goal)

                # plot salient event that this option is targeting
                if option.target_salient_event is not None:
                    target_puck_pos = option.target_salient_event.get_target_position()
                    salient_event = plt.Circle(target_puck_pos, self.salient_tolerance, alpha=0.3,
                                               color=self.target_salient_event_color, label="target salient event")
                    ax.add_patch(salient_event)

            # make legend objects
            endeff_box_marker = Line2D([], [], marker="s", markerfacecolor="none", linestyle="none",
                                       color="k", markersize=10, label="end effector bounding box")
            target_salient_marker = Line2D([], [], marker="o", linestyle="none", label="target salient event",
                                           markersize=10, color=self.target_salient_event_color)
            puck_goal_marker = Line2D([], [], marker="*", linestyle="none", color=self.goal_color,
                                      markersize=10, label="puck goal")
            puck_start_marker = Line2D([], [], marker="*", linestyle="none", color="k",
                                       markersize=10, label="puck start")
            endeff_start_marker = Line2D([], [], marker="x", linestyle="none", color="k",
                                         markersize=10, label="end effector start")
            positive_trajectories_marker = Line2D([], [], color=self.positive_color, label="successful trajectories")
            negative_trajectories_marker = Line2D([], [], color=self.negative_color, label="unsuccessful trajectories")

            initiation_set_legend_handles = [endeff_box_marker, target_salient_marker, puck_goal_marker,
                                             puck_start_marker, endeff_start_marker]
            axs[1, 1].legend(handles=initiation_set_legend_handles, loc="upper right")

            trajectory_legend_handles = initiation_set_legend_handles + [positive_trajectories_marker, negative_trajectories_marker]
            axs[0, 1].legend(handles=trajectory_legend_handles, loc="upper right")
            return fig, axs

        def _plot_trajectories(axis, option, x_idx, y_idx, title):
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

        def _plot_initiation_classifier(axis, data, x_idx, y_idx, title):
            x_y, counts = np.unique(data[:, [x_idx, y_idx]], axis=0, return_counts=True)
            ipdb.set_trace()
            axis.scatter(x_y[:, 0], x_y[:, 1], c=counts, cmap=plt.cm.get_cmap("Blues"))
            axis.set_title(f"{title} Initiation Set Classifier", size=16)
            axis.set_xlabel(self.axis_labels[x_idx], size=14)
            axis.set_ylabel(self.axis_labels[y_idx], size=14)

        print(f"Plotting initiation set of {option.name}")
        titles = ['Endeff', 'Puck']
        boolean_mesh = self.mesh[option.batched_is_init_true(self.mesh)]
        # indices for end effector and puck
        indices = [(0, 1), (3, 4)]

        fig, axs = _setup_plot()
        mesh_axes = axs[0]
        trajectory_axes = axs[1]

        for (x_idx, y_idx), mesh_axis, trajectory_axis, title in zip(indices, mesh_axes, trajectory_axes, titles):
            # plot positive and negative examples
            _plot_trajectories(mesh_axis, option, x_idx, y_idx, title)

            # plot initiation classifier using mesh
            _plot_initiation_classifier(trajectory_axis, boolean_mesh, x_idx, y_idx, title)

        # plot the puck and endeff starting positions. Need to do this after plotting points so it isn't covered up by the mesh
        for ax in axs.flatten():
            ax.scatter(self.hand_start[0], self.hand_start[1], color="k", label="endeff start", marker="x", s=100)
            ax.scatter(self.puck_start[0], self.puck_start[1], color="k", label="puck start", marker="*", s=100)

        # save plot as png
        file_name = f"{option.name}_episode_{episode}_{option.seed}.png"
        plt.savefig(os.path.join(self.path, "initiation_set_plots", file_name))
