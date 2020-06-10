import os
import pickle

import ipdb
import time
import numpy as np
import matplotlib.pyplot as plt
import torch
from matplotlib.lines import Line2D

from simple_rl.agents.func_approx.dsc.SkillChainingPlotterClass import SkillChainingPlotter


class LeapWrapperPlotter(SkillChainingPlotter):
    def __init__(self, task_name, experiment_name, mdp):
        # The true start state is: [-0.007, 0.52], but the hand gets to [0.032, 0.409]
        # within a few steps of resetting the environment. This is probably because the
        # arm starts with an initial z position of 0.12, but it goes to 0.07 very quickly.
        # As a result, Reacher might have to slightly adjust the x and y position to drop
        # the z position.
        self.hand_start = (0.032, 0.409)
        self.puck_start = (0., 0.6)
        self.puck_goal = mdp.goal_state[3:]

        # for plotting axes
        self.endeff_box = np.array([[-0.28, 0.3], [-0.28, 0.9], [0.28, 0.9], [0.28, 0.3], [-0.28, 0.3]])
        self.puck_x_range = (-0.4, 0.4)
        self.puck_y_range = (0.2, 1.)
        self.axis_labels = ['endeff_x', 'endeff_y', 'endeff_z', 'puck_x', 'puck_y']

        # grid of points to plot decision classifiers
        axes_low = [-0.28, 0.3, 0.07, -0.4, 0.2]
        axes_high = [0.28, 0.9, 0.07, 0.4, 1.]
        mesh_axes = [np.arange(axis_min, axis_max, 0.01) for axis_min, axis_max in zip(axes_low, axes_high)]
        mesh_axes[2] = [[0.07]]
        self.mesh = np.column_stack(list(map(np.ravel, np.meshgrid(*mesh_axes, indexing="ij"))))

        # for pmeshgrid
        # step = 0.01
        # meshgrid = [np.arange(axis_min, axis_max + step, step) for axis_min, axis_max in zip(axes_low, axes_high)]
        # meshgrid[2] = [[0.07]]

        # Tolerance of being within goal state or salient events. This is used to plot the
        # radius of the goal and salient events
        self.goal_tolerance = mdp.goal_tolerance
        self.salient_tolerance = mdp.salient_tolerance

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
            self._plot_value_function(option.solver, chainer.seed, episode)
            if (option.get_training_phase() == "initiation" or option.get_training_phase() == "initiation_done") and \
                    option.name != "global_option" and not self.final_initiation_set_has_been_plotted[i]:
                self._plot_initiation_sets(option, episode)

                if option.get_training_phase() == "initiation_done":
                    self.final_initiation_set_has_been_plotted[i] = True

    def _plot_value_function(self, solver, seed, episode):
        t_0 = time.perf_counter()
        CHUNK_SIZE = 250

        # Chunk up the inputs so as to conserve GPU memory
        num_chunks = int(np.ceil(self.mesh.shape[0] / CHUNK_SIZE))
        state_chunks = np.array_split(self.mesh, num_chunks, axis=0)
        values = np.zeros((self.mesh.shape[0],))
        current_idx = 0
        actions = [[-1.0, 0.], [1.0, 0.], [0., -1.0], [0., 1.0]]

        # get values for each state
        for chunk_number, state_chunk in enumerate(state_chunks):
            current_chunk_size = len(state_chunk)
            repeated_states = np.repeat(state_chunk, 4, axis=0)
            repeated_actions = np.tile(actions, (current_chunk_size, 1))
            state_chunk = torch.from_numpy(repeated_states).float().to(solver.device)
            action_chunk = torch.from_numpy(repeated_actions).float().to(solver.device)
            # To get value function from q values, take the argmax across moving up, left, right, and down.
            chunk_values = np.amax(solver.get_qvalues(state_chunk, action_chunk).cpu().numpy().squeeze(1).reshape(-1, 4), axis=1)
            values[current_idx:current_idx + current_chunk_size] = chunk_values
            current_idx += current_chunk_size
        t_1 = time.perf_counter()

        titles = ['Endeff', 'Puck']
        fig, axs = self._setup_plot((1, 2))
        fig.suptitle(f"{solver.name} Value Function Plot")

        # plot endeff pos in left graph and puck pos in right graph
        for (x_idx, y_idx), ax, title in zip(((0, 1), (3, 4)), axs, titles):
            # get average qvalue for each state along unique endeff pos or unique puck pos
            unq_states, idx, cnt = np.unique(self.mesh[:, (x_idx, y_idx)], return_inverse=True, return_counts=True, axis=0)
            avg_qvalue = np.bincount(idx, weights=values) / cnt
            ax.scatter(unq_states[:, 0], unq_states[:, 1], c=avg_qvalue, cmap=plt.cm.get_cmap("Blues"))
            ax.set_title(f"{title} Initiation Set Classifier", size=16)
            ax.set_xlabel(self.axis_labels[x_idx], size=14)
            ax.set_ylabel(self.axis_labels[y_idx], size=14)

        t_2 = time.perf_counter()
        # plt.colorbar()
        file_name = f"{solver.name}_value_function_seed_{seed}_episode_{episode}.png"
        plt.savefig(os.path.join(self.path, "value_function_plots", file_name))
        plt.close()
        t_3 = time.perf_counter()
        print(t_1 - t_0, t_2 - t_1, t_3 - t_2)

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
        self._add_legend(axs[0, 1], trajectories=trajectories, target_salient=has_salient_target)
        self._add_legend(axs[1, 1], target_salient=has_salient_target)

        # save plot as png
        file_name = f"{option.name}_episode_{episode}_{option.seed}.png"
        plt.savefig(os.path.join(self.path, "initiation_set_plots", file_name))
        plt.close()

    def _setup_plot(self, shape):
        GRAPH_WIDTH = 6
        # set up figure and axes
        fig, axs = plt.subplots(shape[0], shape[1], sharex='all', sharey='all', constrained_layout=True)
        fig.set_size_inches(shape[0] * GRAPH_WIDTH, shape[1] * GRAPH_WIDTH)

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

    def _add_legend(self, ax, target_salient=False, trajectories="none"):
        """
        Adds a legend to `ax`.
        Args:
            ax: axis created from plt.subplots() to add legend to
            target_salient: if True, add a legend marker for target salient events
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
