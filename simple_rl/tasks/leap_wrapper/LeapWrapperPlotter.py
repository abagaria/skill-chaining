import os
import numpy as np
import matplotlib.pyplot as plt
import torch
from matplotlib.lines import Line2D
import matplotlib.cm as cm
from matplotlib.colors import Normalize

from simple_rl.mdp.MDPPlotterClass import MDPPlotter


class LeapWrapperPlotter(MDPPlotter):
    def __init__(self, task_name, experiment_name, mdp):
        print("Setting up Plotter")
        # The true start state is: [-0.007, 0.52], but the hand gets to [0.032, 0.409]
        # within a few steps of resetting the environment. This is probably because the
        # arm starts with an initial z position of 0.12, but it goes to 0.07 very quickly.
        # As a result, Reacher might have to slightly adjust the x and y position to drop
        # the z position.
        self.true_hand_start = (-0.007, 0.52)
        self.effective_hand_start = (0.032, 0.409)
        self.puck_start = (0., 0.6)
        self.puck_goal = mdp.goal_state[3:]

        # bounds of possible endeff positions (smaller than possible puck positions)
        self.axis_x_range = (-0.28, 0.28)
        self.axis_y_range = (0.3, 0.9)
        self.axis_labels = ['endeff_x', 'endeff_y', 'endeff_z', 'puck_x', 'puck_y']

        # Tolerance of being within goal state or salient events. This is used to plot the
        # radius of the goal and salient events
        self.goal_tolerance = mdp.goal_tolerance
        self.salient_tolerance = mdp.salient_tolerance
        self.puck_radius = mdp.env.puck_radius

        # only plot the overall mdp goal state if the task is not goal agnostic
        self.task_agnostic = mdp.task_agnostic

        # colors for plotting sawyer features
        self.positive_color = "b"
        self.negative_color = "r"
        self.target_salient_event_color = "green"
        self.goal_color = "orange"
        # matches env puck color
        self.puck_color = '#354ad4'

        # Used to plot pcolormesh for value function and initiation set plots. This is costly
        # to compute, so do it once when initializing the plotter.
        self.puck_positions_to_plot = [(-0.14, 0.7), (-0.1, 0.6), (0, 0.75), (-0.14, 0.5), (0.05, 0.6), (0, 0.45)]
        self.center_points, self.grid_boundaries = self._get_endeff_puck_grids()
        self.arrow_points, self.arrow_mesh = self._get_arrow_points()

        # used when calculating average of value function when grouping by puck pos or endeff pos
        self.endeff_idx, self.endeff_cnt = self._setup_unique_weighted_average((0, 1))
        self.puck_idx, self.puck_cnt = self._setup_unique_weighted_average((3, 4))

        # We only want to plot the final initiation set of each option once. This array will
        # have one boolean entry for each option: True if that option's final initiation set
        # has already been plotted, false otherwise.
        self.final_initiation_set_has_been_plotted = []

        super().__init__(task_name, experiment_name,
                         ["initiation_set_plots", "value_function_plots", "option_policy_plots"])

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
            self._plot_option_policy(option, chainer.seed, episode)

            # if (option.get_training_phase() == "initiation" or option.get_training_phase() == "initiation_done") and \
            #         option.name != "global_option" and not self.final_initiation_set_has_been_plotted[i]:
            self._plot_initiation_sets(option, episode)

            if option.get_training_phase() == "initiation_done":
                self.final_initiation_set_has_been_plotted[i] = True

    def _plot_option_policy(self, option, seed, episode):
        print(f"plotting {option.name}'s policy")

        fig, axs = self._setup_plot((2, 3))
        fig.suptitle(f"{option.solver.name} Policy")
        axs = axs.flatten()
        self._add_legend(axs[2], include_puck=True)

        for i, ax in enumerate(axs):
            # set up axis
            ax.set_xlabel(self.axis_labels[0], size=14)
            ax.set_ylabel(self.axis_labels[1], size=14)

            # get policy data from option solver
            arrow_points = self.arrow_points[i]
            vectors = np.array([option.solver.act(arrow, evaluation_mode=True) for arrow in arrow_points])

            # plot quiver diagram
            ax.quiver(self.arrow_mesh[0], self.arrow_mesh[1], vectors[:, 0], vectors[:, 1], headlength=4, headwidth=2.6)
            self._plot_sawyer_features(ax, puck_pos=arrow_points[0][3:])

        # save plot
        file_name = f"{option.solver.name}_policy_seed_{seed}_episode_{episode}.png"
        plt.savefig(os.path.join(self.path, "option_policy_plots", file_name))
        plt.close()

    def _get_arrow_points(self):
        arm_x = np.linspace(self.axis_x_range[0], self.axis_x_range[1], 10)[1:-1]
        arm_y = np.linspace(self.axis_y_range[0], self.axis_y_range[1], 10)[1:-1]
        arm_z = [0.07]

        meshes = [np.meshgrid(arm_x, arm_y, arm_z, puck_x, puck_y) for (puck_x, puck_y) in self.puck_positions_to_plot]
        return [np.column_stack(list(map(np.ravel, mesh))) for mesh in meshes], np.array(meshes[0])

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
        ax1.pcolormesh(self.grid_boundaries[0], self.grid_boundaries[1], endeff_z, norm=norm, cmap=cmap)
        ax1.set_title("Endeff Value Function", size=16)
        ax1.set_xlabel(self.axis_labels[0], size=14)
        ax1.set_ylabel(self.axis_labels[1], size=14)
        self._plot_sawyer_features(ax1, option=option)

        # plot value function with respect to puck pos
        ax2.pcolormesh(self.grid_boundaries[0], self.grid_boundaries[1], puck_z, norm=norm, cmap=cmap)
        ax2.set_title("Puck Value Function", size=16)
        ax2.set_xlabel(self.axis_labels[3], size=14)
        ax2.set_ylabel(self.axis_labels[4], size=14)
        self._plot_sawyer_features(ax2, option=option)

        file_name = f"{solver.name}_value_function_seed_{seed}_episode_{episode}.png"
        plt.savefig(os.path.join(self.path, "value_function_plots", file_name))
        plt.close()

    def _plot_initiation_sets(self, option, episode):
        def _plot_trajectories(ax, title, x_idx, y_idx):
            positive_trajectories = option.positive_examples
            negative_trajectories = option.negative_examples
            for positive_trajectory in positive_trajectories:
                positive_trajectory = np.array(positive_trajectory)
                ax.plot(positive_trajectory[:, x_idx], positive_trajectory[:, y_idx],
                        label="positive", c=self.positive_color, alpha=0.5, linewidth=1.5)
            for negative_trajectory in negative_trajectories:
                negative_trajectory = np.array(negative_trajectory)
                ax.scatter(negative_trajectory[:, x_idx], negative_trajectory[:, y_idx],
                           label="negative", c=self.negative_color, alpha=0.5)
            ax.set_title(f"{title} Trajectories", size=16)
            ax.set_xlabel(self.axis_labels[x_idx], size=14)
            ax.set_ylabel(self.axis_labels[y_idx], size=14)

        def _plot_initiation_classifier(ax, init_set, title):
            ax.pcolormesh(self.grid_boundaries[0], self.grid_boundaries[1], init_set, cmap=cmap)
            if title.lower() == "endeff":
                x_label = self.axis_labels[0]
                y_label = self.axis_labels[1]
            elif title.lower() == "puck":
                x_label = self.axis_labels[3]
                y_label = self.axis_labels[4]
            else:
                raise NotImplementedError(title)
            ax.set_title(f"{title} Initiation Set Classifier", size=16)
            ax.set_xlabel(x_label, size=14)
            ax.set_ylabel(y_label, size=14)

        print(f"Plotting {option.name} initiation set")

        # indices for end effector and puck
        boolean_mesh = option.batched_is_init_true(self.center_points)
        endeff_inits = self._average_groupby_puck_or_endeff_pos("endeff", boolean_mesh)
        puck_inits = self._average_groupby_puck_or_endeff_pos("puck", boolean_mesh)
        cmap = "Blues"

        fig, axs = self._setup_plot((2, 2))
        fig.suptitle(f"{option.name} Initiation Set", size=24)
        trajectory_axes = axs[0]
        mesh_axes = axs[1]

        _plot_trajectories(trajectory_axes[0], "Endeff", 0, 1)
        _plot_trajectories(trajectory_axes[1], "Puck", 3, 4)
        _plot_initiation_classifier(mesh_axes[0], endeff_inits, "Endeff")
        _plot_initiation_classifier(mesh_axes[1], puck_inits, "Puck")

        # plot end effector bounds, end effector start, puck start, puck goal, and option target
        for axis in axs.flatten():
            self._plot_sawyer_features(axis, option=option)

        # plot legend and colorbar
        trajectories = "all" if len(option.negative_examples) > 0 else "positive"
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
        ax.set_xlim(self.axis_x_range)
        ax.set_ylim(self.axis_y_range)
        ax.set_xticks(np.linspace(self.axis_x_range[0], self.axis_x_range[1], 8))
        ax.set_yticks(np.linspace(self.axis_y_range[0], self.axis_y_range[1], 7))

        for ax in axs.flatten():
            ax.set_aspect("equal")
        return fig, axs

    def _plot_sawyer_features(self, ax, option=None, puck_pos=None):
        """Plots the different Sawyer environment features.

        Plots the puck goal (as a circle with radius = goal_tolerance),
        puck start, and end effector start states on `ax`.
        Args:
            ax: Axis generated by plt.subplots
            option (Option): if option is not None, will try to plot the target_salient_event
            of the option.
            puck_pos (tuple): x,y coordinates of puck position

        Returns:
            None
        """
        # using plt.Circle instead of scatter plot because we need to specify the radius of the
        # goal state (tolerance) in plot units
        if not self.task_agnostic:
            puck_goal = plt.Circle(self.puck_goal, self.goal_tolerance, color=self.goal_color,
                                   label="puck goal", alpha=0.3)
            ax.add_patch(puck_goal)

        # plot salient event that this option is targeting
        if option is not None and option.target_salient_event is not None:
            target_puck_pos = option.target_salient_event.get_target_position()
            salient_event = plt.Circle(target_puck_pos, self.salient_tolerance, alpha=0.3,
                                       color=self.target_salient_event_color, label="target salient event")
            ax.add_patch(salient_event)

        if puck_pos is not None:
            puck = plt.Circle(puck_pos, self.puck_radius, alpha=0.3,
                              color=self.puck_color, label="puck")
            ax.add_patch(puck)

        # plot the puck and endeff starting positions.
        ax.scatter(self.effective_hand_start[0], self.effective_hand_start[1], color="k", label="effective endeff start", marker="x", s=180)
        ax.scatter(self.true_hand_start[0], self.true_hand_start[1], color="k", label="true endeff start", marker="+", s=180)
        ax.scatter(self.puck_start[0], self.puck_start[1], color="k", label="puck start", marker="*", s=400)

    def _add_legend(self, ax, option=None, trajectories="none", include_puck=False):
        """
        Adds a legend to `ax`.
        Args:
            ax: axis created from plt.subplots() to add legend to
            option: if option.target_salient_event is not None, add a legend marker for target salient events
            trajectories:
                "all" - add legend marker for positive and negative trajectories.
                "positive" - add legend marker for positive trajectories only
                "none" - don't add legend marker for trajectories
            include_puck (bool): If true, add puck to legend

        Returns:
            None
        """
        puck_start_marker = Line2D([], [], marker="*", linestyle="none", color="k",
                                   markersize=12, label="puck start")
        effective_endeff_start_marker = Line2D([], [], marker="x", linestyle="none", color="k",
                                               markersize=12, label="effective end effector start")
        true_endeff_start_marker = Line2D([], [], marker="+", linestyle="none", color="k",
                                          markersize=12, label="true end effector start")
        handles = [puck_start_marker, effective_endeff_start_marker, true_endeff_start_marker]

        if not self.task_agnostic:
            puck_goal_marker = Line2D([], [], marker="o", linestyle="none", color=self.goal_color,
                                      markersize=12, label="puck goal")
            handles.append(puck_goal_marker)
        if option is not None and option.target_salient_event is not None:
            target_salient_marker = Line2D([], [], marker="o", linestyle="none", label="target salient event",
                                           markersize=12, color=self.target_salient_event_color)
            handles.append(target_salient_marker)

        if include_puck:
            puck_marker = Line2D([], [], marker="o", linestyle="none", label="puck",
                                 markersize=12, color=self.puck_color)
            handles.append(puck_marker)

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
            num_buckets_along_y = self.grid_boundaries[0].shape[1] - 1
        elif puck_or_endeff.lower() == "puck":
            avg = np.bincount(self.puck_idx, weights=values) / self.puck_cnt
            num_buckets_along_y = self.grid_boundaries[0].shape[1] - 1
        else:
            raise NotImplementedError("Invalid input for `puck_or_endeff`.")

        # we need to take the transpose because pcolormesh traverses along x but our
        # idx and cnt traverse along y because np.unique sorts lexographically
        return avg.reshape((num_buckets_along_y, -1)).T

    def _get_endeff_puck_grids(self, step=0.02):
        """
        Make meshgrids (boundary rectangles) for plt.pcolormesh and get the center point of each rectangle.
        Args:
            step: making this smaller improves resolution but hurts runtime/memory

        Returns:
            center_points, endeff_grid, puck_grid
        """
        axes_low = [self.axis_x_range[0], self.axis_y_range[0], 0.07, self.axis_x_range[0], self.axis_y_range[0]]
        axes_high = [self.axis_x_range[1], self.axis_y_range[1], 0.07, self.axis_x_range[1], self.axis_y_range[1]]

        # use np.meshgrid to make mesh of state space. Only need to return 2D meshes for graphing endeff
        # pos and puck pos.
        mesh_axes = [np.arange(axis_min, axis_max, step) for axis_min, axis_max in zip(axes_low, axes_high)]
        grid_rects = np.meshgrid(*mesh_axes[:2])

        # we need to calculate the center point of each rectangle to get the color of each rectangle (Z dimension)
        center_points = np.array(mesh_axes) + step / 2
        center_points = [center_points[0][:-1], center_points[1][:-1], [0.07], center_points[3][:-1], center_points[4][:-1]]
        center_points = np.column_stack(list(map(np.ravel, np.meshgrid(*center_points))))

        return center_points, grid_rects

    def visualize_ddpg_replay_buffer(self, solver, episode, seed, logdir):
        fig, (ax1, ax2) = self._setup_plot((1, 2))
        fig.suptitle(f"{solver.name} Replay Buffer Value Function Plot")


        states = np.array([exp[0] for exp in solver.replay_buffer.memory])
        actions = np.array([exp[1] for exp in solver.replay_buffer.memory])
        states_tensor = torch.from_numpy(states).float().to(solver.device)
        actions_tensor = torch.from_numpy(actions).float().to(solver.device)
        qvalues = solver.get_qvalues(states_tensor, actions_tensor).cpu().numpy().squeeze(1)
        plt.scatter(states[:, 0], states[:, 1], c=qvalues)
        plt.colorbar()
        file_name = f"{solver.name}_value_function_seed_{seed}_episode_{episode}.png"
        plt.savefig(os.path.join(logdir, "replay_buffer", file_name))
        plt.close()

