import os

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
plt.style.use("seaborn-colorblind")

from simple_rl.agents.func_approx.dsc.SkillChainingPlotterClass import SkillChainingPlotter


class LeapWrapperPlotterClass(SkillChainingPlotter):
    def __init__(self, experiment_name, task_name):
        # The true start state is: [-0.007, 0.52], but the hand gets to [0.032, 0.409]
        # within a few steps of resetting the environment. This is probably because the
        # arm starts with an initial z position of 0.12, but it goes to 0.07 very quickly.
        # As a result, Reacher might have to slightly adjust the x and y position to drop
        # the z position.
        self.hand_start = (0.032, 0.409)
        self.puck_start = (0., 0.6)
        self.puck_goal = (-0.15, 0.6)

        # for plotting axes
        self.x_range = (-0.4, 0.4)
        self.y_range = (0.2, 1.)
        self.axis_labels = ['endeff_x', 'endeff_y', 'endeff_z', 'puck_x', 'puck_y']

        # grid of points to plot decision classifiers
        axes_low = [-0.28, 0.3, 0.06, -0.4, 0.2]
        axes_high = [0.28, 0.9, 0.08, 0.4, 1.]
        meshgrid = np.meshgrid(*[np.arange(axis_min, axis_max, 0.02) for axis_min, axis_max in
                                 zip(axes_low, axes_high)])
        self.mesh = np.vstack(map(np.ravel, meshgrid))

        # Tolerance of being within goal state or salient events. This is used to plot the
        # radius of the goal and salient events
        self.tolerance = 0.03

        super().__init__(task_name, experiment_name)

    def generate_episode_plots(self, chainer, episode):
        """
        Args:
            chainer (SkillChainingAgent): the skill chaining agent we want to plot
            episode (int)
        """
        for option in chainer.trained_options:
            self._plot_initiation_sets(option, episode)
        pass

    def generate_experiment_plots(self, chainer):
        pass

    def _plot_initiation_sets(self, option, episode):
        def _setup_plot():
            # set up figure and axes
            fig, axs = plt.subplots(2, 2, sharex='all', sharey='all')
            fig.set_size_inches(18, 15)
            fig.suptitle(f"{option.name} Initiation Set", size=16)

            # doesn't matter which axis we set these for because sharey and sharex are true
            axs[0, 0].set_xlim(self.x_range)
            axs[0, 0].set_ylim(self.y_range)
            axs[0, 0].set_xticks(np.linspace(self.x_range[0], self.x_range[1], 9))
            axs[0, 0].set_yticks(np.linspace(self.y_range[0], self.y_range[1], 9))

            # plot puck start position, hand start position, and puck goal state
            for ax in axs.flatten():
                ax.scatter(self.hand_start[0], self.hand_start[1], color="k", label="endeff start", marker="x", s=100)
                ax.scatter(self.puck_start[0], self.puck_start[1], color="k", label="puck start", marker="*", s=100)

                # using circle to specify the radius of the goal state (tolerance)
                puck_goal = plt.Circle(self.puck_goal, self.tolerance, color='gold', label="puck goal")
                ax.add_patch(puck_goal)

            return fig, axs

        def _plot_trajectories(axis, option, x_idx, y_idx, title):
            positive_trajectories = option.positive_examples
            negative_trajectories = option.negative_examples
            for positive_trajectory in positive_trajectories:
                axis.plot(positive_trajectory[:, x_idx], positive_trajectory[:, y_idx], label="positive", c="b")
            for negative_trajectory in negative_trajectories:
                axis.plot(negative_trajectory[:, x_idx], negative_trajectory[:, y_idx], label="negative", c="r")
            axis.set_title(f"{title} Trajectories")
            axis.set_xlabel(self.axis_labels[x_idx])
            axis.set_ylabel(self.axis_labels[y_idx])

        def _plot_initiation_classifier(axis, data, x_idx, y_idx, title):
            x_y, counts = np.unique(data, axis=0, return_counts=True)
            axis.scatter(x_y[:, x_idx], x_y[:, y_idx], c=counts, cmap=plt.cm.get_cmap("Blues"))
            axis.set_title(f"{title} Initiation Set Classifier")
            axis.set_xlabel(self.axis_labels[x_idx])
            axis.set_ylabel(self.axis_labels[y_idx])

        titles = ['Endeff', 'Puck']
        boolean_mesh = [state for state in self.mesh if option.is_init_true(state)]
        # indices for end effector and puck
        indices = [(0, 1), (3, 4)]

        fig, (mesh_axes, trajectory_axes) = _setup_plot()

        for (x_idx, y_idx), mesh_axis, trajectory_axis, title in zip(indices, mesh_axes, trajectory_axes, titles):
            # plot positive and negative examples
            _plot_trajectories(mesh_axis, option, x_idx, y_idx, title)

            # plot initiation classifier using mesh
            _plot_initiation_classifier(trajectory_axes, boolean_mesh, x_idx, y_idx, title)

        mesh_axes[1].legend()
        trajectory_axes[1].legend()

        # save plot as png
        file_name = f"{option.name}_episode_{episode}_{option.seed}.png"
        plt.savefig(os.path.join(self.path, "initiation_set_plots", file_name))
