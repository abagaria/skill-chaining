# Python imports.
import numpy as np

# Other imports.
from simple_rl.mdp.StateClass import State

class PointReacherState(State):
    def __init__(self, position, theta, velocity, theta_dot, done, *, goal_component=None):
        """
        Args:
            position (np.ndarray)
            theta (float)
            velocity (np.ndarray)
            theta_dot (float)
            done (bool)
            goal_component (np.ndarray)
        """
        self.position = position
        self.theta = theta
        self.velocity = velocity
        self.theta_dot = theta_dot
        self.goal_component = goal_component

        if goal_component is None:
            features = [position[0], position[1], theta, velocity[0], velocity[1], theta_dot]
        else:
            features = [position[0], position[1], theta, velocity[0],
                        velocity[1], theta_dot, goal_component[0], goal_component[1]]

        State.__init__(self, data=features, is_terminal=done)

    def __str__(self):
        return "x: {}\ty: {}\ttheta: {}\txdot: {}\tydot: {}\tthetadot: {}\tterminal: {}\tgoal_dist: {}\n".format(
                                                                                                  self.position[0],
                                                                                                  self.position[1],
                                                                                                  self.theta,
                                                                                                  self.velocity[0],
                                                                                                  self.velocity[1],
                                                                                                  self.theta_dot,
                                                                                                  self.is_terminal(),
                                                                                                  self.goal_component)

    def __repr__(self):
        return str(self)