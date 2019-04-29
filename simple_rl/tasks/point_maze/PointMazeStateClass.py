# Python imports.
import numpy as np

# Other imports.
from simple_rl.mdp.StateClass import State

class PointMazeState(State):
    def __init__(self, position, has_key, theta, velocity, theta_dot, done):
        """
        Args:
            position (np.ndarray)
            has_key (float)
            theta (float)
            velocity (np.ndarray)
            theta_dot (float)
            done (bool)
        """
        assert has_key == 1. or has_key == 0., "{}".format(has_key)

        self.position = position
        self.has_key = bool(has_key)
        self.theta = theta
        self.velocity = velocity
        self.theta_dot = theta_dot
        features = [position[0], position[1], self.has_key, theta, velocity[0], velocity[1], theta_dot]

        State.__init__(self, data=features, is_terminal=done)

    def __str__(self):
        return "x: {}\ty: {}\thas_key: {}\ttheta: {}\txdot: {}\tydot: {}\tthetadot: {}\tterminal: {}\n".format(
                                                                                                  self.position[0],
                                                                                                  self.position[1],
                                                                                                  self.has_key,
                                                                                                  self.theta,
                                                                                                  self.velocity[0],
                                                                                                  self.velocity[1],
                                                                                                  self.theta_dot,
                                                                                                  self.is_terminal())

    def __repr__(self):
        return str(self)