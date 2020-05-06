# Python imports.
import numpy as np

# Other imports.
from simple_rl.mdp.StateClass import State

class D4RLPointMazeState(State):
    def __init__(self, position, velocity, done):
        """
        Args:
            position (np.ndarray)
            velocity (np.ndarray)
            done (bool)
        """
        self.position = position
        self.velocity = velocity
        features = [position[0], position[1], velocity[0], velocity[1]]

        State.__init__(self, data=features, is_terminal=done)

    def __str__(self):
        return "x: {}\ty: {}\txdot: {}\tydot: {}\tterminal: {}\n".format(self.position[0],
                                                                         self.position[1],
                                                                         self.velocity[0],
                                                                         self.velocity[1],
                                                                         self.is_terminal())

    def __repr__(self):
        return str(self)