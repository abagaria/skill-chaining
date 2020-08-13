# Python imports.
import numpy as np

# Other imports.
from simple_rl.mdp.StateClass import State

class D4RLAntMazeState(State):
    def __init__(self, position, others, done):
        """
        Args:
            position (np.ndarray)
            others (np.ndarray)
            done (bool)
        """
        features = position.tolist() + others.tolist()

        State.__init__(self, data=features, is_terminal=done)

    def __str__(self):
        return "x: {}\ty: {}\tothers:{}\tterminal: {}\n".format(self.features()[0],
                                                                self.features()[1],
                                                                self.features()[2:],
                                                                self.is_terminal())

    def __repr__(self):
        return str(self)
