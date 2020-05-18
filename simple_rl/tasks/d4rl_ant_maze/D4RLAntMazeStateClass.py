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
        self.position = position
        self.others = others
        features = self.position.tolist() + self.others.tolist()

        State.__init__(self, data=features, is_terminal=done)

    def __str__(self):
        return "x: {}\ty: {}\tothers:{}\tterminal: {}\n".format(self.position[0],
                                                                self.position[1],
                                                                self.others,
                                                                self.is_terminal())

    def __repr__(self):
        return str(self)
