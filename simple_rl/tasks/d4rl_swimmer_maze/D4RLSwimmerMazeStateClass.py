# Python imports.
import numpy as np

# Other imports.
from simple_rl.mdp.StateClass import State


class D4RLSwimmerMazeState(State):
    """
    Page 117 of: https://tel.archives-ouvertes.fr/tel-00003985/document
    explains what the different state components are.
    """
    def __init__(self, position, others, is_terminal):
        """
        Args:
            position (np.ndarray)
            others (np.ndarray)
            is_terminal (bool)
        """
        self.position = position
        self.others = others
        features = list(position) + list(others)

        State.__init__(self, data=features, is_terminal=is_terminal)

    def __str__(self):
        return f"position: {self.position}, terminal: {self.is_terminal()}"

    def __repr__(self):
        return str(self)