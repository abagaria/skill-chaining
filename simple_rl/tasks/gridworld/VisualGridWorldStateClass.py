# Python imports
import numpy as np
import matplotlib.pyplot as plt
import pdb

# Local imports
from simple_rl.mdp.StateClass import State


class VisualGridWorldState(State):

    def __init__(self, data=[], is_terminal=False):
        self.data = data
        State.__init__(self, data=self.data, is_terminal=is_terminal)

    def features(self):
        return self.data

    @staticmethod
    def grayscale(image):
        return np.mean(image, axis=-1)

    def render(self):
        plt.imshow(self.data)
        plt.show()
