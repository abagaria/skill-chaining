# Python imports
import numpy as np
from PIL import Image
import pdb

# Local imports
from simple_rl.mdp.StateClass import State

''' GymStateClass.py: Contains a State class for Gym. '''

class GymState(State):
    ''' Gym State class '''

    def __init__(self, data=[], is_terminal=False):
        self.data = self.grayscale(data)
        State.__init__(self, data=self.data, is_terminal=is_terminal)

    def features(self):
        return self.data

    @staticmethod
    def grayscale(image):
        return np.mean(image, axis=-1)

    def render(self):
        img = Image.fromarray(self.data)
        img.show()
