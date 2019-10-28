# Python imports
import numpy as np
from PIL import Image

# Local imports
from simple_rl.mdp.StateClass import State

''' GymStateClass.py: Contains a State class for Gym. '''

class GymState(State):
    ''' Gym State class '''

    def __init__(self, data, position, is_terminal=False):
        self.data = data
        self.position = position
        State.__init__(self, data=data, is_terminal=is_terminal)

    def features(self):
        return self.data

    def render(self):
        img = Image.fromarray(self.data)
        img.show()
