# Python imports
import numpy as np

# Local imports
from simple_rl.mdp.StateClass import State

''' GymStateClass.py: Contains a State class for Gym. '''

class GymState(State):
    ''' Gym State class '''

    def __init__(self, image, position, ram, is_terminal=False):
        self.image = image
        self.position = position
        self.ram = ram
        State.__init__(self, data=image, is_terminal=is_terminal)

    def features(self):
        return self.image

    def get_position(self):
        return np.array(self.position)