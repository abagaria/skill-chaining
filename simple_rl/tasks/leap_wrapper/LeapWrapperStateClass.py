# Python imports
from copy import deepcopy

import numpy as np

# Local imports
from simple_rl.mdp.StateClass import State

''' GymStateClass.py: Contains a State class for Gym. '''

class LeapWrapperState(State):
    ''' Gym State class '''

    def __init__(self, endeff_pos, puck_pos, done):
        """
        Args:
            endeff_pos (np.ndarray)
                x,y,z position of the robot hand
            puck_pos (np.ndarray)
                x,y position of the puck on the table
            done (Boolean)
        """
        # TODO: Make this cleaner because we're reusing endeff_pos
        self.endeff_pos = endeff_pos
        self.puck_pos = puck_pos
        features = [endeff_pos[0], endeff_pos[1], endeff_pos[2], puck_pos[0], puck_pos[1]]

        # must be np.ndarray
        self.position = np.array(features)

        State.__init__(self, data=features, is_terminal=done)
    
    def __str__(self):
        return "hand_xpos: {}\thand_ypos: {}\thand_zpos: {}\tpuck_xpos: {}\tpuck_ypos: {}\tterminal: {}\n".format(self.endeff_pos[0],
                                                                                                        self.endeff_pos[1],
                                                                                                        self.endeff_pos[2],
                                                                                                        self.puck_pos[0],
                                                                                                        self.puck_pos[1],
                                                                                                        self.is_terminal())


    def __repr__(self):
        return str(self)