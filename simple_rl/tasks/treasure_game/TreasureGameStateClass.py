# Python imports
import numpy as np

# Local imports
from simple_rl.mdp.StateClass import State

''' TreasureGameClass.py: Contains a State class for the Treasure Game environment.'''

class TreasureGameState(State):
    ''' Treasure Game State class '''
    def __init__(self, agent_x, agent_y, handle_1_angle, handle_2_angle, key_x, key_y, bolt_locked, coin_x, coin_y, done):
        '''
        Args:
            agent_x (float)
            agent_y (float)
            handle_1_angle (float)
            handle_2_angle (float)
            key_x (float)
            key_y (float)
            bolt_locked (bool)
            coin_x (float)
            coin_y (float)
            done (bool)
        '''
    
        self.agent_x = agent_x
        self.agent_y = agent_y
        self.handle_1_angle = handle_1_angle
        self.handle_2_angle = handle_2_angle
        self.key_x = key_x
        self.key_y = key_y
        self.bolt_locked = bolt_locked
        self.coin_x = coin_x
        self.coin_y = coin_y
        
        # TODO: added to play nice with DSC
        self.position = [self.agent_x, self.agent_y]

        features = [self.agent_x, self.agent_y,
                    self.handle_1_angle, self.handle_2_angle,
                    self.key_x, self.key_y,
                    self.bolt_locked,
                    self.coin_x, self.coin_y]
        
        State.__init__(self, data=features, is_terminal=done)

    def __str__(self):
        string = "agent_x: {}\t agent_y: {}\t ".format(self.agent_x, self.agent_y)
        string += "handle_1_angle: {}\t handle_2_angle: {}\t ".format(self.handle_1_angle, self.handle_2_angle)
        string += "key_x: {}\t key_y: {}\t ".format(self.key_x, self.key_y)
        string += "bolt_locked: {}\t ".format(self.bolt_locked)
        string += "coin_x: {}\t coin_y: {}\t ".format(self.coin_x, self.coin_y)
        string += "terminal: {}\t ".format(self.is_terminal())
        string += "\n"
        return string

    def __repr__(self):
        return str(self)

    def to_rgb(self, x_dim, y_dim):
        # 3 by x_length by y_length array with values 0 (0) --> 1 (255)
        board = np.zeros(shape=[3, x_dim, y_dim])
        # print self.data, self.data.shape, x_dim, y_dim
        return self.data