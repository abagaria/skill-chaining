# Python imports.
from __future__ import print_function
import numpy as np

# Other imports.
from simple_rl.mdp.StateClass import State

class PinballState(State):
    def __init__(self, x, y, xdot, ydot, is_terminal=False):
        self.x = x
        self.y = y
        self.xdot = xdot
        self.ydot = ydot

        State.__init__(self, data=[x, y, xdot, ydot], is_terminal=is_terminal)

    def get_position(self):
        return np.array([self.x, self.y])

    def state_space_size(self):
        return len(self.data)

    def __hash__(self):
        return hash(tuple(self.data))

    def __str__(self):
        return "(x: {}, y: {}, xdot: {}, ydot: {}, term: {})".format(*tuple(self.data), self.is_terminal())

    def __repr__(self):
        return str(self)

    def __eq__(self, other):
        return isinstance(other, PinballState) and self.data == other.data

    def __ne__(self, other):
        return not self == other