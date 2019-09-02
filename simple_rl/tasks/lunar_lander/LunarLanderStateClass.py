# Python imports
from PIL import Image

# Local imports
from simple_rl.mdp.StateClass import State

class LunarLanderState(State):

    def __init__(self, data=[], is_terminal=False):
        self.data = data
        State.__init__(self, data=data, is_terminal=is_terminal)

    def features(self):
        return self.data

    def render(self):
        img = Image.fromarray(self.data)
        img.show()
