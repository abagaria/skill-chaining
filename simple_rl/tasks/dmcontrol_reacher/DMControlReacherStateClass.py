from simple_rl.mdp.StateClass import State

class DMControlReacherState(State):
    def __init__(self, features, reward, info):
        self.inf = info
        self.data = features
        self.reward = reward
        is_terminal = self.is_terminal()
        State.__init__(self, data=features, is_terminal=is_terminal)

    def __repr__(self):
        return str(self)

    def features(self):
        return self.data

    def is_terminal(self):
        if self.reward > 0:
            return True
        return False

    def info(self):
        ret = self.inf['internal_state']
        return ret
