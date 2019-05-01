from simple_rl.mdp.StateClass import State


class FixedReacherState(State):
    def __init__(self, features, done):
        self.position = features[:2]
        State.__init__(self, data=features, is_terminal=done)

    def __str__(self):
        return "x: {}\t y: {}\t Others: {}\tdone: {}".format(self.position[0], self.position[1],
                                                             self.features()[2:], self.is_terminal())

    def __repr__(self):
        return str(self)
