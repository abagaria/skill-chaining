from simple_rl.mdp.StateClass import State

class AntFourRoomState(State):
    def __init__(self, position, other_features, done):
        self.position = position
        self.other_features = other_features

        features = position.tolist()
        features += other_features.tolist()

        State.__init__(self, data=features, is_terminal=done)

    def __str__(self):
        return "x: {}\ty: {}\tother_features: {}".format(self.position[0], self.position[1], self.other_features)

    def __repr__(self):
        return str(self)
