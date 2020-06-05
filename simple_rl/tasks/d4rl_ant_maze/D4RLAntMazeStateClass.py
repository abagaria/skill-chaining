# Python imports.
import numpy as np

# Other imports.
from simple_rl.mdp.StateClass import State

class D4RLAntMazeState(State):
    def __init__(self, position, others, done, *, goal_component=None):
        """
        Args:
            position (np.ndarray)
            others (np.ndarray)
            done (bool)
            goal_component (np.ndarray)
        """
        self.position = position
        self.others = others
        self.goal_component = goal_component

        if goal_component is None:
            features = self.position.tolist() + self.others.tolist()
        else:
            features = self.position.tolist() + self.others.tolist() + self.goal_component.tolist()

        State.__init__(self, data=features, is_terminal=done)

    def __str__(self):
        return "x: {}\ty: {}\tothers:{}\tterminal: {}\tgoal_diff: {}\n".format(self.position[0],
                                                                               self.position[1],
                                                                               self.others,
                                                                               self.is_terminal(),
                                                                               self.goal_component)

    def __repr__(self):
        return str(self)
