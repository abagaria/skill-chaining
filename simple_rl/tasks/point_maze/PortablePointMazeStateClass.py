# Python imports.
import numpy as np

# Other imports.
from simple_rl.mdp.PortableStateClass import PortableState

class PortablePointMazeState(PortableState):
    def __init__(self, pspace_features, aspace_features, pspace_done):
        """
        Args:
            pspace_features (np.ndarray)
            aspace_features (np.ndarray)
            pspace_done (bool)
        """
        # Problem-space features
        self.position = pspace_features[:2]
        self.has_key = pspace_features[2]
        self.theta = pspace_features[3]
        self.velocity = pspace_features[4:6]
        self.theta_dot = pspace_features[6]

        # Agent-space features
        self.agent_space_features = aspace_features

        PortableState.__init__(self, pspace_features, aspace_features, pspace_done)

    def __str__(self):
        pspace_str = "x: {}\ty: {}\thas_key: {}\ttheta: {}\txdot: {}\tydot: {}\tthetadot: {}\tterminal: {}\n".format(
                                                                                                  self.position[0],
                                                                                                  self.position[1],
                                                                                                  self.has_key,
                                                                                                  self.theta,
                                                                                                  self.velocity[0],
                                                                                                  self.velocity[1],
                                                                                                  self.theta_dot,
                                                                                                  self.is_terminal())
        aspace_str = "Range Sensor: {}".format(self.agent_space_features)
        return pspace_str + "\t" + aspace_str

    def __repr__(self):
        return str(self)

    def initiation_classifier_features(self):
        return self.agent_space_features

    @staticmethod
    def initiation_classifier_feature_indices():
        return list(range(19))
