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
        self.position = pspace_features[:2]
        PortableState.__init__(self, pspace_features, aspace_features, pspace_done)

    def __str__(self):
        pspace_str = "Problem-space: {}".format(self.pspace_features())
        aspace_str = "Agent-space: {}".format(self.aspace_features())
        return pspace_str + "\t" + aspace_str

    def __repr__(self):
        return str(self)

    def initiation_classifier_features(self):
        features = self.aspace_features()
        return features
        # return np.array([features[1], features[3], features[5], features[7]])

    @staticmethod
    def initiation_classifier_feature_indices():
        # return [1, 3, 5, 7]
        return list(range(8))
