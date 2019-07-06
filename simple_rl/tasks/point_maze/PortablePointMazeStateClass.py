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
        self.door_obs = aspace_features[:8]
        self.key_obs = aspace_features[8:12]
        self.lock_obs = aspace_features[12:16]

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
        aspace_str = "Door obs: {}\tKey Obs: {}\t Lock Obs: {}".format(self.door_obs, self.key_obs, self.lock_obs)
        return pspace_str + "\t" + aspace_str

    def __repr__(self):
        return str(self)

    def initiation_classifier_features(self):
        key_features = self.key_obs[:1]   #  self.key_obs[:2]    # distance, angle
        lock_features = self.lock_obs[:1] # self.lock_obs[:2]  # distance, angle
        # door_features = self.door_obs[:2] # self.door_obs[:4]  # distance1, distance2, angle1, angle2
        has_key = np.array([self.has_key])
        return np.concatenate((key_features, lock_features, has_key), axis=0)

    @staticmethod
    def initiation_classifier_feature_indices():
        # door_obs_idx = [0, 1]  # list(range(4))
        key_obs_idx = [8]  # list(range(8, 10))
        lock_obs_idx = [12]  # list(range(12, 14))
        has_key_idx = [16]
        return key_obs_idx + lock_obs_idx + has_key_idx
