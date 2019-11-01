import pdb
import warnings
import time
import numpy as np
from collections import defaultdict
from sklearn.neighbors import KernelDensity


class DensityModel(object):
    def __init__(self, bandwidth=0.3, use_full_state=False):
        self.bandwidth = bandwidth
        self.use_full_state = use_full_state

        # r_tol of 0.05 implies an error tolerance of 5%
        self.model = KernelDensity(bandwidth=bandwidth)
        self.fitted = False

        # For KDE plotting
        self.number_of_fits = 0
        self.state_to_count = defaultdict(lambda : 0.)

    def _fit(self, state_buffer):
        """
        Args:
            state_buffer (np.ndarray): Array with each row representing the features of the state
        """
        if self.use_full_state:
            input_buffer = state_buffer
        else:  # Use only the position elements of the state vector
            input_buffer = state_buffer[:, :2]

        start_time = time.time()
        self.model.fit(input_buffer)
        end_time = time.time()

        fitting_time = end_time - start_time
        if fitting_time >= 1:
            print("\rDensity Model took {} seconds to fit".format(end_time - start_time))
        self.fitted = True

        return fitting_time

    def _get_log_prob(self, states):
        """
        Args:
            states (np.ndarray)
        Returns:
            log_probabilities (np.ndarray): log probability of each state in `states`
        """
        if self.use_full_state:
            X = states
        else:  # Use only the position elements of the state vector
            X = states[:, :2]
        log_pdf = self.model.score_samples(X)
        return log_pdf

    def update(self, stored_states):
        fitting_time = self._fit(stored_states)
        self.number_of_fits += 1
        return fitting_time

    @staticmethod
    def _get_pseudo_count(probability, recoded_probability):
        """ Convert the log probability densities to counts. """

        if probability == 0.:
            return 0.
        if probability >= recoded_probability:
            return np.inf

        return max(0, probability * (1. - recoded_probability) / (recoded_probability - probability))

    def get_online_exploration_bonus(self, stored_states, next_state, beta=0.05):
        """
        Given a transition in the environment of the form (s, a, r, s'), compute the
        exploration bonus under the density model.
        Args:
            stored_states (list): list of numpy arrays representing the current replay buffer
            next_state (np.ndarray): s'
            beta (float): Coefficient balancing intrinsic vs extrinsic rewards

        Returns:
            bonus (float): Exploration bonus for landing in s'
        """
        # Fit on the replay buffer of size (BATCH_SIZE) for the 1st time
        if not self.fitted:
            self.update(stored_states)

        # Un-squeeze the batch dimension
        next_state = next_state[None, ...]

        # Get the log probability density under the old model
        log_p_sprime = self._get_log_prob(next_state)[0]

        # Re-fit with B U s'
        augmented_data = np.concatenate((stored_states, next_state), axis=0)
        self.update(augmented_data)

        # Get the log probability density under the new model
        new_log_p_sprime = self._get_log_prob(next_state)[0]

        # Convert the log probabilities to probabilities using stable exponentiation
        probability = self._log_probability_to_probability(log_p_sprime)
        recoded_probability = self._log_probability_to_probability(new_log_p_sprime)

        # Convert probabilities to pseudo-counts
        pseudo_count = self._get_pseudo_count(probability, recoded_probability)

        # Logging
        self.state_to_count[tuple(next_state[0])] = pseudo_count

        # Replace NaNs with 0s
        with warnings.catch_warnings():
            warnings.filterwarnings('error')
            try:
                bonus = beta * np.power(pseudo_count + 0.01, -0.5)
            except Warning as e:
                print(np.expm1(log_p_sprime), np.expm1(new_log_p_sprime), probability, recoded_probability, pseudo_count)
        bonus = np.nan_to_num(bonus)

        return bonus

    def infer_probabilities(self, states):
        """ Used for plotting probability map over state-space. """
        log_probabilities = self.model.score_samples(states)
        probabilities = np.exp(log_probabilities)
        probabilities[probabilities > 1] = np.expm1(log_probabilities[probabilities > 1])
        probabilities[probabilities < 0] = np.expm1(log_probabilities[probabilities < 0])
        return probabilities

    @staticmethod
    def _log_probability_to_probability(log_prob):
        p1 = np.exp(log_prob)
        p2 = np.expm1(log_prob)
        # if p1 < 0 or p1 > 1:
        #     return p2
        return p1
