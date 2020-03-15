import numpy as np

from simple_rl.agents.func_approx.exploration.optimism.ExplorationBonusClass import ExplorationBonus
from simple_rl.agents.func_approx.exploration.optimism.latent.CountingLatentSpaceClass import CountingLatentSpace
from simple_rl.agents.func_approx.exploration.optimism.latent.utils import get_mean_std, normalize
import ipdb


class LatentCountExplorationBonus(ExplorationBonus):
    def __init__(self, state_dim, action_dim, latent_dim=2, lam=.1, epsilon=0.1,
                 writer=None, *, experiment_name, pixel_observation, normalize_states,
                 bonus_scaling_term, lam_scaling_term, optimization_quantity, num_frames):
        """

        Args:
            state_dim:
            action_dim:
            latent_dim:
            lam:
            epsilon:
            experiment_name:
            pixel_observation:
            normalize_states (bool): Whether to take care of normalization here or not
            bonus_scaling_term (str)
            lam_scaling_term (str)
            optimization_quantity (str)
            num_frames (int)
        """
        super(LatentCountExplorationBonus, self).__init__()

        self.counting_space = CountingLatentSpace(state_dim=state_dim, action_dim=action_dim,
                                                  latent_dim=latent_dim, epsilon=epsilon,
                                                  phi_type="function", experiment_name=experiment_name,
                                                  pixel_observations=pixel_observation, lam=lam,
                                                  optimization_quantity=optimization_quantity, writer=writer,
                                                  bonus_scaling_term=bonus_scaling_term,
                                                  lam_scaling_term=lam_scaling_term)

        self.un_normalized_sns_buffer = []
        self.actions = list(range(0, action_dim))
        self.un_normalized_action_buffers = [[] for _ in self.actions]
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.latent_dim = latent_dim
        self.normalize_states = normalize_states
        self.bonus_scaling_term = bonus_scaling_term
        self.num_frames = num_frames

        # Normalization constants
        self.mean_state = np.array([0.])
        self.std_state = np.array([1.])

    def add_transition(self, state, action, next_state=None):
        assert isinstance(state, np.ndarray), type(state)

        # Modify the stacking of the frames to allow for different stacking
        # between the RL agent and the exploration module
        state = self.modify_num_frames(state)
        next_state = self.modify_num_frames(next_state) if next_state is not None else next_state

        assert state.shape in (self.state_dim, (self.state_dim, )), state.shape

        if next_state is not None:
            self.un_normalized_sns_buffer.append((state, next_state))

        # We store un-normalized states in our own buffer, which gets normalized
        # once before calling `train()` on the latent counting class
        self.un_normalized_action_buffers[action].append(state)

        # We store normalized states (although with stale normalization constants)
        # in the counting latent space class
        if self.normalize_states:
            state = normalize(state, self.mean_state, self.std_state)

        self.counting_space.add_transition(state, action)

    def modify_num_frames(self, state):
        if state.shape[0] == self.num_frames:
            return state
        return state[-self.num_frames:, ...]

    def batched_modify_num_frames(self, states):
        if states.shape[1] == self.num_frames:
            return states
        return states[:, -self.num_frames:, ...]

    def _set_mean_and_std_state(self, sns_buffer):
        if sns_buffer is not None:
            states = [sns[0] for sns in sns_buffer]
            mean, std = get_mean_std(states)

            self.mean_state = mean
            self.std_state = std

    def _normalize_sns_buffer(self, sns_buffer):
        normalized_training_data = []
        for s, ns in sns_buffer:
            norm_s = normalize(s, self.mean_state, self.std_state)
            norm_ns = normalize(ns, self.mean_state, self.std_state)
            normalized_training_data.append((norm_s, norm_ns))

        return normalized_training_data

    def _normalize_action_buffers(self, action_buffers):
        return [normalize(b, self.mean_state, self.std_state) for b in action_buffers]

    def train(self, mode="entire", epochs=50):
        assert mode in ("entire", "partial")
        if mode == "entire":
            self._train_entire(epochs)
        else:
            self._train_partial(epochs)

    def _train_partial(self, epochs):
        self._train_counting_space(epochs)

    def _train_entire(self, epochs):
        self.counting_space.reset_model()
        self._train_counting_space(epochs)

    def _train_counting_space(self, epochs):
        state_next_state_buffer = self.un_normalized_sns_buffer if len(self.un_normalized_sns_buffer) > 0 else None
        action_buffers = [np.array(b) for b in self.un_normalized_action_buffers]

        if self.normalize_states:
            self._set_mean_and_std_state(state_next_state_buffer)
            state_next_state_buffer = self._normalize_sns_buffer(state_next_state_buffer)
            action_buffers = self._normalize_action_buffers(action_buffers)

        self.counting_space.train(state_next_state_buffer=state_next_state_buffer,
                                  action_buffers=action_buffers, epochs=epochs)

    def get_sns_buffer(self, normalized=False):
        """
        This is
        This gets the sns buffer, un
        """
        if normalized:
            if not self.normalize_states:
                raise Warning("You probably shouldn't be getting normalized states if you're not using normalization dummy!")
            return self._normalize_sns_buffer(self.un_normalized_sns_buffer)

        # Make sure they can't mess with the underlying list.
        return list(self.un_normalized_sns_buffer)

        # raise Exception("My impression is that you really don't want to be calling this from the outside. We always pass in raw states, and always convert them internally.")
        # if self.normalize_states:
        #     return self._normalize_sns_buffer(self.un_normalized_sns_buffer)
        # raise Warning("Asking for normalized buffer when self.normalize_states is False")

    def get_exploration_bonus(self, state, action=None):
        """

        Args:
            state (np.ndarray): a numpy array representing a single state.
            action (None or int): Either the action idx, or None if you want all action-bonuses for state.

        Returns:
            bonus (np.ndarray): represents count-bonuses for an action, or for all actions.
                          Shape (1,) for the former and (num_actions,) for the latter.

        """
        state_dim = self.counting_space.state_dim

        state = self.modify_num_frames(state)

        assert state.shape in ((state_dim,), state_dim), (state_dim, state.shape)
        assert isinstance(state, np.ndarray)

        if self.normalize_states:
            state = normalize(state, self.mean_state, self.std_state)

        if action is not None:
            count = self.counting_space.get_counts(np.expand_dims(state, 0), action)
            bonus = self._counts_to_bonus(count)
            assert bonus.shape == (1,), bonus.shape
            return bonus

        # If `action` is not specified, return the exploration bonuses
        # corresponding to all the actions in the MDP
        bonuses = []
        for action in self.actions:
            count = self.counting_space.get_counts(np.expand_dims(state, 0), action)[0]
            bonus = self._counts_to_bonus(count)
            bonuses.append(bonus)
        bonuses = np.array(bonuses)
        bonuses = np.expand_dims(bonuses, 0)
        # import pdb; pdb.set_trace()
        assert bonuses.shape == (1, len(self.actions)), bonuses
        return bonuses

    def get_batched_exploration_bonus(self, states):
        assert isinstance(states, np.ndarray), type(states)

        states = self.batched_modify_num_frames(states)

        if self.normalize_states:
            states = normalize(states, self.mean_state, self.std_state)

        counts = []
        for action in self.actions:
            action_counts = self.counting_space.get_counts(states, action)
            counts.append(action_counts)

        count_array = np.array(counts).T

        assert count_array.shape == (states.shape[0], len(self.actions))

        return self._counts_to_bonus(count_array)

    def _counts_to_bonus(self, counts):
        return 1. / np.sqrt(counts + 1e-2)

    def get_counts(self, X, buffer_idx):
        """
        We're doing it here too so we don't feel tempted to go down in to novelty_tracker when we want counts.
        Because that caused a headache with normalization.
        Args:
            X (np.ndarray): states
            buffer_idx (int): action

        Returns:
            counts

        """
        if self.normalize_states:
            X = normalize(X, self.mean_state, self.std_state)
        return self.counting_space.get_counts(X, buffer_idx)
