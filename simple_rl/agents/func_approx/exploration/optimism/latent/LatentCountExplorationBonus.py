import numpy as np

from simple_rl.agents.func_approx.exploration.optimism.ExplorationBonusClass import ExplorationBonus
from simple_rl.agents.func_approx.exploration.optimism.latent.CountingLatentSpaceClass import CountingLatentSpace
import pdb


class LatentCountExplorationBonus(ExplorationBonus):
    def __init__(self, state_dim, action_dim, latent_dim=2, lam=1., epsilon=0.1, *, experiment_name, pixel_observation):
        super(LatentCountExplorationBonus, self).__init__()

        self.counting_space = CountingLatentSpace(state_dim=state_dim, action_dim=action_dim,
                                                  latent_dim=latent_dim, epsilon=epsilon,
                                                  phi_type="function", experiment_name=experiment_name,
                                                  pixel_observations=pixel_observation, lam=lam,
                                                  optimization_quantity="bonus")

        self.sns_buffer = []
        self.actions = list(range(0, action_dim))
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.latent_dim = latent_dim

    def add_transition(self, state, action, next_state=None):
        assert isinstance(state, np.ndarray), type(state)
        assert state.shape in (self.state_dim, (self.state_dim, )), state.shape

        self.counting_space.add_transition(state, action)

        if next_state is not None:
            self.sns_buffer.append((state, next_state))

    def train(self, mode="entire", epochs=100):
        assert mode in ("entire", "partial")
        if mode == "entire":
            self._train_entire(epochs)
        else:
            self._train_partial(epochs)

    def _train_partial(self, epochs):
        state_next_state_buffer = self.sns_buffer if len(self.sns_buffer) > 0 else None
        self.counting_space.train(state_next_state_buffer=state_next_state_buffer, epochs=epochs)

    def _train_entire(self, epochs):
        self.counting_space.reset_model()
        state_next_state_buffer = self.sns_buffer if len(self.sns_buffer) > 0 else None
        self.counting_space.train(state_next_state_buffer=state_next_state_buffer, epochs=epochs)

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
        assert state.shape in ((state_dim,), state_dim), (state_dim, state.shape)
        assert isinstance(state, np.ndarray)

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
        counts = []
        for action in self.actions:
            action_counts = self.counting_space.get_counts(states, action)
            counts.append(action_counts)

        count_array = np.array(counts).T

        assert count_array.shape == (states.shape[0], len(self.actions))

        return self._counts_to_bonus(count_array)

    def _counts_to_bonus(self, counts):
        return 1. / np.sqrt(counts + 1e-2)