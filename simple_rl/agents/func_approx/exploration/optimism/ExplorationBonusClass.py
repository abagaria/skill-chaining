class ExplorationBonus:
    def __init__(self):
        super(ExplorationBonus, self).__init__()

    def train(self):
        pass

    def add_transition(self, state, action, next_state=None):
        pass

    def get_exploration_bonus(self, state, action=None):
        """ Given a single state, action pair, return the corresponding exploration bonus. """
        raise NotImplementedError

    def get_batched_exploration_bonus(self, states):
        raise NotImplementedError
