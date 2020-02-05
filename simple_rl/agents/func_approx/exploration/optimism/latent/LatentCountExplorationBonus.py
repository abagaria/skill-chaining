from simple_rl.agents.func_approx.exploration.optimism.ExplorationBonusClass import ExplorationBonus


class LatentCountExplorationBonus(ExplorationBonus):
    def __init__(self):
        super(LatentCountExplorationBonus, self).__init__()

    def get_exploration_bonus(self, state, action=None):
        pass

    def get_batched_exploration_bonus(self, states):
        pass