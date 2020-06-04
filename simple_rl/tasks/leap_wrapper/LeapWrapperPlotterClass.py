from simple_rl.agents.func_approx.dsc.SkillChainingPlotterClass import SkillChainingPlotter


class LeapWrapperPlotterClass(SkillChainingPlotter):
    def __init__(self, experiment_name):
        SkillChainingPlotter.__init__(self, "sawyer", experiment_name)

    def generate_episode_plots(self, chainer, episode):
        """
        Args:
            chainer (SkillChainingAgent): the skill chaining agent we want to plot
            episode (int)
        """
        pass


    def generate_experiment_plots(self, chainer):
        pass

