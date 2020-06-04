from pathlib import Path
import abc

class SkillChainingPlotter(metaclass=abc.ABCMeta):
    def __init__(self, task_name, experiment_name):
        '''
        Args:
            task_name (str): The name of the current task, so we know where to save plots
            experiment_name (str): The name of the current task, so we know where to save plots
        '''
        self.task_name = task_name
        self.experiment_name = experiment_name

        self.path = Path.cwd()/"plots"/task_name/experiment_name
        self.path.mkdir(parents=True, exist_ok=True)

    @abc.abstractmethod
    def generate_episode_plots(self, chainer, episode):
        '''
        Args:
            chainer (SkillChainingAgent): the skill chaining agent we want to plot
            episode (int): the current episode
        '''
        pass

    @abc.abstractmethod
    def generate_experiment_plots(self, chainer):
        '''
        Args:
            chainer (SkillChainingAgent): the skill chaining agent we want to plot
        '''
        pass