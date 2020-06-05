import os
import abc

import ipdb


class SkillChainingPlotter(metaclass=abc.ABCMeta):
    def __init__(self, task_name, experiment_name):
        """
        Args:
            task_name (str): The name of the current task, so we know where to save plots
            experiment_name (str): The name of the current experiment, so we know where to save plots
        """
        self.path = rotate_file_name(os.path.join("plots", task_name, experiment_name))
        ipdb.set_trace()
        create_log_dir(self.path)

    @abc.abstractmethod
    def generate_episode_plots(self, chainer, episode):
        """
        Args:
            chainer (SkillChainingAgent): the skill chaining agent we want to plot
            episode (int): the current episode
        """
        # initiation set
        # value function
        # low level shaped rewards
        # high level shaped rewards
        pass

    @abc.abstractmethod
    def generate_experiment_plots(self, chainer):
        """
        Args:
            chainer (SkillChainingAgent): the skill chaining agent we want to plot
        """
        pass


def rotate_file_name(file_path):
    if os.path.isdir(file_path):
        suffix = 1
        file_path += "_" + str(suffix)
        while os.path.isdir(file_path):
            suffix += 1
            file_path = file_path.rsplit("_", 1)[0] + "_" + str(suffix)
        return file_path
    else:
        return file_path


def create_log_dir(directory_path):
    try:
        os.makedirs(directory_path)
    except OSError:
        print("Creation of the directory %s failed" % os.path.join(os.getcwd(), directory_path))
    else:
        print("Successfully created the directory %s " % os.path.join(os.getcwd(), directory_path))
