from simple_rl.tasks.point_maze.environments.maze_env import MazeEnv
from simple_rl.tasks.point_maze.environments.swimmer import SwimmerEnv


class SwimmerMazeEnv(MazeEnv):
    MODEL_CLASS = SwimmerEnv
