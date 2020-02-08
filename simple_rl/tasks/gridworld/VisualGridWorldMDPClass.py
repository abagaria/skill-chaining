# Python imports.
import numpy as np
from copy import deepcopy

# Other imports.
from simple_rl.tasks.gridworld.gridworld import GridWorld
from simple_rl.mdp import MDP
from simple_rl.tasks.gridworld.sensors import SensorChain, ResampleSensor, ImageSensor, NoisySensor, NullSensor, UnsqueezeSensor
from simple_rl.tasks.gridworld.VisualGridWorldStateClass import VisualGridWorldState


class VisualGridWorldMDP(MDP):
    def __init__(self, pixel_observation, grid_size, noise_level, seed):
        self.pixel_observation = pixel_observation
        self.grid_size = grid_size
        self.noise_level = noise_level
        self.seed = seed
        self.render = False

        np.random.seed(seed)

        assert grid_size in ("small", "medium", "large")

        if grid_size == "small":
            grid_size = 2
            self.env = GridWorld(grid_size, grid_size)
            scale = 14
            self.goal_loc = (1, 1)
        elif grid_size == "medium":
            grid_size = 4
            self.env = GridWorld(grid_size, grid_size)
            scale = 7
            self.goal_loc = (3, 3)
        else:
            grid_size = 14
            self.env = GridWorld(grid_size, grid_size)
            scale = 2
            self.goal_loc = (13, 13)

        if pixel_observation:
            sensors = [
                ImageSensor(range=((0, self.env._rows), (0, self.env._cols)), pixel_density=1),
                ResampleSensor(scale=scale),
                UnsqueezeSensor(dim=0)
            ]
            if noise_level > 0:
                sensors.append(NoisySensor(sigma=0.1))
            self.state_dim = (1, scale * grid_size, scale * grid_size)
        else:
            sensors = [NullSensor()]
            self.state_dim = 2

        self.sensor = SensorChain(sensors)

        self.env.reset_goal(self.goal_loc)
        self.env.reset_agent(init_loc=(0, 0))

        init_state = self.sensor.observe(self.env.agent.position)
        init_state = VisualGridWorldState(init_state)

        MDP.__init__(self, range(0, 4), self._transition_func, self._reward_func, init_state)

    def _reward_func(self, state, action):
        next_state, reward, done = self.env.step(action)
        next_obs = self.sensor.observe(next_state)

        self.next_state = VisualGridWorldState(next_obs, is_terminal=done)

        return reward + 1

    def _transition_func(self, state, action):
        return self.next_state

    def reset(self):
        self.env.reset_agent(init_loc=(0, 0))

    def __str__(self):
        return "visual-grid-world-mdp"

    def get_position(self):
        return deepcopy(self.env.agent.position)
