# Python imports.
import numpy as np

# Other imports.
from simple_rl.tasks.gridworld.gridworld import GridWorld
from simple_rl.mdp import MDP
from simple_rl.tasks.gridworld.sensors import SensorChain, ResampleSensor, ImageSensor, NoisySensor, NullSensor
from simple_rl.tasks.gridworld.VisualGridWorldStateClass import VisualGridWorldState


class VisualGridWorldMDP(MDP):
    def __init__(self, pixel_observation, use_small_grid, noise_level, seed):
        self.pixel_observation = pixel_observation
        self.use_small_grid = use_small_grid
        self.noise_level = noise_level
        self.seed = seed
        self.render = False

        np.random.seed(seed)

        if use_small_grid:
            grid_size = 2
            self.env = GridWorld(grid_size, grid_size)
            scale = 14
            self.goal_loc = (1, 1)
        else:
            grid_size = 4
            self.env = GridWorld(grid_size, grid_size)
            scale = 7
            self.goal_loc = (3, 3)

        if pixel_observation:
            sensors = [
                ImageSensor(range=((0, self.env._rows), (0, self.env._cols)), pixel_density=1),
                ResampleSensor(scale=scale),
            ]
            if noise_level > 0:
                sensors.append(NoisySensor(sigma=0.1))
            self.state_dim = (scale * grid_size, scale * grid_size)
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

        return reward

    def _transition_func(self, state, action):
        return self.next_state

    def reset(self):
        self.env.reset_agent(init_loc=(0, 0))

    def __str__(self):
        return "visual-grid-world-mdp"

    def get_position(self):
        return self.env.agent.position
