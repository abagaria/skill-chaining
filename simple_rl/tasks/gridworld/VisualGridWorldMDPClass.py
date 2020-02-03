# Python imports.
import numpy as np

# Other imports.
from simple_rl.tasks.gridworld.gridworld import GridWorld
from simple_rl.mdp import MDP
from simple_rl.tasks.gridworld.sensors import SensorChain, ResampleSensor, ImageSensor, NoisySensor, NullSensor


class VisualGridWorldMDP(MDP):
    def __init__(self, pixel_observation, use_small_grid, noise_level, seed):
        self.pixel_observation = pixel_observation
        self.use_small_grid = use_small_grid
        self.noise_level = noise_level
        self.seed = seed

        np.random.seed(seed)

        if use_small_grid:
            self.env = GridWorld(2, 2)
            scale = 14
        else:
            self.env = GridWorld(4, 4)
            scale = 7

        if pixel_observation:
            sensors = [
                ImageSensor(range=((0, self.env._rows), (0, self.env._cols)), pixel_density=1),
                ResampleSensor(scale=scale),
            ]
            if noise_level > 0:
                sensors.append(NoisySensor(sigma=0.1))
        else:
            sensors = [NullSensor()]

        self.sensor = SensorChain(sensors)

        self.env.reset_agent()
        self.env.reset_goal()
        init_state = self.env.agent.position

        MDP.__init__(self, range(1, 4), self._transition_func, self._reward_func, init_state)

    def _reward_func(self, state, action):
        next_state, reward, done = self.env.step(action)
        next_obs = self.sensor.observe(next_state)

        self.next_state = next_obs

        return reward


    def _transition_func(self, state, action):
        return self.next_state

    def reset(self):

