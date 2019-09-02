"""An observation wrapper that augments observations by pixel values."""

import numpy as np

from gym import spaces
from gym.spaces import Box
from gym import ObservationWrapper
import pdb


class PixelObservationWrapper(ObservationWrapper):
    """Augment observations by pixel values."""

    def __init__(self, env):
        """Initializes a new pixel Wrapper.

        Args:
            env: The environment to wrap.
        """

        super(PixelObservationWrapper, self).__init__(env)

        render_kwargs = {'mode': 'rgb_array'}

        # Extend observation space with pixels.
        pixels = self.env.render(**render_kwargs)

        if np.issubdtype(pixels.dtype, np.integer):
            low, high = (0, 255)
        elif np.issubdtype(pixels.dtype, np.float):
            low, high = (-float('inf'), float('inf'))
        else:
            raise TypeError(pixels.dtype)

        self.observation_space = spaces.Box(shape=pixels.shape, low=low, high=high, dtype=pixels.dtype)

        self._env = env
        self._render_kwargs = render_kwargs

    def observation(self, observation):
        return self.env.render(**self._render_kwargs)


class ResizeObservation(ObservationWrapper):
    r"""Downsample the image observation to a square image. """
    def __init__(self, env, shape):
        super(ResizeObservation, self).__init__(env)
        if isinstance(shape, int):
            shape = (shape, shape)
        assert all(x > 0 for x in shape), shape
        self.shape = tuple(shape)

        obs_shape = self.shape + self.observation_space.shape[2:]
        self.observation_space = Box(low=0, high=255, shape=obs_shape, dtype=np.uint8)

    def observation(self, observation):
        import cv2
        observation = cv2.resize(observation, self.shape[::-1], interpolation=cv2.INTER_AREA)
        if observation.ndim == 2:
            observation = np.expand_dims(observation, -1)
        return observation


class GrayScaleObservation(ObservationWrapper):
    r"""Convert the image observation from RGB to gray scale. """
    def __init__(self, env, keep_dim=False):
        super(GrayScaleObservation, self).__init__(env)
        self.keep_dim = keep_dim

        assert len(env.observation_space.shape) == 3 and env.observation_space.shape[-1] == 3, env.observation_space.shape
        obs_shape = self.observation_space.shape[:2]
        if self.keep_dim:
            self.observation_space = Box(low=0, high=255, shape=(obs_shape[0], obs_shape[1], 1), dtype=np.uint8)
        else:
            self.observation_space = Box(low=0, high=255, shape=obs_shape, dtype=np.uint8)

    def observation(self, observation):
        import cv2
        observation = cv2.cvtColor(observation, cv2.COLOR_RGB2GRAY)
        if self.keep_dim:
            observation = np.expand_dims(observation, -1)
        return observation