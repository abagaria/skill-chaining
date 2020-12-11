import numpy as np
import ipdb
import math

from gym import utils
from gym.envs.mujoco import mujoco_env


class SwimmerEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    FILE = "swimmer.xml"
    ORI_IND = 2

    def __init__(self, file_path):
        mujoco_env.MujocoEnv.__init__(self, file_path, 4) # was 4
        utils.EzPickle.__init__(self)

    def step(self, a):
        ctrl_cost_coeff = 0.0001
        xposbefore = self.sim.data.qpos[0]
        self.do_simulation(a, self.frame_skip)
        xposafter = self.sim.data.qpos[0]
        reward_fwd = (xposafter - xposbefore) / self.dt
        reward_ctrl = - ctrl_cost_coeff * np.square(a).sum()
        reward = reward_fwd + reward_ctrl
        ob = self._get_obs()
        return ob, reward, False, dict(reward_fwd=reward_fwd, reward_ctrl=reward_ctrl)

    def _get_obs(self):
        """
        The Mujoco _get_obs does not return the swimmer's position (qpos[:2]), but
        this method has been modified to return position as well as joint angles.
        """
        qpos = self.sim.data.qpos
        qvel = self.sim.data.qvel
        return np.concatenate([qpos.flat, qvel.flat])

    def reset_model(self):
        self.set_state(
            self.init_qpos + self.np_random.uniform(low=-.1, high=.1, size=self.model.nq),
            self.init_qvel + self.np_random.uniform(low=-.1, high=.1, size=self.model.nv)
        )
        self.set_orientation(np.random.uniform(-math.pi, math.pi)) # set header
        return self._get_obs()

    def get_ori(self):
        return self.sim.data.qpos[self.__class__.ORI_IND]

    def set_xy(self, xy):
        qpos = self.sim.data.qpos
        qpos[0] = xy[0]
        qpos[1] = xy[1]
        qvel = self.sim.data.qvel
        self.set_state(qpos, qvel)

    def set_orientation(self, ori):
        qpos = self.sim.data.qpos
        qpos[self.__class__.ORI_IND] = ori
        qvel = self.sim.data.qvel
        self.set_state(qpos, qvel)

    def get_xy(self):
        return self.sim.data.qpos[:2]