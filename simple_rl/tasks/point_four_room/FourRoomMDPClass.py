import math
import numpy as np
from gym import utils
from gym.envs.mujoco import mujoco_env
import pdb
import xml.etree.ElementTree as ET

from simple_rl.mdp import MDP

class FourRoomMDP(MDP):
    def __init__(self):
        pass

    def _transition_func(self, state, action):
        pass

    def _reward_func(self, state, action):
        pass

    def execute_agent_action(self, action, option_idx=None):
        pass

    def reset(self):
        pass

    def is_goal_state(self, state):
        pass

    def distance_to_goal(self, position):
        pass

    def state_space_size(self):
        pass

    def action_space_size(self):
        pass

    def action_space_bound(self):
        pass

    def is_primitive_action(self, action):
        pass

    def __str__(self):
        pass

class PointEnv(mujoco_env.MujocoEnv, utils.EzPickle):
  FILE = "/home/abagaria/git-repos/skill-chaining/simple_rl/tasks/point_four_room/four_room.xml"
  ORI_IND = 2

  def __init__(self, expose_all_qpos=True, frame_skip=4):
    self._expose_all_qpos = expose_all_qpos
    file_path = self.FILE
    tree = ET.parse(file_path)

    goal_xy = tree.find(".//geom[@name='target']").attrib["pos"].split()
    self.goal_xy = np.array([float(goal_xy[0]), float(goal_xy[1])])

    print("Goal position: ", self.goal_xy)

    mujoco_env.MujocoEnv.__init__(self, file_path, frame_skip)
    utils.EzPickle.__init__(self)

  @property
  def physics(self):
    return self.sim

  def get_goal_position(self):
      return self.goal_xy

  def _step(self, a):
    return self.step(a)

  def step(self, action):
    # action[0] = 0.2 * action[0]
    self.physics.data.ctrl[:] = action
    for _ in range(0, self.frame_skip):
      self.physics.step()
    next_obs = self._get_obs()

    reward = -1.
    done = False
    info = {}
    return next_obs, reward, done, info

  def _get_obs(self):
    if self._expose_all_qpos:
      return np.concatenate([
          self.physics.data.qpos.flat[:3],  # Only point-relevant coords.
          self.physics.data.qvel.flat[:3]])
    return np.concatenate([
        self.physics.data.qpos.flat[2:3],
        self.physics.data.qvel.flat[:3]])

  def reset_model(self):
    qpos = self.init_qpos + self.np_random.uniform(
        size=self.physics.model.nq, low=-.1, high=.1)
    qvel = self.init_qvel + self.np_random.randn(self.physics.model.nv) * .1

    # Set everything other than point to original position and 0 velocity.
    qpos[3:] = self.init_qpos[3:]
    qvel[3:] = 0.
    self.set_state(qpos, qvel)
    return self._get_obs()

  def get_ori(self):
    return self.physics.data.qpos[self.__class__.ORI_IND]

  def set_xy(self, xy):
    qpos = np.copy(self.physics.data.qpos)
    qpos[0] = xy[0]
    qpos[1] = xy[1]

    qvel = self.physics.data.qvel
    self.set_state(qpos, qvel)

  def get_xy(self):
    qpos = np.copy(self.physics.data.qpos)
    return qpos[:2]
