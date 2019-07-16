# Copyright 2018 The TensorFlow Authors All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Adapted from rllab maze_env.py."""

import os
import tempfile
import xml.etree.ElementTree as ET
import math
import numpy as np
import gym
import pdb

from simple_rl.tasks.point_maze.environments import maze_env_utils

# Directory that contains mujoco xml files.
# MODEL_DIR = '/Users/akhil/git-repos/models/research/efficient-hrl/environments/assets'
MODEL_DIR = 'simple_rl/tasks/point_maze/environments/assets'


class MazeEnv(gym.Env):
  MODEL_CLASS = None

  MAZE_HEIGHT = None
  MAZE_SIZE_SCALING = None

  def __init__(
      self,
      maze_id=None,
      train_mode=True,
      maze_height=0.5,
      maze_size_scaling=8,
      n_bins=0,
      sensor_range=3.,
      sensor_span=2 * math.pi,
      observe_blocks=False,
      put_spin_near_agent=False,
      top_down_view=False,
      manual_collision=False,
      *args,
      **kwargs):
    self._maze_id = maze_id

    model_cls = self.__class__.MODEL_CLASS
    if model_cls is None:
      raise "MODEL_CLASS unspecified!"
    filename = model_cls.FILE if train_mode else "point_test.xml"
    xml_path = os.path.join(MODEL_DIR, filename)
    tree = ET.parse(xml_path)
    worldbody = tree.find(".//worldbody")

    self.MAZE_HEIGHT = height = maze_height
    self.MAZE_SIZE_SCALING = size_scaling = maze_size_scaling
    self._n_bins = n_bins
    self._sensor_range = sensor_range * size_scaling
    self._sensor_span = sensor_span
    self._observe_blocks = observe_blocks
    self._put_spin_near_agent = put_spin_near_agent
    self._top_down_view = top_down_view
    self._manual_collision = manual_collision

    self.MAZE_STRUCTURE = structure = maze_env_utils.construct_maze(maze_id=self._maze_id)
    self.elevated = any(-1 in row for row in structure)  # Elevate the maze to allow for falling.
    self.blocks = any(
        any(maze_env_utils.can_move(r) for r in row)
        for row in structure)  # Are there any movable blocks?

    torso_x, torso_y = self._find_robot()
    self._init_torso_x = torso_x
    self._init_torso_y = torso_y
    self._init_positions = [
        (x - torso_x, y - torso_y)
        for x, y in self._find_all_robots()]

    self._xy_to_rowcol = lambda x, y: (2 + (y + size_scaling / 2) / size_scaling,
                                       2 + (x + size_scaling / 2) / size_scaling)
    self._view = np.zeros([5, 5, 3])  # walls (immovable), chasms (fall), movable blocks

    # Map specifying the index positions of all doors in the Maze Structure
    self.door_map = {
        "rooms_1_2": (4, 2),
        "rooms_1_4": (2, 4),
        "rooms_2_3": (6, 4),
        "rooms_3_4": (4, 6)
    }

    self.dt = 0.02            # Taken from point.xml
    self.max_distance = 100.  # Larger than the actual max distance
    self.max_angle = 180.     # degrees

    height_offset = 0.
    if self.elevated:
      # Increase initial z-pos of ant.
      height_offset = height * size_scaling
      torso = tree.find(".//body[@name='torso']")
      torso.set('pos', '0 0 %.2f' % (0.75 + height_offset))
    if self.blocks:
      # If there are movable blocks, change simulation settings to perform
      # better contact detection.
      default = tree.find(".//default")
      default.find('.//geom').set('solimp', '.995 .995 .01')

    self.movable_blocks = []
    for i in range(len(structure)):
      for j in range(len(structure[0])):
        struct = structure[i][j]
        if struct == 'r' and self._put_spin_near_agent:
          struct = maze_env_utils.Move.SpinXY
        if self.elevated and struct not in [-1]:
          # Create elevated platform.
          ET.SubElement(
              worldbody, "geom",
              name="elevated_%d_%d" % (i, j),
              pos="%f %f %f" % (j * size_scaling - torso_x,
                                i * size_scaling - torso_y,
                                height / 2 * size_scaling),
              size="%f %f %f" % (0.5 * size_scaling,
                                 0.5 * size_scaling,
                                 height / 2 * size_scaling),
              type="box",
              material="",
              contype="1",
              conaffinity="1",
              rgba="0.9 0.9 0.9 1",
          )
        if struct == 1:  # Unmovable block.
          # Offset all coordinates so that robot starts at the origin.
          ET.SubElement(
              worldbody, "geom",
              name="block_%d_%d" % (i, j),
              pos="%f %f %f" % (j * size_scaling - torso_x,
                                i * size_scaling - torso_y,
                                height_offset +
                                height / 2 * size_scaling),
              size="%f %f %f" % (0.5 * size_scaling,
                                 0.5 * size_scaling,
                                 height / 2 * size_scaling),
              type="box",
              material="",
              contype="1",
              conaffinity="1",
              rgba="0.4 0.4 0.4 1",
          )
        elif maze_env_utils.can_move(struct):  # Movable block.
          # The "falling" blocks are shrunk slightly and increased in mass to
          # ensure that it can fall easily through a gap in the platform blocks.
          name = "movable_%d_%d" % (i, j)
          self.movable_blocks.append((name, struct))
          falling = maze_env_utils.can_move_z(struct)
          spinning = maze_env_utils.can_spin(struct)
          x_offset = 0.25 * size_scaling if spinning else 0.0
          y_offset = 0.0
          shrink = 0.1 if spinning else 0.99 if falling else 1.0
          height_shrink = 0.1 if spinning else 1.0
          movable_body = ET.SubElement(
              worldbody, "body",
              name=name,
              pos="%f %f %f" % (j * size_scaling - torso_x + x_offset,
                                i * size_scaling - torso_y + y_offset,
                                height_offset +
                                height / 2 * size_scaling * height_shrink),
          )
          ET.SubElement(
              movable_body, "geom",
              name="block_%d_%d" % (i, j),
              pos="0 0 0",
              size="%f %f %f" % (0.5 * size_scaling * shrink,
                                 0.5 * size_scaling * shrink,
                                 height / 2 * size_scaling * height_shrink),
              type="box",
              material="",
              mass="0.001" if falling else "0.0002",
              contype="1",
              conaffinity="1",
              rgba="0.9 0.1 0.1 1"
          )
          if maze_env_utils.can_move_x(struct):
            ET.SubElement(
                movable_body, "joint",
                armature="0",
                axis="1 0 0",
                damping="0.0",
                limited="true" if falling else "false",
                range="%f %f" % (-size_scaling, size_scaling),
                margin="0.01",
                name="movable_x_%d_%d" % (i, j),
                pos="0 0 0",
                type="slide"
            )
          if maze_env_utils.can_move_y(struct):
            ET.SubElement(
                movable_body, "joint",
                armature="0",
                axis="0 1 0",
                damping="0.0",
                limited="true" if falling else "false",
                range="%f %f" % (-size_scaling, size_scaling),
                margin="0.01",
                name="movable_y_%d_%d" % (i, j),
                pos="0 0 0",
                type="slide"
            )
          if maze_env_utils.can_move_z(struct):
            ET.SubElement(
                movable_body, "joint",
                armature="0",
                axis="0 0 1",
                damping="0.0",
                limited="true",
                range="%f 0" % (-height_offset),
                margin="0.01",
                name="movable_z_%d_%d" % (i, j),
                pos="0 0 0",
                type="slide"
            )
          if maze_env_utils.can_spin(struct):
            ET.SubElement(
                movable_body, "joint",
                armature="0",
                axis="0 0 1",
                damping="0.0",
                limited="false",
                name="spinable_%d_%d" % (i, j),
                pos="0 0 0",
                type="ball"
            )

    torso = tree.find(".//body[@name='torso']")
    geoms = torso.findall(".//geom")
    for geom in geoms:
      if 'name' not in geom.attrib:
        raise Exception("Every geom of the torso must have a name "
                        "defined")

    goal_xy = tree.find(".//geom[@name='target']").attrib["pos"].split()
    self.goal_xy = np.array([float(goal_xy[0]), float(goal_xy[1])])

    key_xy = tree.find(".//geom[@name='key']").attrib["pos"].split()
    self.key_xy = np.array([float(key_xy[0]), float(key_xy[1])])

    print("Goal position: ", self.goal_xy)
    print("Key  position: ", self.key_xy)

    self.has_key = False

    _, file_path = tempfile.mkstemp(text=True, suffix=".xml")
    tree.write(file_path)

    self.wrapped_env = model_cls(*args, file_path=file_path, **kwargs)

    # Debug info
    door_coord_map = self._find_doors()
    center_x, center_y = self._get_center_point(door_coord_map)
    print("Door coord map: ", door_coord_map)
    print("center: ", center_x, center_y)

  def get_ori(self):
    return self.wrapped_env.get_ori()

  def get_top_down_view(self):
    self._view = np.zeros_like(self._view)

    def valid(row, col):
      return self._view.shape[0] > row >= 0 and self._view.shape[1] > col >= 0

    def update_view(x, y, d, row=None, col=None):
      if row is None or col is None:
        x = x - self._robot_x
        y = y - self._robot_y
        th = self._robot_ori

        row, col = self._xy_to_rowcol(x, y)
        update_view(x, y, d, row=row, col=col)
        return

      row, row_frac, col, col_frac = int(row), row % 1, int(col), col % 1
      if row_frac < 0:
        row_frac += 1
      if col_frac < 0:
        col_frac += 1

      if valid(row, col):
        self._view[row, col, d] += (
            (min(1., row_frac + 0.5) - max(0., row_frac - 0.5)) *
            (min(1., col_frac + 0.5) - max(0., col_frac - 0.5)))
      if valid(row - 1, col):
        self._view[row - 1, col, d] += (
            (max(0., 0.5 - row_frac)) *
            (min(1., col_frac + 0.5) - max(0., col_frac - 0.5)))
      if valid(row + 1, col):
        self._view[row + 1, col, d] += (
            (max(0., row_frac - 0.5)) *
            (min(1., col_frac + 0.5) - max(0., col_frac - 0.5)))
      if valid(row, col - 1):
        self._view[row, col - 1, d] += (
            (min(1., row_frac + 0.5) - max(0., row_frac - 0.5)) *
            (max(0., 0.5 - col_frac)))
      if valid(row, col + 1):
        self._view[row, col + 1, d] += (
            (min(1., row_frac + 0.5) - max(0., row_frac - 0.5)) *
            (max(0., col_frac - 0.5)))
      if valid(row - 1, col - 1):
        self._view[row - 1, col - 1, d] += (
            (max(0., 0.5 - row_frac)) * max(0., 0.5 - col_frac))
      if valid(row - 1, col + 1):
        self._view[row - 1, col + 1, d] += (
            (max(0., 0.5 - row_frac)) * max(0., col_frac - 0.5))
      if valid(row + 1, col + 1):
        self._view[row + 1, col + 1, d] += (
            (max(0., row_frac - 0.5)) * max(0., col_frac - 0.5))
      if valid(row + 1, col - 1):
        self._view[row + 1, col - 1, d] += (
            (max(0., row_frac - 0.5)) * max(0., 0.5 - col_frac))

    # Draw ant.
    robot_x, robot_y = self.wrapped_env.get_body_com("torso")[:2]
    self._robot_x = robot_x
    self._robot_y = robot_y
    self._robot_ori = self.get_ori()

    structure = self.MAZE_STRUCTURE
    size_scaling = self.MAZE_SIZE_SCALING
    height = self.MAZE_HEIGHT

    # Draw immovable blocks and chasms.
    for i in range(len(structure)):
      for j in range(len(structure[0])):
        if structure[i][j] == 1:  # Wall.
          update_view(j * size_scaling - self._init_torso_x,
                      i * size_scaling - self._init_torso_y,
                      0)
        if structure[i][j] == -1:  # Chasm.
          update_view(j * size_scaling - self._init_torso_x,
                      i * size_scaling - self._init_torso_y,
                      1)

    # Draw movable blocks.
    for block_name, block_type in self.movable_blocks:
      block_x, block_y = self.wrapped_env.get_body_com(block_name)[:2]
      update_view(block_x, block_y, 2)

    return self._view


  def _find_doors(self):
      """
      |--------------------|
      |  room4  |  room3   |
      |--------------------|
      |  room1  |  room2   |
      |--------------------|
      """
      door_coord_map = {
          "rooms_1_4" : (0., 4.),
          "rooms_1_2" : (4., 0.),
          "rooms_2_3" : (8., 4.),
          "rooms_3_4" : (4., 8.)
      }
      return door_coord_map

  @staticmethod
  def room_to_doors(room):
      if room == "room1":
          return "rooms_1_2", "rooms_1_4"
      if room == "room2":
          return "rooms_2_3", "rooms_1_2"
      if room == "room3":
          return "rooms_3_4", "rooms_2_3"
      if room == "room4":
          return "rooms_1_4", "rooms_3_4"
      raise ValueError("got room {}".format(room))

  def get_agent_room_doors(self):
      agent_room = self.get_agent_room()
      return self.room_to_doors(agent_room)

  @staticmethod
  def _get_center_point(door_coord_map):
      center_x = (door_coord_map["rooms_1_4"][0] + door_coord_map["rooms_2_3"][0]) / 2.
      center_y = (door_coord_map["rooms_1_2"][1] + door_coord_map["rooms_3_4"][1]) / 2.
      return center_x, center_y

  @staticmethod
  def _wrap_agent_orientation(theta):
      return ((np.deg2rad(theta) + np.pi) % (2 * np.pi ) - np.pi) * (180. / np.pi)

  def location_to_room(self, x, y):
      door_coord_map = self._find_doors()
      center_x, center_y = self._get_center_point(door_coord_map)

      if x <= center_x and y <= center_y:
          return "room1"
      if x >= center_x and y <= center_y:
          return "room2"
      if x <= center_x and y >= center_y:
          return "room4"
      if x >= center_x and y >= center_y:
          return "room3"

      pdb.set_trace()

  def get_key_distance(self, agent_position):
      distance_vector = self.key_xy - agent_position
      return np.linalg.norm(distance_vector)

  def get_lock_distance(self, agent_position):
      distance_vector = self.goal_xy - agent_position
      return np.linalg.norm(distance_vector)

  def get_lock_room_doors(self):
      lock_room = self.get_lock_room()
      return self.room_to_doors(lock_room)

  def get_adjacent_lock_room_door(self):
      if self.get_agent_room() != self.get_key_room():
          return None
      door1, door2 = self.get_agent_room_doors()
      lock_room_doors = self.get_lock_room_doors()
      if door1 in lock_room_doors:
          return door1
      elif door2 in lock_room_doors:
          return door2
      pdb.set_trace()  # How can we not be adjacent to the lock-room if we are in the key-room?

  def get_door_distances(self, agent_position):
      door1, door2 = self.get_agent_room_doors()
      door_coord_map = self._find_doors()

      # If we are in the key-room, we only get the distance to the lock-room
      adjacent_lock_room_door = self.get_adjacent_lock_room_door()
      if adjacent_lock_room_door is not None:
          lock_room_door_location = door_coord_map[adjacent_lock_room_door]
          lock_room_door_distance = np.linalg.norm(agent_position - lock_room_door_location)
          return lock_room_door_distance, self.max_distance

      door1_location = door_coord_map[door1]
      door2_location = door_coord_map[door2]
      door1_distance = np.linalg.norm(agent_position - door1_location)
      door2_distance = np.linalg.norm(agent_position - door2_location)
      return door1_distance, door2_distance

  def get_key_angle(self, agent_position, agent_orientation):
      distance_vector = self.key_xy - agent_position
      x_key, y_key = distance_vector[0], distance_vector[1]
      key_angle = np.arctan2(y_key, x_key) * 180 / np.pi
      agent_key_angle = self._wrap_agent_orientation(agent_orientation) - key_angle
      return self._wrap_agent_orientation(agent_key_angle)

  def get_lock_angle(self, agent_position, agent_orientation):
      distance_vector = self.goal_xy - agent_position
      x_lock, y_lock = distance_vector[0], distance_vector[1]
      lock_angle = np.arctan2(y_lock, x_lock) * 180 / np.pi
      agent_lock_angle = self._wrap_agent_orientation(agent_orientation) - lock_angle
      return self._wrap_agent_orientation(agent_lock_angle)

  def get_door_angles(self, agent_position, agent_orientation):
      def get_door_angle(door):
          door_location = door_coord_map[door]
          door_distance_vector = door_location - agent_position
          door_angle = np.arctan2(door_distance_vector[1], door_distance_vector[0]) * 180 / np.pi
          agent_door_angle = door_angle - self._wrap_agent_orientation(agent_orientation)
          return self._wrap_agent_orientation(agent_door_angle)
      door1, door2 = self.get_agent_room_doors()
      door_coord_map = self._find_doors()

      # If we are in the key-room, we only get the distance to the lock-room
      adjacent_lock_room_door = self.get_adjacent_lock_room_door()
      if adjacent_lock_room_door is not None:
          agent_lock_room_door_angle = get_door_angle(adjacent_lock_room_door)
          return agent_lock_room_door_angle, self.max_angle

      agent_door1_angle = get_door_angle(door1)
      agent_door2_angle = get_door_angle(door2)
      return agent_door1_angle, agent_door2_angle

  def get_egocentric_obs(self):
      key_obs = self.get_key_obs()                 # [d, theta, v, omega]
      lock_obs = self.get_lock_obs()               # [d, theta, v, omega]
      door_obs = self.get_door_obs()               # [d1, d2, theta1, theta2, v1, v2, omega1, omega2]
      return door_obs + key_obs + lock_obs + [self.has_key]

  def get_key_obs(self):
      key_room = self.get_key_room()
      agent_room = self.get_agent_room()
      if key_room == agent_room:

          agent_previous_position = self.previous_agent_xy
          agent_previous_orientation = self.previous_agent_ori
          agent_current_position = self.wrapped_env.get_xy()
          agent_current_orientation = self.wrapped_env.get_ori()

          current_key_distance = self.get_key_distance(agent_current_position)
          current_key_angle = self.get_key_angle(agent_current_position, agent_current_orientation)

          previous_key_distance = self.get_key_distance(agent_previous_position)
          previous_key_angle = self.get_key_angle(agent_previous_position, agent_previous_orientation)
          linear_velocity = (current_key_distance - previous_key_distance) / self.dt
          angular_velocity = (current_key_angle - previous_key_angle) / self.dt

          return [current_key_distance, current_key_angle, linear_velocity, angular_velocity]

      return [self.max_distance, self.max_angle, 0., 0.]

  def get_lock_obs(self):
      lock_room = self.get_lock_room()
      agent_room = self.get_agent_room()
      if lock_room == agent_room:

          agent_previous_position = self.previous_agent_xy
          agent_previous_orientation = self.previous_agent_ori
          agent_current_position = self.wrapped_env.get_xy()
          agent_current_orientation = self.wrapped_env.get_ori()

          current_lock_distance = self.get_lock_distance(agent_current_position)
          current_lock_angle = self.get_lock_angle(agent_current_position, agent_current_orientation)

          previous_lock_distance = self.get_lock_distance(agent_previous_position)
          previous_lock_angle = self.get_lock_angle(agent_previous_position, agent_previous_orientation)
          linear_velocity = (current_lock_distance - previous_lock_distance) / self.dt
          angular_velocity = (current_lock_angle - previous_lock_angle) / self.dt

          return [current_lock_distance, current_lock_angle, linear_velocity, angular_velocity]

      return [self.max_distance, self.max_angle, 0., 0.]

  def get_door_obs(self):
      agent_previous_position = self.previous_agent_xy
      agent_previous_orientation = self.previous_agent_ori
      agent_current_position = self.wrapped_env.get_xy()
      agent_current_orientation = self.wrapped_env.get_ori()

      current_door_distances = self.get_door_distances(agent_current_position)
      current_door_angles = self.get_door_angles(agent_current_position, agent_current_orientation)

      agent_current_room = self.get_agent_room()
      agent_previous_room = self.location_to_room(agent_previous_position[0], agent_previous_position[1])

      # If the agent has not changed rooms, compute and return the door related velocities
      if agent_current_room == agent_previous_room:
          previous_door_distances = self.get_door_distances(agent_previous_position)
          previous_door_angles = self.get_door_angles(agent_previous_position, agent_previous_orientation)

          door1_linear_velocity = (current_door_distances[0] - previous_door_distances[0]) / self.dt
          door2_linear_velocity = (current_door_distances[1] - previous_door_distances[1]) / self.dt
          door1_angular_velocity = (current_door_angles[0] - previous_door_angles[0]) / self.dt
          door2_angular_velocity = (current_door_angles[1] - previous_door_angles[1]) / self.dt

          return [current_door_distances[0], current_door_distances[1], current_door_angles[0], current_door_angles[1],
                  door1_linear_velocity, door2_linear_velocity, door1_angular_velocity, door2_angular_velocity]

      # If the agent has just transitioned b/w rooms, the velocity to the current room's doors is undefined
      return [current_door_distances[0], current_door_distances[1], current_door_angles[0], current_door_angles[1], 0., 0., 0., 0.]

  def get_agent_room(self):
      agent_position = self.wrapped_env.get_xy()
      agent_room = self.location_to_room(agent_position[0], agent_position[1])
      return agent_room

  def get_key_room(self):
      key_position = self.key_xy
      key_room = self.location_to_room(key_position[0], key_position[1])
      return key_room

  def get_lock_room(self):
      lock_position = self.goal_xy
      lock_room = self.location_to_room(lock_position[0], lock_position[1])
      return lock_room

  def get_range_sensor_obs(self):
    """Returns egocentric range sensor observations of maze."""
    robot_x, robot_y, robot_z = self.wrapped_env.get_body_com("torso")[:3]
    ori = self.get_ori()

    structure = self.MAZE_STRUCTURE
    size_scaling = self.MAZE_SIZE_SCALING
    height = self.MAZE_HEIGHT

    segments = []
    # Get line segments (corresponding to outer boundary) of each immovable
    # block or drop-off.
    for i in range(len(structure)):
      for j in range(len(structure[0])):
        if structure[i][j] in [1, -1]:  # There's a wall or drop-off.
          cx = j * size_scaling - self._init_torso_x
          cy = i * size_scaling - self._init_torso_y
          x1 = cx - 0.5 * size_scaling
          x2 = cx + 0.5 * size_scaling
          y1 = cy - 0.5 * size_scaling
          y2 = cy + 0.5 * size_scaling
          struct_segments = [
              ((x1, y1), (x2, y1)),
              ((x2, y1), (x2, y2)),
              ((x2, y2), (x1, y2)),
              ((x1, y2), (x1, y1)),
          ]
          for seg in struct_segments:
            segments.append(dict(
                segment=seg,
                type=structure[i][j],
            ))
    # Get line segments (corresponding to outer boundary) of each movable
    # block within the agent's z-view.
    for block_name, block_type in self.movable_blocks:
      block_x, block_y, block_z = self.wrapped_env.get_body_com(block_name)[:3]
      if (block_z + height * size_scaling / 2 >= robot_z and
          robot_z >= block_z - height * size_scaling / 2):  # Block in view.
        x1 = block_x - 0.5 * size_scaling
        x2 = block_x + 0.5 * size_scaling
        y1 = block_y - 0.5 * size_scaling
        y2 = block_y + 0.5 * size_scaling
        struct_segments = [
            ((x1, y1), (x2, y1)),
            ((x2, y1), (x2, y2)),
            ((x2, y2), (x1, y2)),
            ((x1, y2), (x1, y1)),
        ]
        for seg in struct_segments:
          segments.append(dict(
              segment=seg,
              type=block_type,
          ))

    sensor_readings = np.zeros((self._n_bins, 3))  # 3 for wall, drop-off, block
    for ray_idx in range(self._n_bins):
      ray_ori = (ori - self._sensor_span * 0.5 +
                 (2 * ray_idx + 1.0) / (2 * self._n_bins) * self._sensor_span)
      ray_segments = []
      # Get all segments that intersect with ray.
      for seg in segments:
        p = maze_env_utils.ray_segment_intersect(
            ray=((robot_x, robot_y), ray_ori),
            segment=seg["segment"])
        if p is not None:
          ray_segments.append(dict(
              segment=seg["segment"],
              type=seg["type"],
              ray_ori=ray_ori,
              distance=maze_env_utils.point_distance(p, (robot_x, robot_y)),
          ))
      if len(ray_segments) > 0:
        # Find out which segment is intersected first.
        first_seg = sorted(ray_segments, key=lambda x: x["distance"])[0]
        seg_type = first_seg["type"]
        idx = (0 if seg_type == 1 else  # Wall.
               1 if seg_type == -1 else  # Drop-off.
               2 if maze_env_utils.can_move(seg_type) else  # Block.
               None)
        if first_seg["distance"] <= self._sensor_range:
          sensor_readings[ray_idx][idx] = (self._sensor_range - first_seg["distance"]) / self._sensor_range

    return sensor_readings

  def _get_obs(self):
    wrapped_obs = self.wrapped_env._get_obs()
    if self._top_down_view:
      view = [self.get_top_down_view().flat]
    else:
      view = []

    if self._observe_blocks:
      additional_obs = []
      for block_name, block_type in self.movable_blocks:
        additional_obs.append(self.wrapped_env.get_body_com(block_name))
      wrapped_obs = np.concatenate([wrapped_obs[:3]] + additional_obs +
                                   [wrapped_obs[3:]])

    wrapped_obs = np.hstack([wrapped_obs[:2]] + [self.has_key] + [wrapped_obs[2:]])
    range_sensor_obs = self.get_range_sensor_obs()
    return np.concatenate([wrapped_obs,
                           range_sensor_obs.flat] +
                           view + [[self.t * 0.001]])

  def reset(self):
    self.t = 0
    self.trajectory = []
    self.wrapped_env.reset()
    self.has_key = False
    self.previous_agent_xy = np.array([0., 0.])
    self.previous_agent_ori = 0.
    init_pspace = self._get_obs()
    init_aspace = self.get_egocentric_obs()
    return init_pspace, init_aspace

  @property
  def viewer(self):
    return self.wrapped_env.viewer

  def render(self, *args, **kwargs):
    return self.wrapped_env.render(*args, **kwargs)

  @property
  def observation_space(self):
    shape = self._get_obs().shape
    high = np.inf * np.ones(shape)
    low = -high
    return gym.spaces.Box(low, high)

  @property
  def action_space(self):
    return self.wrapped_env.action_space

  def _find_robot(self):
    structure = self.MAZE_STRUCTURE
    size_scaling = self.MAZE_SIZE_SCALING
    for i in range(len(structure)):
      for j in range(len(structure[0])):
        if structure[i][j] == 'r':
          return j * size_scaling, i * size_scaling
    assert False, 'No robot in maze specification.'

  def _find_all_robots(self):
    structure = self.MAZE_STRUCTURE
    size_scaling = self.MAZE_SIZE_SCALING
    coords = []
    for i in range(len(structure)):
      for j in range(len(structure[0])):
        if structure[i][j] == 'r':
          coords.append((j * size_scaling, i * size_scaling))
    return coords

  def _is_in_collision(self, pos):
    x, y = pos
    structure = self.MAZE_STRUCTURE
    size_scaling = self.MAZE_SIZE_SCALING
    for i in range(len(structure)):
      for j in range(len(structure[0])):
        if structure[i][j] == 1:
          minx = j * size_scaling - size_scaling * 0.5 - self._init_torso_x
          maxx = j * size_scaling + size_scaling * 0.5 - self._init_torso_x
          miny = i * size_scaling - size_scaling * 0.5 - self._init_torso_y
          maxy = i * size_scaling + size_scaling * 0.5 - self._init_torso_y
          if minx <= x <= maxx and miny <= y <= maxy:
            return True
    return False

  def is_in_goal_position(self, pos):
    return self.distance_to_goal_position(pos) <= 0.6

  def is_in_key_position(self, pos):
      return self.distance_to_key_position(pos) <= 0.6

  def distance_to_goal_position(self, pos):
    return np.linalg.norm(pos - self.goal_xy)

  def distance_to_key_position(self, pos):
      return np.linalg.norm(pos - self.key_xy)

  def step(self, action):
    outer_reward = 0.
    done = False

    self.t += 1

    if self._manual_collision:
      old_pos = self.wrapped_env.get_xy()
      self.previous_agent_xy = np.copy(old_pos)
      self.previous_agent_ori = np.copy(self.wrapped_env.get_ori())
      inner_next_obs, inner_reward, done, info = self.wrapped_env.step(action)
      new_pos = self.wrapped_env.get_xy()
      if self._is_in_collision(new_pos):
        self.wrapped_env.set_xy(old_pos)

      if self.is_in_key_position(new_pos):
          self.has_key = True

      # Point-env will give -1 reward for every step
      # Reaching the goal neutralizes the step penalty
      if self.is_in_goal_position(new_pos):
        if self.has_key:
            outer_reward = 1.
            done = True
    else:
      inner_next_obs, inner_reward, done, info = self.wrapped_env.step(action)
    next_obs = self._get_obs()
    next_ego_obs = self.get_egocentric_obs()
    return next_obs, next_ego_obs, inner_reward + outer_reward, done, info
