import pickle
import numpy as np
import gym
import itertools
from simple_rl.tasks.d4rl_ant_maze.D4RLAntMazeMDPClass import D4RLAntMazeMDP
from PIL import Image
import xml.etree.ElementTree as ET
import os
import ipdb
from simple_rl.tasks.ant_four_rooms.AntFourRoomsMDPClass import AntFourRoomsMDP

def change_color(color_str): 
    # TODO change XML path
    xml_path = '/home/jsenthil/d4rl/d4rl/locomotion/assets/ant.xml'
    tree = ET.parse(xml_path) 

    torso = tree.find(".//body[@name='torso']")
    geoms = torso.findall(".//geom")

    if color_str:
        print(f"Changing ant color to {color_str}")
        for geom in geoms:
            geom.attrib["rgba"] = color_str
    tree.write(xml_path)
    
    # TODO if color change doesn't work, uncomment the following lines (and set to your path)
    # os.chdir('/home/abagaria/d4rl')
    # os.system("pip install -e .")

    # os.chdir('/home/jsenthil/skill-chaining')

### custom _read_pixels_as_in_window because env.viewer._read_pixels_as_in_window() is low res
import copy
from mujoco_py.utils import rec_copy, rec_assign
def _read_pixels_as_in_window(resolution = (2200,2000)):
        if env.viewer.sim._render_context_offscreen is None:
            env.viewer.sim.render(*resolution)
        offscreen_ctx = env.viewer.sim._render_context_offscreen
        window_ctx = env.viewer.sim._render_context_window
        # Save markers and overlay from offscreen.
        saved = [copy.deepcopy(offscreen_ctx._markers),
                 copy.deepcopy(offscreen_ctx._overlay),
                 rec_copy(offscreen_ctx.cam)]
        # Copy markers and overlay from window.
        offscreen_ctx._markers[:] = window_ctx._markers[:]
        offscreen_ctx._overlay.clear()
        offscreen_ctx._overlay.update(window_ctx._overlay)
        rec_assign(offscreen_ctx.cam, rec_copy(window_ctx.cam))

        img = env.viewer.sim.render(*resolution)
        # Restore markers and overlay to offscreen.
        offscreen_ctx._markers[:] = saved[0][:]
        offscreen_ctx._overlay.clear()
        offscreen_ctx._overlay.update(saved[1])
        rec_assign(offscreen_ctx.cam, saved[2])
        return np.flip(img,0)        

def reset():
    # TODO change env here
    mdp = D4RLAntMazeMDP("umaze")
    env = mdp.env
    env.render()
    env.viewer._read_pixels_as_in_window = _read_pixels_as_in_window
    return env

# TODO load your trajectory pickle file here
with open("umaze-traj.pkl", "rb") as f:
    transitions = pickle.load(f)

colors = ["0.5 0 0 1", "0.9 0.09 0.3 1", "0.96 0.5 0.18 1", "1 1 0.1 1", "0.8 0.98 0.23 1", "0.23 0.7 0.3 1", "0.27 0.94 0.94 1", "0 0.5 0.8 1", "0.56 0.11 0.7 1", "0.96 0.2 0.9 1", "0 0 0 1", "1 1 1 1"] 
import random
random.shuffle(colors)

env = reset()
i = 0
for _, traj in transitions:
    change_color(colors[-1])
    colors.pop()

    from simple_rl.tasks.d4rl_ant_maze.D4RLAntMazeMDPClass import D4RLAntMazeMDP
    env = reset() 
    
    for t in traj:
        print(i)
        s = t[-1].features()
        qpos = s[:15]
        qvel = s[15:]
        # env.env.set_state(qpos, qvel)
        env.wrapped_env.set_state(qpos, qvel)
        image = env.viewer._read_pixels_as_in_window()
        Image.fromarray(image).save(f'video/test_{i}.png')
        i += 1

# TODO
# in the ant.py file, this is an example configuration you can set
# for your viewer_setup() method:

# def viewer_setup(self):
#     self.viewer.cam.distance = self.model.stat.extent * 1.5
#     self.viewer.cam.elevation = -65
#     self.viewer.cam.lookat[0] += 5         # x,y,z offset from the object (works if trackbodyid=-1)
#     self.viewer.cam.lookat[1] += 3
#     self.viewer.cam.lookat[2] += 0

# You can move the elevation/xyz position to however you see fit