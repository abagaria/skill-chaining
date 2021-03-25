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

def change_goal(x, y):
    # TODO change xml path
    xml_path = '/home/jsenthil/d4rl/d4rl/locomotion/assets/ant.xml'
    tree = ET.parse(xml_path) 

    geoms = tree.findall(".//geom")

    geoms[-1].attrib["pos"] = f"{x} {y} 0"
    tree.write(xml_path)

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
    mdp = D4RLAntMazeMDP("large")
    env = mdp.env
    env.render()
    env.viewer._read_pixels_as_in_window = _read_pixels_as_in_window
    return env

# TODO load your trajectory pickle file here
with open("icml-trajectories/start-to-top-right-trajectory.pkl", "rb") as f:
    first = pickle.load(f)

with open("icml-trajectories/top-right-to-bottom-right-trajectory.pkl", "rb") as f:
    second = pickle.load(f)

with open("icml-trajectories/bottom-right-to-top-left-trajectory.pkl", "rb") as f:
    third = pickle.load(f)

with open("icml-trajectories/top-left-to-start-trajectory.pkl", "rb") as f:
    fourth = pickle.load(f)

overall = [first, second, third, fourth]

colors = ["0.5 0 0 1", "0.9 0.09 0.3 1", "0.96 0.5 0.18 1", "1 1 0.1 1", "0.8 0.98 0.23 1", 
          "0.23 0.7 0.3 1", "0.27 0.94 0.94 1", "0 0.5 0.8 1", "0.56 0.11 0.7 1", 
          "0.1 0.2 0.3 1", "0.4 0.9 0.5 1", "0.7 0.4 0.9 1", "1 0.5 0.6 1",
          "0.96 0.2 0.9 1", "0 0 0 1", "0.5 0.4 0.2 0.8", "0.4 0.3 0.1 0.9", "0.4 0.3 0.6 0.9",
          "0.4 0.8 0.2 0.9", "0.33 0.66 0.99 0.9", "0.24 0.36 0.12 0.9", "0.68 0.48 0.38 0.9", "0.17 0.67 0.37 0.9"] 

env = reset()
i = 0
for transitions in overall:
    pos = transitions[-1][1][-1][-1].position
    change_goal(pos[0], pos[1])
    print("***changing position***")
    j = 0
    for _, traj in transitions:
        change_color(colors[j % len(colors)])

        from simple_rl.tasks.d4rl_ant_maze.D4RLAntMazeMDPClass import D4RLAntMazeMDP
        env = reset() 
        
        for t in traj:
            print(i)
            s = t[-1].features()
            qpos = s[:15]
            qvel = s[15:]
            env.wrapped_env.set_state(qpos, qvel)
            image = env.viewer._read_pixels_as_in_window()
            Image.fromarray(image).save(f'video/test_{i}.png')
            i += 1
        j += 1