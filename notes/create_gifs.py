import imageio
import os
import sys
import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
    '--opt_ids',
    nargs='*',
    type=int
)
parser.add_argument(
    '--num_frames',
    nargs='*',
    type=int
)

# Image info
args = parser.parse_args()
opt_ids, num_frames = args.opt_ids, args.num_frames
clfs = ['tc_svm', 'oc_svm']
dir_path = 'clf_plots/'
gif_dir_path = 'clf_gifs/'

# Gif info
duration = 0.5

for i, opt_id in enumerate(opt_ids):
    for clf in clfs:
        gif_path = gif_dir_path + 'option_{}-{}.gif'.format(opt_id, clf)
        with imageio.get_writer(gif_path, mode='I', duration=duration) as writer:
            for j in range(1, num_frames[i]+1):
                frames_path = dir_path + \
                    'option_{}-{}-{}.png'.format(opt_id, clf, j)
                writer.append_data(imageio.imread(frames_path))

