import imageio
import os
import sys
import argparse
from pathlib import Path

def create_clf_gifs(opt_ids, num_frames, duration, plot_dir_path, gif_dir_path):
    # Create directory
    Path(gif_dir_path).mkdir(exist_ok=True)

    # Generate gifs
    for i, opt_id in enumerate(opt_ids):
        gif_path = gif_dir_path + '/' + 'option_{}.gif'.format(opt_id)
        with imageio.get_writer(gif_path, mode='I', duration=duration) as writer:
            for j in range(1, num_frames[i]+1):
                frames_path = plot_dir_path + '/' + \
                    'option_{}-{}.png'.format(opt_id, j)
                writer.append_data(imageio.imread(frames_path))


def create_state_prob_gifs(opt_ids, num_frames, duration, plot_dir_path, gif_dir_path):
    # Create directory
    Path(gif_dir_path).mkdir(exist_ok=True)
    
    # Generate gifs
    clfs = ['initiation_classifier', 'termination_classifier']
    for clf in clfs:
        for i, opt_id in enumerate(opt_ids):
            gif_path = gif_dir_path + '/' + 'option_{}_{}.gif'.format(opt_id, clf)
            with imageio.get_writer(gif_path, mode='I', duration=duration) as writer:
                for j in range(1, num_frames[i]+1):
                    frames_path = plot_dir_path + '/' + \
                        'option_{}_{}_{}.png'.format(opt_id, clf, j)
                    writer.append_data(imageio.imread(frames_path))

def main(args):
    # Create gifs
    create_clf_gifs(args.opt_ids, args.num_frames, duration=0.5, plot_dir_path = '{}/clf_plots'.format(args.test_path), gif_dir_path='{}/clf_gifs'.format(args.test_path))
    create_state_prob_gifs(args.opt_ids, args.num_frames, duration = 0.5, plot_dir_path = '{}/state_prob_plots'.format(args.test_path), gif_dir_path = '{}/state_prob_gifs'.format(args.test_path))

if __name__ == "__main__":
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
    parser.add_argument(
        '--test_path',
        type=str
    )

    main(parser.parse_args())
