import imageio
import os
import sys
import argparse
from pathlib import Path
import json

def json_to_dict(json_file):
    new_dict = {}
    for k,v in json_file.items():
        new_dict[k] = [int(item) for item in v.split(',')]
    return new_dict

def create_clf_gifs(opt_dict, duration, plot_dir_path, gif_dir_path):
    # Create directory
    Path(gif_dir_path).mkdir(exist_ok=True)

    # Generate gifs
    for opt_name, opt_range in opt_dict.items():
        gif_path = gif_dir_path + '/{}.gif'.format(opt_name)
        start, end = opt_range
        with imageio.get_writer(gif_path, mode='I', duration=duration) as writer:
            for i in range(start, end+1):
                frames_path = plot_dir_path + '/{}_{}.png'.format(opt_name, i)
                writer.append_data(imageio.imread(frames_path))

def create_state_prob_gifs(opt_dict, duration, plot_dir_path, gif_dir_path):
    # Create directory
    Path(gif_dir_path).mkdir(exist_ok=True)
    
    # Generate gifs
    clfs = ['initiation_classifier', 'pessimistic_classifier']
    for clf in clfs:
        for opt_name, opt_range in opt_dict.items():
            gif_path = gif_dir_path + '/{}_{}.gif'.format(opt_name, clf)
            start, end = opt_range
            with imageio.get_writer(gif_path, mode='I', duration=duration) as writer:
                for i in range(start, end+1):
                    frames_path = plot_dir_path + '/{}_{}_{}.png'.format(opt_name, clf, j)
                    writer.append_data(imageio.imread(frames_path))

def main(args):
    # Create gifs
    create_clf_gifs(opt_dict=json_to_dict(args.opt_json),
                    duration = 0.5, plot_dir_path = '{}/clf_plots'.format(args.test_path),
                    gif_dir_path='{}/clf_gifs'.format(args.test_path))
    # create_state_prob_gifs(opt_names=args.opt_names,
    #                        num_frames=args.num_frames,
    #                        duration = 0.5,
    #                        plot_dir_path = '{}/state_prob_plots'.format(args.test_path),
    #                        gif_dir_path = '{}/state_prob_gifs'.format(args.test_path))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # parser.add_argument(
    #     '--opt_names',
    #     nargs='*',
    #     type=str
    # )
    # parser.add_argument(
    #     '--num_frames',
    #     nargs='*',
    #     type=int
    # )
    parser.add_argument(
        '--test_path',
        type=str
    )
    parser.add_argument(
        '--opt_json',
        type=json.loads
    )

    main(parser.parse_args())
