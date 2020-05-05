import imageio
import os
import sys
import argparse
from pathlib import Path
import json
from PIL import Image, ImageFont, ImageDraw

def json_to_dict(json_file):
    new_dict = {}
    for k,v in json_file.items():
        new_dict[k] = [int(item) for item in v.split(',')]
    return new_dict

def add_text_to_img(img_path, title, save_path, font_path):
    img = Image.open(img_path)
    draw = ImageDraw.Draw(img)
    width, height = img.size

    font = ImageFont.truetype(font_path, 15)
    text_w, text_h = draw.textsize(title, font)

    # Draw text bottom right of image
    draw.text((width - text_w, height - text_h), title, (0,0,0), font=font)

    img.save(save_path)

def create_clf_gifs(opt_dict, duration, plot_dir_path, gif_dir_path, skip_window=0):
    # Create directory
    Path(gif_dir_path).mkdir(exist_ok=True)

    # Generate gifs
    for opt_name, opt_range in opt_dict.items():
        gif_path = gif_dir_path + '/{}.gif'.format(opt_name)
        start, end = opt_range
        with imageio.get_writer(gif_path, mode='I', duration=duration) as writer:
            for i in range(start, end+1):
                if skip_window % i == 0:
                    frames_path = plot_dir_path + '/{}_{}.png'.format(opt_name, i)
                    save_path = plot_dir_path + '/text_{}_{}.png'.format(opt_name, i)
                    
                    # Add episode to image
                    add_text_to_img(frames_path, str(i), save_path, "plotting/Arial.ttf")
                    
                    writer.append_data(imageio.imread(save_path))
    
        print("Gif {} saved!".format(gif_path))

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
                    duration = 0.5,
                    plot_dir_path = '{}/clf_plots'.format(args.test_path),
                    gif_dir_path='{}/clf_gifs'.format(args.test_path),
                    skip_window=args.skip_window)
    
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
        type=json.loads,
        default={}
    )
    parser.add_argument(
        '--skip_window',
        type=int,
        default=0
    )
    # Example
    # python plotting/create_gifs.py --test_path '(debug) test/plots' --opt_json '{"option_1" : "2,4", "option_2" : "3,4", "overall_goal_policy_option" : "1,4"}'

    main(parser.parse_args())
