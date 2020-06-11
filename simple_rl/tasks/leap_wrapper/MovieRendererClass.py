import numpy as np
import imageio

class MovieRenderer(object):
    def __init__(self,
        clip_length,
        width, 
        height, 
        num_channels,
        framerate=240,
        num_clips=1,
        output_folder="",
        clip_name="movie",
        wait_between_clips=0):
        
        self.output_folder = output_folder
        self.clip_name = clip_name
        self.framerate = framerate
        self.num_clips = num_clips
        self.clip_index = 0
        self.wait_between_clips = wait_between_clips
        self.wait_index = None

        self.clip_dimensions = (
            clip_length,
            width,
            height,
            num_channels)

        self._reset_movie()

    def _add_frame(self, frame):
        if self.frame_index >= self.clip_dimensions[0]:
            print(f"Saving clip ({self.clip_index + 1}/{self.num_clips})")
            self._save_movie()
            self._reset_movie()
            self.clip_index += 1

            self.wait_index = 0
        
        self.movie[self.frame_index, :, :, :] = frame
        self.frame_index += 1

    def _should_wait(self):
        if self.wait_index is not None:
            if self.wait_index < self.wait_between_clips:
                self.wait_index += 1
                return True
            else:
                self.wait_index = None
                return False
        else:
            return False

    def add_frame(self, frame):
        assert(frame.shape == self.clip_dimensions[1:])
        
        if self.clip_index < self.num_clips and not self._should_wait():
            self._add_frame(frame)


    def _save_movie(self):
        imageio.mimwrite(
            f'{self.clip_name}_{self.clip_index + 1}.mp4', 
            self.movie, 
            fps=self.framerate)


    def _reset_movie(self):
        self.frame_index = 0
        self.movie = np.zeros(self.clip_dimensions, dtype=np.uint8)

    



