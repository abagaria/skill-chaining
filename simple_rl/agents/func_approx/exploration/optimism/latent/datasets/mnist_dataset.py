import numpy as np
import torch
from torchvision import datasets, transforms
from collections import defaultdict


class MNISTDataset(object):
    def __init__(self, mode, classes=tuple(range(10)), num_examples=np.inf):

        self.train_set = datasets.MNIST('../data', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ]))
        self.test_set = datasets.MNIST('../data', train=False, transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ]))
        self.mode = mode
        self.classes = classes
        self.num_examples = num_examples

        assert mode in ("train", "test"), mode

        if mode == "train":
            self.data = self.train_set
        else:
            self.data = self.test_set

        # label -> list representing action_buffer
        self.data_dictionary = self._populate_data_dictionary()

        # list of all the buffers. Each element of this list is a np array
        # representing the states in a particular action's buffer
        self.action_buffers = [self._get_examples(c) for c in classes]

    def __call__(self, *args, **kwargs):
        return self.action_buffers

    def _populate_data_dictionary(self):
        data_dictionary = defaultdict(list)
        for image, label in self.data:  # type: torch.tensor, int
            if label in self.classes and len(data_dictionary[label]) < self.num_examples:
                data_dictionary[label].append(image.data.numpy())
        return data_dictionary

    def _get_examples(self, digit_label):
        """
        Given a class label, return self.num_examples number of images belonging to that class label.

        Args:
            digit_label (int)

        Returns:
            digit_images (np.ndarray): `num_examples` number of images of digit `digit_label`
        """
        assert digit_label in self.classes, self.classes

        class_images = self.data_dictionary[digit_label]
        return np.array(class_images)


if __name__ == "__main__":
    dset = MNISTDataset("train", classes=(0,), num_examples=20)
