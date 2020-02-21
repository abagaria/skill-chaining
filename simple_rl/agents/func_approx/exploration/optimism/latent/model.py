import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb


class DensePhiNetwork(nn.Module):
    def __init__(self, feature_size, h1=32, h2=16, latent_size=2, device=torch.device("cuda")):
        super(DensePhiNetwork, self).__init__()
        self.feature_size = feature_size
        self.latent_size = latent_size
        self.device = device

        self.fc1 = nn.Linear(feature_size, h1)
        self.fc2 = nn.Linear(h1, h2)
        self.fc3 = nn.Linear(h2, latent_size)

        self.to(device)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x


class MNISTConvPhiNetwork(nn.Module):
    """ Taken from https://github.com/pytorch/examples/blob/master/mnist """

    def __init__(self, input_shape=(1, 28, 28), feature_size=32, latent_size=4, device=torch.device("cuda")):
        super(MNISTConvPhiNetwork, self).__init__()
        self.input_shape = input_shape
        num_features = input_shape[0]
        self.feature_size = feature_size
        self.latent_size = latent_size
        self.device = device

        self.conv1 = nn.Conv2d(num_features, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout2d(0.25)

        fc_input_shape = self._get_fc_input_size()

        self.dropout2 = nn.Dropout2d(0.5)
        self.fc1 = nn.Linear(fc_input_shape, 128) # 9216 for mnist
        self.fc2 = nn.Linear(128, latent_size)

        self.to(device)

    def _conv_part(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        return x

    def _dense_part(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        return x

    def _get_fc_input_size(self):
        """
        By passing an example image through the conv part of the network,
        we can get a direct example of what size a returned value will be,
        without having to do any math of our own.

        Returns:
            A number, which should be the input size of the first linear layer.
        """
        sample_image = torch.zeros(1,*self.input_shape) # the 1 is for batch.
        with torch.no_grad():
            image_after_conv = self._conv_part(sample_image)
            flattened_image = torch.flatten(image_after_conv)
        output_size = len(flattened_image)
        assert isinstance(output_size, int)
        return output_size


    def forward(self, x):
        x = self._conv_part(x)
        x = torch.flatten(x, 1)
        x = self._dense_part(x)
        return x
