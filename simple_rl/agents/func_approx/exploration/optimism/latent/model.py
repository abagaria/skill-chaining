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

    def __init__(self, feature_size=32, latent_size=4, device=torch.device("cuda")):
        super(MNISTConvPhiNetwork, self).__init__()
        self.feature_size = feature_size
        self.latent_size = latent_size
        self.device = device

        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout2d(0.25)
        self.dropout2 = nn.Dropout2d(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, latent_size)

        self.to(device)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        return x
        # output = torch.tanh(x)  # TODO: Test this change
        # return output
