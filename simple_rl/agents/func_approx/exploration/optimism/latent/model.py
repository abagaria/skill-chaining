import torch
import torch.nn as nn
import torch.nn.functional as F


class DensePhiNetwork(nn.Module):
    def __init__(self, feature_size, h1=24, h2=12, latent_size=2, device=torch.device("cuda")):
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
        x = torch.tanh(self.fc3(x))

        return x

class ConvPhiNetwork(nn.Module):
    def __init__(self, feature_size, h1=6, h2=3, latent_size=2, device=torch.device("cuda")):
        """

        Args:
            feature_size (tuple): The size of the image? Assumes only 1 input channel
            h1 (int): num channels after first conv network
            h2 (int): num channels after second conv network
            latent_size (int): Size of output dimension
            device (str): Something like gpu or cpu
        """
        super(ConvPhiNetwork, self).__init__()
        assert isinstance(feature_size, tuple), feature_size
        assert len(feature_size) == 2, feature_size
        self.feature_size = feature_size
        self.latent_size = latent_size
        self.device = device

        self.conv1 = nn.Conv2d(1, h1, kernel_size=4, stride=2)
        self.bn1 = nn.BatchNorm2d(h1)
        self.conv2 = nn.Conv2d(h1, h2, kernel_size=4, stride=2)
        self.bn2 = nn.BatchNorm2d(h2)
        self.fc3 = nn.Linear(h2 * int(feature_size[0] / 4) * int(feature_size[1] / 4), 32)
        self.head = nn.Linear(32, latent_size)

    def forward(self, x):
        assert x.shape[1:] == (*self.feature_size, 1)
        x = x.float() / 255
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.fc3(x.view(x.size(0), -1)))
        return self.head(x)



