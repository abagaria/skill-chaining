import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb


class DenseQNetwork(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, seed, fc1_units=256, fc2_units=128):
        """
        Set up the layers of the DQN
        Args:
            state_size (int): number of states in the state variable (can be continuous)
            action_size (int): number of actions in the discrete action domain
            seed (int): random seed
            fc1_units (int): size of the hidden layer
            fc2_units (int): size of the hidden layer
        """
        super(DenseQNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, fc1_units)
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        self.fc3 = nn.Linear(fc2_units, action_size)

        self.bn1 = nn.BatchNorm1d(fc1_units)
        self.bn2 = nn.BatchNorm1d(fc2_units)

    def forward(self, state):
        """
        DQN forward pass
        Args:
            state (torch.tensor): convert env.state into a tensor

        Returns:
            logits (torch.tensor): score for each possible action (1, num_actions)
        """
        x = F.relu(self.fc1(state))
        x = self.bn1(x)
        x = F.relu(self.fc2(x))
        x = self.bn2(x)
        return self.fc3(x)


class ConvQNetwork(nn.Module):
    def __init__(self, in_channels=4, n_actions=14, seed=0):
        """
        Initialize Deep Q Network
        Args:
            in_channels (int): number of input channels
            n_actions (int): number of outputs
            seed (int): random seed
        """
        super(ConvQNetwork, self).__init__()
        torch.manual_seed(seed)

        self.conv1 = nn.Conv2d(in_channels, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout2d(0.25)
        self.dropout2 = nn.Dropout2d(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, n_actions)

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

    def initialize_with_smaller_network(self, smaller_net, init_q_value=0.):
        """
        Given a DQN over K actions, create a DQN over K + 1 actions. This is needed when we augment the
        MDP with a new action in the form of a learned option.
        Args:
            smaller_net (QNetwork)
            init_q_value (float)
        """
        def copy_layer(bigger, smaller):
            for my_param, source_param in zip(bigger.parameters(), smaller.parameters()):
                my_param.data.copy_(source_param)

        copy_layer(self.conv1, smaller_net.conv1)
        copy_layer(self.conv2, smaller_net.conv2)
        copy_layer(self.conv3, smaller_net.conv3)
        copy_layer(self.fc4, smaller_net.fc4)
        copy_layer(self.bn1, smaller_net.bn1)
        copy_layer(self.bn2, smaller_net.bn2)
        copy_layer(self.bn3, smaller_net.bn3)

        smaller_num_labels = smaller_net.head.out_features
        self.head.weight[:smaller_num_labels, :].data.copy_(smaller_net.head.weight)
        self.head.bias[:smaller_num_labels].data.copy_(smaller_net.head.bias)

        new_action_idx = self.head.out_features - 1
        self.head.bias[new_action_idx].data.fill_(init_q_value)

