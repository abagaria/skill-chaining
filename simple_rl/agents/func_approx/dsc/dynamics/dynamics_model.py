import torch
import torch.nn as nn


class DynamicsModel(nn.Module):
    def __init__(self, state_size, action_size, device, mean_x=None, mean_y=None, mean_z=None, std_x=None, std_y=None,
                 std_z=None):
        super(DynamicsModel, self).__init__()

        self.device = device

        if mean_x is not None:
            self.set_standardization_vars(mean_x, mean_y, mean_z, std_x, std_y, std_z)

        self.model = nn.Sequential(
            nn.Linear(state_size + action_size, 500),
            nn.LeakyReLU(),
            nn.Linear(500, 500),
            nn.LeakyReLU(),
            nn.Linear(500, state_size)
        )

    def forward(self, state, action):
        state = (state - self.mean_x) / self.std_x
        action = (action - self.mean_y) / self.std_y
        cat = torch.cat([state, action], dim=1)
        return self.model(cat)

    def predict_next_state(self, state, action):
        pred = self.forward(state, action)
        return (pred * self.std_z) + self.mean_z + state

    def set_standardization_vars(self, mean_x, mean_y, mean_z, std_x, std_y, std_z):
        self.mean_x = self._numpy_to_torch(mean_x)
        self.mean_y = self._numpy_to_torch(mean_y)
        self.mean_z = self._numpy_to_torch(mean_z)
        self.std_x = self._numpy_to_torch(std_x)
        self.std_y = self._numpy_to_torch(std_y)
        self.std_z = self._numpy_to_torch(std_z)

    def compare_state(self, state, action, state_p):
        pred = self.forward(state, action)
        return (pred * self.std_z) + self.mean_z + state, (state_p * self.std_z) + self.mean_z + state

    def _numpy_to_torch(self, arr):
        return torch.from_numpy(arr).to(self.device).float()

    def __getstate__(self):
        return {
            "model": self.model.state_dict(),
            "mean_x": self.mean_x.cpu().numpy(),
            "mean_y": self.mean_y.cpu().numpy(),
            "mean_z": self.mean_z.cpu().numpy(),
            "std_x": self.std_x.cpu().numpy(),
            "std_y": self.std_y.cpu().numpy(),
            "std_z": self.std_z.cpu().numpy(),
        }

    def __setstate__(self, state_dictionary):
        self.model.load_state_dict(state_dictionary["model"])
        self.model.to(self.device)
        mean_x = state_dictionary["mean_x"]
        mean_y = state_dictionary["mean_y"]
        mean_z = state_dictionary["mean_z"]
        std_x = state_dictionary["std_x"]
        std_y = state_dictionary["std_y"]
        std_z = state_dictionary["std_z"]
        self.set_standardization_vars(mean_x, mean_y, mean_z, std_x, std_y, std_z)