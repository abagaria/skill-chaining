import torch
import torch.nn as nn

from torch.utils.data import DataLoader

class DynamicsModel(nn.Module):
    def __init__(self, state_size, action_size, mean_x, mean_y, mean_z, std_x, std_y, std_z, device):
        super(DynamicsModel, self).__init__()

        self.device = device
        self.set_standardization_vars(mean_x, mean_y, mean_z, std_x, std_y, std_z)
        
        self.model = nn.Sequential(
            nn.Linear(state_size + action_size, 500),
            nn.LeakyReLU(),
            nn.Linear(500, 500),
            nn.LeakyReLU(),
            nn.Linear(500, state_size)
        )
    
    def _numpy_to_torch(self, arr):
        return torch.from_numpy(arr).to(self.device).float()
    
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