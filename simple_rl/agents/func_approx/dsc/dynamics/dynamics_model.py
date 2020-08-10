import torch
import torch.nn as nn

from torch.utils.data import DataLoader
from simple_rl.agents.func_approx.dsc.dynamics.rollouts import RandomRolloutCollector

class DynamicsModel(nn.Module):
    def __init__(self, state_size, action_size, mean_x, mean_y, mean_z, std_x, std_y, std_z):
        super(DynamicsModel, self).__init__()

        self.device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
        self.set_standardization_vars(mean_x, mean_y, mean_z, std_x, std_y, std_z)

        self.model = nn.Sequential(
            nn.Linear(state_size + action_size, 500),
            nn.LeakyReLU(),
            nn.Dropout(p=0.1),
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

if __name__== "__main__":
    # use gpu if available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
   
    collector = RandomRolloutCollector(10)
    train = collector.get_train_data()
    test_sameloc = collector.construct_data(epochs=1, random_position=False)
    test_diffloc = collector.construct_data(epochs=1, random_position=True)
    training_gen = DataLoader(train, batch_size=32, shuffle=True, drop_last=True)
    test_sameloc_gen = DataLoader(test_sameloc, batch_size=32, shuffle=False, drop_last=True)
    test_diffloc_gen = DataLoader(test_diffloc, batch_size=32, shuffle=False, drop_last=True)

    model = DynamicsModel(6, 2, *collector.get_standardization_vars())
    model.to(device)
    loss_function = nn.MSELoss().to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

    train_list = []
    test_sameloc_list = []
    test_diffloc_list = []

    for epoch in range(1, 200):
        total_loss = 0
        for states, actions, states_p in training_gen:
            states = states.to(device).float()
            actions = actions.to(device).float()
            states_p = states_p.to(device).float()
            
            optimizer.zero_grad()
            states_pred = model.forward(states, actions)
            loss = loss_function(states_p, states_pred)
            loss.backward()
            optimizer.step()
            
            total_loss += loss_function(*model.compare_state(states, actions, states_p))
        
        train_list.append(total_loss.cpu().detach().item())
        
#        with torch.no_grad():
#            total_loss2 = 0
#            for states, actions, states_p in test_sameloc_gen:
#                states = states.to(device).float()
#                actions = actions.to(device).float()
#                states_p = states_p.to(device).float()
#                
#                # states_pred = model.forward(states, actions)
#                # loss = loss_function(states_p, states_pred) 
#                total_loss2 += loss_function(*model.compare_state(states, actions, states_p))
#            
#            test_sameloc_list.append(total_loss2.cpu().detach().item())
#            
#            test_loss = 0
#            for states, actions, states_p in test_diffloc_gen:
#                states = states.to(device).float()
#                actions = actions.to(device).float()
#                states_p = states_p.to(device).float()
#                
#                # states_pred = model.forward(states, actions)
#                # loss = loss_function(states_p, states_pred) 
#                test_loss += loss_function(*model.compare_state(states, actions, states_p))
#            
#            test_diffloc_list.append(test_loss.cpu().detach().item())
        
        print("Epoch {} total loss {}".format(epoch, total_loss / 10))
#        print("Epoch {} test loss {}".format(epoch, total_loss2))
#        print("Epoch {} test diff point loss {}".format(epoch, test_loss))
#        print("")
    
    torch.save(model, "./model.pth")
    
    import matplotlib.pyplot as plt
    for i in range(len(train_list)):
        train_list[i] = train_list[i] / 10
    plt.figure()
    plt.plot(train_list)
    plt.plot(test_sameloc_list)
    plt.plot(test_diffloc_list)
    plt.show()
    
#    s, a, s_p = train[500]
#    s = torch.from_numpy(s).float()
#    s = s.unsqueeze(0)
#    a = torch.from_numpy(a).float()
#    a = a.unsqueeze(0)
#    s_pred = model(s.cuda(), a.cuda())
#    
    