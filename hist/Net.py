import torch.nn as nn
import torch
class Net(nn.Module):
    def __init__(self,inp_N):
        super(Net, self).__init__()
        hidden_layer_size=6
        self.fc1 = nn.Linear(inp_N, hidden_layer_size)
        self.fc2 = nn.Linear(hidden_layer_size, 3)

    def forward(self, x):        
        x = torch.sigmoid(self.fc1(x))
        x = self.fc2(x)
      
        return x