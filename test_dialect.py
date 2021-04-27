import torch
from rms_script import load_data
#import torchvision
import numpy as np
import torch.nn as nn
import torch.optim as optim
#import torchvision.transforms as transforms
import torch.nn.functional as F
#import rms_script
from learning import *

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
#        self.pool = nn.MaxPool2d(2, 2)# 14
#        self.conv1 = nn.Conv2d(3, 6,5,stride=1)#29
        hidden_layer_size=2
        self.fc1 = nn.Linear(1, hidden_layer_size)
        self.fc2 = nn.Linear(hidden_layer_size, 1)

    def forward(self, x):
#        x = (F.relu(self.conv1(x)))
#        x = x.view(-1, 32*32*3)
        
        x = torch.sigmoid(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        # x = F.softmax(x)
#        _,x = torch.max(x)
        
        return x
    
net1 = Net()
data_learn = [torch.tensor([[0]],dtype = torch.float), 
              torch.tensor([[1]],dtype = torch.float)]

targs_learn=torch.tensor([[1],[0]],dtype=torch.float)
targs_learn2=torch.tensor([[0],[1]],dtype=torch.float)

learning(net1, lr=1,epoches_N=2000 , 
         data_learn=data_learn, targs_learn=targs_learn)

print('{}\t{}'.format(net1(data_learn[0]),net1(data_learn[1])))

net2= Net()
learningLocal(net = net2, cnet=net1, lr=1,epoches_N=2000 , 
         data_learn=data_learn, targs_learn=targs_learn)


print('{}\t{}'.format(net2(data_learn[0]),net2(data_learn[1])))



