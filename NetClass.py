import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
net_lin_s=4
conv2_maps_n=2
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        
        self.conv1 = nn.Conv2d(1, 8, 15)
#        self.indPool= nn.MaxPool2d(1,return_indices=True)
        self.pool = nn.MaxPool2d(18,return_indices=True)
#        self.conv2 = nn.Conv2d(10, 10, 2)
#        self.pool2=torch.max
        
        self.fc1 = nn.Linear(8 , 4)
        self.fc2 = nn.Linear(4, 2)
#        self.fc3 = nn.Linear(84, 10)
#        self.activation=torch.sigmoid
        self.activation=F.relu
        

    def forward(self, x):
        self.y=1
        x,self.y= self.pool(self.activation(self.conv1(x)))#15 #1
#        x = (self.activation(self.conv2(x)))#
#        print(x.shape)
        x = x.view(-1, 8)
#        x = F.softmax(self.fc1(x))
        x= torch.sigmoid(self.fc1(x))
#        
        x = F.softmax(self.fc2(x))
#        x = self.fc3(x)
        return x
    
    def getInd(self):
        return self.y
    
    
    
class BillNet(nn.Module):
    def __init__(self):
        super(BillNet, self).__init__()
        
        self.conv1 = nn.Conv2d(1, 9, 5)
#        self.indPool= nn.MaxPool2d(1,return_indices=True)
        self.pool = nn.MaxPool2d(3)#36
        self.pool2 = nn.MaxPool2d(3)#36
        self.conv2 = nn.Conv2d(9, 4, 4)#81
#        self.pool2=torch.max
        
        self.fc1 = nn.Linear(36 , 7)
        self.fc2 = nn.Linear(7, 2)
#        self.fc3 = nn.Linear(84, 10)
#        self.activation=torch.sigmoid
        self.activation=F.relu
        

    def forward(self, x):
        self.y=1
        x= self.pool(self.activation(self.conv1(x)))#15 #1
        x = self.pool2(self.activation(self.conv2(x)))#
#        print(x.shape)
        x = x.view(-1, 36)
        x = self.activation(self.fc1(x))
#        x= torch.sigmoid(self.fc1(x))
#        
        x = F.softmax(self.fc2(x))
#        x = self.fc3(x)
        return x
    

class CrossNet(nn.Module):
    def __init__(self):
        super(CrossNet, self).__init__()
        
        self.conv1 = nn.Conv2d(1, 9, 6)
#        self.indPool= nn.MaxPool2d(1,return_indices=True)
        self.pool = nn.MaxPool2d(5)#36
        self.pool2 = nn.MaxPool2d(2)#36
        self.conv2 = nn.Conv2d(9, 3, 4)#81
#        self.pool2=torch.max
        
        self.fc1 = nn.Linear(12 , 7)
        self.fc2 = nn.Linear(7, 2)
#        self.fc3 = nn.Linear(84, 10)
#        self.activation=torch.sigmoid
        self.activation=F.relu
#        4!

    def forward(self, x):
        self.y=1
        x= self.pool(self.activation(self.conv1(x)))#15 #1
        x = self.pool2(self.activation(self.conv2(x)))#
#        print(x.shape)
        x = x.view(-1, 12)
        x = self.activation(self.fc1(x))
#        x= torch.sigmoid(self.fc1(x))
        x=self.fc2(x)
        x = F.softmax(x)
#        x = self.fc3(x)
        return x
