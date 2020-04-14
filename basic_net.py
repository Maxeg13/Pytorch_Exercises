import torch
import torchvision
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms


import pylab
import matplotlib.pyplot as plt

#device = torch.device('cuda')

x_data = torch.tensor([[x/10.-4/10.] for x in range(9)],dtype=torch.float)
y_data = np.array([7., 4, 2 , 1 ,0, 1, 2 , 4 , 7],dtype=float)/40.
y_data=torch.tensor([[x-3.5/40.] for x in y_data] ,dtype=torch.float)

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
#        self.pool = nn.MaxPool2d(2, 2)# 14
#        self.conv1 = nn.Conv2d(3, 6,5,stride=1)#29
        self.fc1 = nn.Linear(1, 14)
        self.fc2 = nn.Linear(14, 1)

    def forward(self, x):
#        x = (F.relu(self.conv1(x)))
#        x = x.view(-1, 32*32*3)
        
        x = torch.sigmoid(self.fc1(x))-0.5
        x = torch.sigmoid(self.fc2(x))-0.5
        
        
        return x
    
net=Net() 
#device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#net.to(device)
#x_data.to(device)
#y_data.to(device)

   
optimizer = optim.SGD(net.parameters(), lr=20., momentum=0.8)
learning_rate=0.0
for epoch in range(700):  # loop over the dataset multiple times

    running_loss = 0.0
    output_lst=[]
    optimizer.zero_grad()
    output = net(x_data)
    for x in output:
        output_lst.append(x);    
#    output_lst.append(output)
#        print(len(output_lst))
    criterion = nn.MSELoss()
#        net.zero_grad()
    loss = criterion(output, y_data)
#    loss=(y_data-output).t()@(y_data-output)
    loss.backward()
    optimizer.step() 
   
        
        
        
        
        
    if ((epoch%10)==0):
        pylab.plt.clf()
        pylab.plt.plot(np.array(x_data),np.array(y_data),'or-',np.array(x_data),np.array(output_lst),'g*--')
        pylab.plt.ylabel('some numbers')
        pylab.plt.pause(0.0005)
        pylab.plt.show()
        print("epoch %d"%epoch)

        # print statistics
#        running_loss += loss.item()
#        if i % 2000 == 1999:    # print every 2000 mini-batches
#            print('[%d, %5d] loss: %.3f' %
#                  (epoch + 1, i + 1, running_loss / 2000))
#            running_loss = 0.0

print('Finished Training')