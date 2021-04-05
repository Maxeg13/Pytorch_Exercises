import torch
from rms_script import load_data
#import torchvision
import numpy as np
import torch.nn as nn
import torch.optim as optim
#import torchvision.transforms as transforms
#import torch.nn.functional as F
#import rms_script

import pylab
import matplotlib.pyplot as plt

#device = torch.device('cuda')
channels_N=4
#x_data = torch.tensor([[x/10.-4/10.] for x in range(9)],dtype=torch.float)
#y_data = np.array([7., 4, 2 , 1 ,0, 1, 2 , 4 , 7],dtype=float)/40.
#y_data=torch.tensor([[x-3.5/40.] for x in y_data] ,dtype=torch.float)

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
#        self.pool = nn.MaxPool2d(2, 2)# 14
#        self.conv1 = nn.Conv2d(3, 6,5,stride=1)#29
        hidden_layer_size=12
        self.fc1 = nn.Linear(4, hidden_layer_size)
        self.fc2 = nn.Linear(hidden_layer_size, 2)

    def forward(self, x):
#        x = (F.relu(self.conv1(x)))
#        x = x.view(-1, 32*32*3)
        
        x = torch.sigmoid(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
#        x = F.softmax(x)
#        _,x = torch.max(x)
        
        return x
def learning(net, optimizer,epoches_N , data_learn, targs_learn):# 4000
    for epoch in range(epoches_N):
        optimizer.zero_grad();
        ind1=np.random.randint(2);# рандомный класс
        ind2=np.random.randint(data_learn[ind1].shape[0]) # индекс рандомного сэмпла
        #этого класса
        learn_batch=data_learn[ind1][ind2]
        learn_batch=torch.tensor(learn_batch,dtype=torch.float)
        output=net(learn_batch);
        criterion = nn.MSELoss(); #тут может быть метод наименьших квадратов
    #        net.zero_grad()
    #    loss = criterion(output, torch.tensor([ind1],dtype=torch.float))
        loss = criterion(output, targs_learn[ind1])
        
        loss.backward()# нашел dw
        optimizer.step()
    
    
net=Net() 

subject='fingers'
_,data_learn_mid=load_data(subject,'mid_3')
x,data_learn_prox=load_data(subject,'prox_1')

t=np.array([a for a in range(x.shape[0])],dtype=np.float)/1000.
t1=np.array([a for a in range(data_learn_prox.shape[0])],dtype=np.float)/1000.
shift=np.array([[a for a in range(4)]],dtype=np.float)*250.
plt.plot(t,x*0.4+shift,LineWidth=1)
plt.plot(t1, data_learn_prox+shift,LineWidth=3,LineStyle='--')
plt.xlabel('time, s')
plt.ylabel('EMG')
plt.title('signals, '+'Oksana')
#plt.holdon

#расширь тестовые данные
#^^^^^^^^^^^^^^^^^^^^^^^
_,data_test_mid=load_data(subject,'mid_3')
_,data_test_prox=load_data(subject,'prox_1')


#получи 2 картинки (второй график - классы на выходе)
#^^^^^^^^^^^^^^^^^^^^^^^
fig1=plt.figure(1);
fig2=plt.figure(2);
ax1 = fig1.gca();
#ax1.clear();
ax2 = fig2.gca();
ax1.plot(data_learn_mid)
ax1.set_ylim(0,200)
ax2.plot(data_learn_prox)
ax2.set_ylim(0,200)

fig1.legend(['1st chan','2nd chan','3rd chan','4th chan'])
#
#data_learn_mid=torch.ones((100,4),dtype=torch.float)*10
#data_learn_prox=torch.ones((100,4),dtype=torch.float)*40
#data_test_mid=torch.ones((100,4),dtype=torch.float)*11
#data_test_prox=torch.ones((100,4),dtype=torch.float)*41




data_learn=[data_learn_mid,data_learn_prox]
data_test=[data_test_mid,data_test_prox]
targs_learn=torch.tensor([[1,0],[0,1]],dtype=torch.float)

norm_val=1./70
for x,y in data_test,data_learn:
    x*=norm_val
    y*=norm_val
optimizer = optim.SGD(net.parameters(), lr=0.2, momentum=0.2)

learning(net, optimizer,epoches_N=4000 , data_learn=data_learn, targs_learn=targs_learn)

    
    
print("test time!")
#print(net(torch.tensor(data_test[0],dtype=torch.float)))
print('\n\n\n')
#print(net(torch.tensor(data_test[1],dtype=torch.float)))

o1= net(torch.tensor(data_test[0],dtype=torch.float))   
o2=net(torch.tensor(data_test[1],dtype=torch.float))
sh_beg1 = 1 * o1.shape[0]//3
sh_beg2 = 1 * o2.shape[0]//3
sh_end1 = 2 * o1.shape[0]//3  
sh_end2 = 2 * o2.shape[0]//3  
#print(torch.sum(o1[:,0]>o1[:,1],dtype=torch.float)/o1.shape[0])
#print(torch.sum(o2[:,0]<o2[:,1],dtype=torch.float)/o2.shape[0])
print('Accuracy, точность:')
print((torch.sum(o1[sh_beg1 : sh_end1,0]>o1[sh_beg1 : sh_end1,1],dtype=torch.float)+torch.sum(o2[sh_beg2 : sh_end2,0]<o2[sh_beg2 :sh_end2,1],dtype=torch.float))/((sh_end1-sh_beg1)+(sh_end2-sh_beg2)))
print((torch.sum(o1[:,0]>o1[:,1],dtype=torch.float)+torch.sum(o2[:,0]<o2[:,1],dtype=torch.float))/(o1.shape[0]+o2.shape[0]))
print('среднеарифм:')
print((torch.sum(o1[:,0]>o1[:,1],dtype=torch.float)/o1.shape[0]+torch.sum(o2[:,0]<o2[:,1],dtype=torch.float)/o2.shape[0])/2.)

    
    
    
    
    
    
