import torch
from rms_script import load_data
#import torchvision
import numpy as np
import torch.nn as nn
import torch.optim as optim
# import torchvision.transforms as transforms
#import torch.nn.functional as F
#import rms_script
  
def learning(net, lr,epoches_N , data_learn, targs_learn):# 4000
    optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.2)
    for epoch in range(epoches_N):
        optimizer.zero_grad();
        ind1=np.random.randint(len(data_learn));# рандомный класс
        # ind2=np.random.randint(data_learn[ind1].shape[0]) # индекс рандомного сэмпла
        #этого класса
        learn_batch=data_learn[ind1]
        output=net(learn_batch);
        criterion = nn.MSELoss(); #тут может быть метод наименьших квадратов
    #        net.zero_grad()
    #    loss = criterion(output, torch.tensor([ind1],dtype=torch.float))
        loss = criterion(output, targs_learn[ind1])
        
        loss.backward()# нашел dw
        optimizer.step()
        
def learningLocal(net, cnet, lr,epoches_N , data_learn, targs_learn):# 4000
    optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.2)
    for epoch in range(epoches_N):
        optimizer.zero_grad();
        ind1=np.random.randint(len(data_learn));# рандомный класс
        # ind2=np.random.randint(data_learn[ind1].shape[0]) # индекс рандомного сэмпла
        #этого класса
        learn_batch=cnet(data_learn[ind1])
        output=net(learn_batch);
        criterion = nn.MSELoss(); #тут может быть метод наименьших квадратов
    #        net.zero_grad()
    #    loss = criterion(output, torch.tensor([ind1],dtype=torch.float))
        loss = criterion(output, targs_learn[ind1])
        
        loss.backward()# нашел dw
        optimizer.step()