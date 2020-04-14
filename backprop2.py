# Training Data
from timeit import default_timer as timer  
import numpy as np
import pylab
import matplotlib.pyplot as plt
import torch
import pdb


hidden_s=5
x_data = np.array(range(-4,5),dtype=float)/20
y_data = np.array([7, 4, 2 , 1 ,0, 1, 2 , 4 , 7],dtype=float)/20

k=1
w1 = torch.randn(hidden_s,1)*k  # a random guess: random value
w1.requires_grad=True
w1_ = torch.randn(hidden_s,1)*k
w1_.requires_grad=True
w2 = torch.randn(1,hidden_s)*k  # a random guess: random value
w2.requires_grad=True
w2_=torch.randn(1,1)*k
w2_.requires_grad=True
#w21=torch.randn(1,1)*k
#w21.requires_grad=True

# our model forward pass
def forward(x):
#    z=w1.mm(torch.tensor([[1.,x]]).t()) 
    z=w1*x;
    z=z+w1_;
#    print(z)
    z=z.tanh()-0.5
#    y=w2.mm(z)
#    z_ap=torch.cat((z,torch.tensor([[1.]])),0)
    y=w2@z
    y=y+w2_;
    y=y.tanh()-0.5
    return y 



# Loss function
#@jit(target ="cuda")  
def loss(x, y):
    y_pred = forward(x)
    return (y_pred - y) * (y_pred - y)



dat=[0,1]
pylab.plot(dat)
pylab.ion()
pylab.show()   
speed_k=0.6
start=timer()
for epoch in range(160):
#    if epoch<80:
#        speed_k=0.7
#    else:
#        speed_k=3
        
    for x_val, y_val in zip(x_data, y_data):

        l = loss(x_val, y_val)
#        l.to(device)
        l.backward()
        w1.data = w1.data - speed_k * w1.grad.data
        w1.grad.data.zero_();
        w2.data = w2.data - speed_k* w2.grad.data
        w2.grad.data.zero_();
        w1_.data = w1_.data - speed_k * w1_.grad.data
        w1_.grad.data.zero_();
        w2_.data = w2_.data - speed_k * w2_.grad.data
        w2_.grad.data.zero_();
#        w.grad.item()
#    print("progress:", epoch,  "loss=", l)

    print("epoch=",epoch,", loss=",l)
    y=[]
    for x in x_data:
        y.append(forward(x));
    
    pylab.plt.clf()
    pylab.plt.plot(x_data,y_data,'or-',x_data,y,'g*--')
    pylab.plt.ylabel('some numbers')
    pylab.plt.pause(0.0005)
    pylab.plt.show()
print("wasted ",timer()-start)