#import torchvision
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.optim as optim
from PIL import Image



def imshow(img,ax):    
#    ax1.plot(x, y)
    img = img.detach()      # unnormalize
    npimg = img.numpy()
#    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    ax.imshow(npimg)
#    ax2.imshow(npimg)
    

from NetClass import Net
net=Net()
#net=nn.DataParallel(net)


optimizer = optim.SGD(net.parameters(), lr=0.3, momentum = 0.1)
targs_learn=torch.tensor([[1,0],[0,1]],dtype=torch.float)
#targs_learn.to(device)
#net(img[:1,:,:].unsqueeze(0))


scale=1
shift=1
N=11
#img=Image.open(path+"/"+"rect2"+".png");
#img=trans(img);

print(net(dataLearn[0][10].unsqueeze(0)))
print(net(dataLearn[1][8].unsqueeze(0)))

#learn_N1=len(dataLearn[1])


        
#        print(class_i, class_+num)
#        
#w1=torch.tensor(net.conv1.weight)
#w_img=backTrans(w1[0,:1]/scale+shift)
#w_img.save('data/figures/kernel1_init.png') 

#w1=torch.tensor(net.conv1.weight)
#w_img=backTrans(w1[1,:1]/scale+shift)
#w_img.save('data/figures/kernel2_init.png') 
    


for epoch in range(2500):
    optimizer.zero_grad();
    ind1=np.random.randint(2);
    ind2=np.random.randint(len(dataLearn[ind1]));
    output=net(dataLearn[ind1][ind2].unsqueeze(0));
    criterion = nn.MSELoss();
#        net.zero_grad()
#    loss = criterion(output, torch.tensor([ind1],dtype=torch.float))
#    if epoch%30==0:
#        imshow(w1[0,0],ax1)
#        imshow(w1[1,0],ax2)
    loss = criterion(output, targs_learn[ind1])    
    loss.backward()
    optimizer.step()

f, axs = plt.subplots(2, 4)

#PATH = './form_net2.pth'
#net = Net()
#net.load_state_dict(torch.load(PATH))



w2=net.conv1.weight
#imshow(dataLearn[0][230].squeeze(),ax1)
for i in range(8):
    imshow(w2[i].sum(0),axs[i//4,i%4])
#imshow(w2[5].sum(0),ax2)

print(net(dataLearn[0][10].unsqueeze(0)))
print(net(dataLearn[1][8].unsqueeze(0)))
#w_img=backTrans(w2[0,:1])
#w_img.save('data/figures/kernel1.png')   
#
#w2=net.conv1.weight
#w_img=backTrans(w2[1,:1])
#w_img.save('data/figures/kernel2.png') 



#print(w2.min())
#print(w2.max())
#
#o1=[net(dataTest[0][i].unsqueeze(0))[0] for i in range(test_N) ]
#print("\n\no1 =",o1)
#o2=[net(dataTest[1][i].unsqueeze(0))[0] for i in range(test_N) ]
#print("\n\no2 =",o2)
#
#
#PATH = './form_net2.pth'
#torch.save(net.state_dict(), PATH)
#dw=w2-w1 
#print(dw)
#img_=backTrans(img[:1,:,:])
#img1=trans(img_);
