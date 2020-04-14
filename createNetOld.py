#import torchvision
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.optim as optim
from PIL import Image
import torchvision.transforms as transforms
import matplotlib.pyplot as plt


def imshow(img,ax):
    
#    ax1.plot(x, y)
    img = img.detach()      # unnormalize
    npimg = img.numpy()
#    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    ax.imshow(npimg)
#    ax2.imshow(npimg)
    

trans=transforms.ToTensor()
backTrans=transforms.ToPILImage()

#net_lin_s=1
#conv2_maps_n=3
#
#class Net(nn.Module):
#    def __init__(self):
#        super(Net, self).__init__()
#        
#        self.conv1 = nn.Conv2d(1, 2,21)
#        self.pool = nn.MaxPool2d(2)
#        self.conv2 = nn.Conv2d(2, conv2_maps_n, 2)
#        self.pool2=nn.MaxPool2d(5)
#        
#        self.fc1 = nn.Linear(conv2_maps_n*net_lin_s*net_lin_s , 2)
##        self.fc2 = nn.Linear(5, 2)
##        self.fc3 = nn.Linear(84, 10)
#        self.activation=torch.sigmoid
##        self.activation=F.relu
#        
#
#    def forward(self, x):
#        x = self.pool(self.activation(self.conv1(x)))#32 #12 6
#        x = self.pool2(self.activation(self.conv2(x)))#10 5
#        x = x.view(-1, conv2_maps_n*net_lin_s*net_lin_s)
#        x = nn.functional.softmax(self.fc1(x))
##        x= self.activation(self.fc1(x))
##        
##        x = F.relu(self.fc2(x))
##        x = self.fc3(x)
#        return x
##    def result(self,x):
        
from NetClass import Net
net=Net()
optimizer = optim.SGD(net.parameters(), lr=0.3, momentum=0.1)
targs_learn=torch.tensor([[1,0],[0,1]],dtype=torch.float)
#net(img[:1,:,:].unsqueeze(0))

classes=["circle","rect"];
path="./data/figures"
scale=1
shift=1
N=11
#img=Image.open(path+"/"+"rect2"+".png");
#img=trans(img);



nums=[str(1+x) for x in range(N)];

dataLearn=[[],[]]
for class_i,class_ in enumerate(classes):
    for num in nums:
        img=Image.open(path+"/"+class_+num+".png");
        img=trans(img)
        dataLearn[class_i].append((img[:1,:,:]-img[:1,:,:].mean())/img[:1,:,:].std()*.6)
learn_N=len(dataLearn[0])

       
dataTest=[[],[]]
for class_i,class_ in enumerate(classes):
    for num in nums:
        img=Image.open(path+"/"+class_+num+".png");
        img=trans(img)
        dataTest[class_i].append((img[:1,:,:]-img[:1,:,:].mean())/img[:1,:,:].std()*.6)        
test_N=len(dataTest[0])        
#        print(class_i, class_+num)
        
w1=torch.tensor(net.conv1.weight)
w_img=backTrans(w1[0,:1]/scale+shift)
w_img.save('data/figures/kernel1_init.png') 

w1=torch.tensor(net.conv1.weight)
w_img=backTrans(w1[1,:1]/scale+shift)
w_img.save('data/figures/kernel2_init.png') 
    



for epoch in range(2500):
    optimizer.zero_grad();
    ind1=np.random.randint(2);
    ind2=np.random.randint(learn_N);
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

f, (ax1, ax2) = plt.subplots(1, 2)

#PATH = './form_net2.pth'
#net = Net()
#net.load_state_dict(torch.load(PATH))



w2=net.conv1.weight
imshow(w2[0,0],ax1)
imshow(w2[1,0],ax2)

w_img=backTrans(w2[0,:1])
w_img.save('data/figures/kernel1.png')   

w2=net.conv1.weight
w_img=backTrans(w2[1,:1])
w_img.save('data/figures/kernel2.png') 



print(w2.min())
print(w2.max())

o1=[net(dataTest[0][i].unsqueeze(0))[0] for i in range(test_N) ]
print("\n\no1 =",o1)
o2=[net(dataTest[1][i].unsqueeze(0))[0] for i in range(test_N) ]
print("\n\no2 =",o2)


PATH = './form_net2.pth'
torch.save(net.state_dict(), PATH)
#dw=w2-w1 
#print(dw)
#img_=backTrans(img[:1,:,:])
#img1=trans(img_);
