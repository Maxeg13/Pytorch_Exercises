#runfile('C:/Users/chibi/.spyder-py3/backprop/test.py', wdir='C:/Users/chibi/.spyder-py3/backprop')

import cv2
import numpy as np
import torch
from PIL import Image
import torchvision.transforms as transforms
trans=transforms.ToTensor()
backTrans=transforms.ToPILImage()
import matplotlib.pyplot as plt

def max(a,b):
    if a>b:
        return a
    else:
        return b
    
def min(a,b):
    if a<b:
        return a
    else:
        return b
    
cap = cv2.VideoCapture(0)

def nothing(x):
    pass
# Creating a window for later use
cv2.namedWindow('source')
cv2.namedWindow('result')

# Starting with 100 to prevent error while masking
h,s,v = 100,100,100

size = 480, 640, 3
result_ac=np.zeros(size, dtype=np.uint8)
result=np.zeros([30,45,1],dtype=np.float)

def imshow(img,ax):  
    img = img.detach()      # unnormalize
    npimg = img.numpy()
    ax.imshow(npimg)

PATH = './form_net2.pth'
from NetClass import Net
#net = Net()
#net.load_state_dict(torch.load(PATH))
classes=["neg","rect"];
path="./data/figures"
N=10
nums=[str(1+x) for x in  range(N)];

dataLearn=[[],[]];
for num in nums:
    img=Image.open(path+"/"+"rect"+num+".png");
    img=trans(img)
    dataLearn[1].append((img[:1,:,:]-img[:1,:,:].mean())/img[:1,:,:].std()*.6)


_, frame = cap.read()

f, (ax1, ax2) = plt.subplots(1, 2)


gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

#frame_=torch.tensor(gray.transpose((2,0,1)))
frame__=torch.tensor(gray,dtype=torch.float)/255
frame__.detach()
imshow(frame__,ax1)
for i in range(200):
    y=np.random.randint(350)+60;
    x=np.random.randint(600);
    fr=frame__[y:(y+32),x:(x+32)]
    fr=fr.unsqueeze(0)
    dataLearn[0].append((fr-fr.mean())/fr.std()*0.6);



print(result.max())
#    gray = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
cv2.imshow('source',frame)   
#cv2.imshow('result',resized)  
k = cv2.waitKey(500) & 0xFF


cap.release()

cv2.destroyAllWindows()