#runfile('C:/Users/chibi/.spyder-py3/backprop/test.py', wdir='C:/Users/chibi/.spyder-py3/backprop')

import cv2
import numpy as np
import torch

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

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

def nothing(x):
    pass
# Creating a window for later use
cv2.namedWindow('source')
cv2.namedWindow('result')

# Starting with 100 to prevent error while masking
h,s,v = 100,100,100

# Creating track bar
cv2.createTrackbar('h', 'result',0,179,nothing)
cv2.createTrackbar('s', 'result',0,255,nothing)
cv2.createTrackbar('v', 'result',0,255,nothing)

size = 480, 640, 3
result_ac=np.zeros(size, dtype=np.uint8)
result=np.zeros([30,45,1],dtype=np.float)


PATH = './form_net2.pth'
from NetClass import Net
#net = Net()
net=net.to(device)
#

while(1):

    _, frame = cap.read()
    
    
    
#    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    frame_=torch.tensor(frame.transpose((2,0,1)))
    frame__=torch.tensor(frame_,dtype=torch.float)*(1./70*0.6)
#    frame__=frame_
    frame__.detach()
    frame__=frame__.to(device)
    
#    frame_mean=frame__.mean();
#    frame__=frame__-frame_mean
#    i=range(0,frame__.shape[2]-200,15);
#    j=range(0,frame__.shape[1]-200,15);
    back_ind=10
    for i_,i in enumerate(range(70,frame__.shape[1]-92,16)):
        for j_,j in enumerate(range(0,frame__.shape[2]-50,16)):
            frame___=frame__[:1,i:(32+i),j:(32+j)].unsqueeze(0)
#            if i%30==0:
#            print(i)
#            1
            result[i_,j_,0]=(net((frame___-frame___.mean())).squeeze(0)[1].item())
#            if(net.getInd()!=1):
#            print(net.getInd()[0,0].item());
            ind=net.getInd()[0,6].item();
            cj=ind%18;
            ci=ind//18;
            if(result[i_,j_,0]>0.92):
                cv2.rectangle(frame,(j+cj-back_ind,i+ci-back_ind),(j+cj+30-back_ind,i+ci+30-back_ind),(100,100,255),thickness=2);  
#            res[i_,j_,0]=1.
            resized=cv2.resize(result,(500,400), interpolation = cv2.INTER_NEAREST);

    print(result.max())
#    gray = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    cv2.imshow('source',frame)   
    cv2.imshow('result',resized)  
    k = cv2.waitKey(500) & 0xFF
    if k == 27:
        break

cap.release()

cv2.destroyAllWindows()