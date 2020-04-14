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

#device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#print(device)

def nothing(x):
    pass
# Creating a window for later use

cv2.namedWindow('result')
cv2.namedWindow('source')
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
#net=net.to(device)
#

while(1):
    _, frameCLR = cap.read()
#    ex=frame.copy()
    frameCLR=cv2.resize(frameCLR,(int(1.333*300),300), interpolation = cv2.INTER_LINEAR);
    frame = cv2.cvtColor(frameCLR, cv2.COLOR_BGR2GRAY)
    
#    frame=np.flip(frame,1)
    
    
    frame_=torch.tensor(frame.transpose((0,1)))
    frame_=frame_.unsqueeze(0);
    frame__=torch.tensor(frame_,dtype=torch.float)*(1./70*0.6)
#    frame__=frame_
    frame__.detach()
#    frame__=frame__.to(device)
    
#    frame_mean=frame__.mean();
#    frame__=frame__-frame_mean
#    i=range(0,frame__.shape[2]-200,15);
#    j=range(0,frame__.shape[1]-200,15);
    back_ind=0
    for i_,i in enumerate(range(70,frame__.shape[1]-100,13)):
        for j_,j in enumerate(range(0,frame__.shape[2]-60,13)):
            frame___=frame__[:1,i:(40+i),j:(40+j)].unsqueeze(0)
#            if i%30==0:
#            print(i)
            #
            result[i_,j_,0]=(net((frame___-frame___.mean())).squeeze(0)[1].item())
#            if(net.getInd()!=1):
#            print(net.getInd()[0,0].item());
#            ind=net.getInd()[0,2].item();
#            cj=ind%18;
#            ci=ind//18;
            ci=0;
            cj=0;
            ind=5
            if(result[i_,j_,0]>0.7):
#                cv2.rectangle(frameCLR,(j+cj-back_ind,i+ci-back_ind),(j+cj+40-back_ind,i+ci+40-back_ind),(100,100,255),thickness=2);  
                
#                print(cX,cY)
                
                res=1.9-np.array(frame___[0,0,ind:40-ind,ind:40-ind])
                res[res<0.6]=0
                M = cv2.moments(res)
                cX=0;
                cY=0;
                if(M["m00"]!=0):
                    cX = int(M["m10"] / M["m00"])
                    cY = int(M["m01"] / M["m00"])
                
                cv2.rectangle(frameCLR,(j+cX+ind,i+cY+ind),(j+cX+ind,i+cY+ind),(100,100,255),thickness=5)
                cv2.rectangle(res,(cX,cY),(cX,cY),(255,100,0),thickness=4)
                cv2.imshow('result',res) 
#            res[i_,j_,0]=1.
            resized=cv2.resize(result,(500,400), interpolation = cv2.INTER_NEAREST);

    print(result.max())
#    gray = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    cv2.imshow('source',frameCLR)   
    
    
     
    k = cv2.waitKey(500) & 0xFF
    if k == 27:
        break

cap.release()

cv2.destroyAllWindows()