import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch

W=640//3
H=370//3
size = (W,H);

def imshow(img,ax):    
#    ax1.plot(x, y)
    img = img.detach()      # unnormalize
    npimg = img.numpy()
#    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    ax.imshow(npimg)

def rotate_bound(image, angle):
    # grab the dimensions of the image and then determine the
    # center


    (h, w) = image.shape[:2]
    std_=58
    (cX, cY) = (w // 2+np.random.randint(std_)-std_//2.3, h // 2+np.random.randint(std_)-std_//2.3)
    # grab the rotation matrix (applying the negative of the
    # angle to rotate clockwise), then grab the sine and cosine
    # (i.e., the rotation components of the matrix)
    M = cv2.getRotationMatrix2D((cX, cY), -angle, 1.0)
#    cos = np.abs(M[0, 0])
#    sin = np.abs(M[0, 1])
    # compute the new bounding dimensions of the image
#    nW = int((h * sin) + (w * cos))
#    nH = int((h * cos) + (w * sin))
    # adjust the rotation matrix to take into account translation
#    M[0, 2] += (nW / 2) - cX
#    M[1, 2] += (nH / 2) - cY
    # perform the actual rotation and return the image
    return cv2.warpAffine(image, M, (w,h))
#import imutils

#image = cv2.imread("data//billiards//bil2.jpg", cv2.IMREAD_GRAYSCALE);


#resized=cv2.resize(image,nsize, interpolation = cv2.INTERSECT_PARTIAL);


#res=rotate_bound(resized,10)


#path=".//data//billiards"
path=".//data//lines"
N=8
nums=[str(1+x) for x in  range(N)];
dataLearn=[[],[]]

cap = cv2.VideoCapture(0)

_, frame = cap.read()
#frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
frame=cv2.resize(frame,(int(400),300), interpolation = cv2.INTER_LINEAR);
frame__=torch.tensor(frame,dtype=torch.float)*(1./70*0.6)
frame__.detach()

for i in range(80):
    y=np.random.randint(190)+60;
    x=np.random.randint(250);
    fr=frame__[y:(y+40),x:(x+40)]
    fr=fr.unsqueeze(0)
    dataLearn[0].append((fr-fr.mean())/fr.std()*0.6);


image = cv2.imread(path+"//neutral.jpg", cv2.IMREAD_GRAYSCALE);
image=cv2.resize(image,(int(400),300), interpolation = cv2.INTER_LINEAR);
for i in range(40):
    y=np.random.randint(190)+60;
    x=np.random.randint(250);
    fr=torch.tensor(image[y:(y+40),x:(x+40)],dtype=torch.float)
    fr=fr.unsqueeze(0)
    dataLearn[0].append((fr-fr.mean())/fr.std()*0.6);

for i in range(4):
    for num in nums:
        image = cv2.imread(path+"//rhomb"+num+".png", cv2.IMREAD_GRAYSCALE);
              
        k=np.random.randint(50)/100
#        nW=int(W/(2.2+k));
#        nH=int(H/(2.2+k));
        nW=int(W/(2.2+k));
        nH=int(W/(2.2+k));
        
        #print(nW,nH)
        nsize= (nW,nH)
        resized=cv2.resize(image,nsize, interpolation = cv2.INTERSECT_PARTIAL);
    
        res=rotate_bound(resized,np.random.randint(40)-20)
        res_=res[res.shape[0]//2-20:res.shape[0]//2+20,res.shape[1]//2-20:res.shape[1]//2+20]
        if(np.random.randint(2)):
            res_=np.flip(res_,1)
        dataLearn[0].append(torch.tensor((res_[:,:]-res_[:,:].mean())/res_[:,:].std()*.6,dtype=torch.float).unsqueeze(0))



for i in range(17):
    for num in nums:
        image = cv2.imread(path+"//cross"+num+".png", cv2.IMREAD_GRAYSCALE);
              
        k=np.random.randint(50)/100
#        nW=int(W/(2.2+k));
#        nH=int(H/(2.2+k));
        nW=int(W/(2.2+k));
        nH=int(W/(2.2+k));
        
        #print(nW,nH)
        nsize= (nW,nH)
        resized=cv2.resize(image,nsize, interpolation = cv2.INTERSECT_PARTIAL);
    
        res=rotate_bound(resized,np.random.randint(40)-20)
        res_=res[res.shape[0]//2-20:res.shape[0]//2+20,res.shape[1]//2-20:res.shape[1]//2+20]
        if(np.random.randint(2)):
            res_=np.flip(res_,1)
        dataLearn[1].append(torch.tensor((res_[:,:]-res_[:,:].mean())/res_[:,:].std()*.6,dtype=torch.float).unsqueeze(0))





f, axs = plt.subplots(8, 8)


for i in range(32):
    l1=np.random.randint(len(dataLearn[0]))

    imshow((dataLearn[0][l1].squeeze()),axs[i//8,i%8])
    
for i in range(32,64):
    l2=np.random.randint(len(dataLearn[1]))
    imshow((dataLearn[1][l2].squeeze()),axs[i//8,i%8])
