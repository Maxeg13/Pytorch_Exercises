import math
import matplotlib.pyplot as plt
import numpy as np

#____ВАЖНО_____
#^^^^^^^^^^^^^^^
channels_N=4
stride=1
wind=200


#plt.close
# Data for plotting
def load_data(subject,file_name):
    data = open("./data/"+subject+"/"+file_name+".txt", "r").readlines()
    N=len(data)
    emg=np.zeros((N,channels_N),dtype=float)
    

    rms_N=int((N-wind)/stride)
    rms=np.zeros((rms_N,channels_N),dtype=float)
    
    for i in range(1,N):
        a=data[i].split(';');
        for j in range(1,channels_N+1):
            emg[i-1,j-1]=float(a[j])
    emg=emg-np.mean(emg,0)
    
    #for j in range(0,8):
    for i in range(rms_N):
        rms[i,:]=np.std(emg[(i*stride):((i)*stride+wind),:],0)
        
#    emg,rms=load_data(subject,"mid_3")    
        

    
    return(emg, rms)
 


#fig1.ylabel('RMS')
#fig1.xlabel('time')
#
#ax2.plot(rms)
#fig2.legend(['1st chan','2nd chan','3rd chan','4th chan'])
#fig2.ylabel('RMS')
#fig2.xlabel('time')
