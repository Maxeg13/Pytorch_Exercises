import matplotlib.pyplot as plt
import numpy as np

#____ВАЖНО_____
#^^^^^^^^^^^^^^^
# channels_N=4



#plt.close
#Data for plotting
def load_data(channels_N,chans, file_name):
    data = open(
        "C:/Users/DrPepper/Desktop/ech_monitor/data/"+file_name+".txt", "r").readlines()
    N=len(data)
    emg=np.zeros((N,channels_N),dtype=float)   

   
    
    for i in range(1,N):
        a=data[i].split(';');
        for j in range(0,channels_N):
            emg[i-1,j]=float(a[chans[j]+1])
    emg=emg-np.mean(emg,0)       

    return(emg)
 


#fig1.ylabel('RMS')
#fig1.xlabel('time')
#
#ax2.plot(rms)
#fig2.legend(['1st chan','2nd chan','3rd chan','4th chan'])
#fig2.ylabel('RMS')
#fig2.xlabel('time')
