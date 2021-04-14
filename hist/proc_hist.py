from matplotlib import pyplot as plt
from loadFile import load_data
import numpy as np
from PIL import Image
from Histogram import Hist
np.set_printoptions(formatter={'float': '{: 0.2f}'.format})


plt.close() 


                        
N=10  
hist=Hist(N=N,lim = 4)



x=[[0,0,0], [0,0,1]]
for i in range(100):
    for xx in x:
        hist.step(xx)   

fig = np.zeros((N, N, 3), dtype=np.uint8)

layer_ind = 6
        
for i in range(N):
    for j in range(N):
        fig[i,j]=hist.vals[i,j,layer_ind]
        
fig = Image.fromarray(fig, 'RGB')
fig.show()


!^^^^^^^^^^^^^^^
channels_N=4
plot_shift = 2000
chans = [2,3,4,5]
pca_file='23'

emg = load_data(channels_N,chans,pca_file)
N=emg.shape[0]

plt.figure(0)
plt.plot(emg+np.array([[-x*plot_shift for x in range(4)]]))


plt.figure(1)
plt.plot(emg[:,0],emg[:,1], '*')

M=emg.transpose().dot(emg)
M = M/M[0,0]
print(M)
lambs, vecs = np.linalg.eigh(M)
plt.plot(vecs[0,:],vecs[1,:], 'or',linewidth=10)