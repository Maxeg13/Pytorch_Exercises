from matplotlib import pyplot as plt
from loadFile import load_data
import numpy as np
np.set_printoptions(formatter={'float': '{: 0.2f}'.format})
plt.close() 

class Hist:
    def __init__(self, N):
        self.N=N
        vals = np.zeros((N,N))

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



# test

test_file = '1'
emg_test = load_data(channels_N,chans,test_file)

plt.figure(2)
plt.plot(emg.dot(vecs[:,0]))
plt.plot(emg_test.dot(vecs[:,0])-plot_shift)

print(vecs.transpose().dot(vecs))