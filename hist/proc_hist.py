from matplotlib import pyplot as plt
from loadFile import load_data
import numpy as np
from Histogram import Hist
# np.set_printoptions(formatter={'float': '{: 0.2f}'.format})

plt.close() 
                        
hist_N = 8  
layer_ind = 6
shots_N = 6
channels_N=4
plot_shift = 2000
chans = [2,3,4,5]
test_file = '23'



# Извлечение ЭМГ

emg = load_data(channels_N,chans, test_file)
emg/=50
emg_size = emg.shape[0]
emg_chunk_size = int(emg_size/shots_N)

plt.figure(0)
plt.plot(emg+np.array([[-x*10 for x in range(4)]]))

# Наделаем снимки

hist = Hist(N = hist_N, lim = 4)
shots=[]
for iter in range(shots_N):
    for t in range(iter*emg_chunk_size, (iter+1)*emg_chunk_size):
        hist.step(emg[t,:]) 
        # print(t)
    shots.append(hist.vals)
 
 


fig, axs = plt.subplots(nrows=hist_N, ncols=shots_N, figsize=(hist_N, hist_N), sharey=True)

for i in range(hist_N):
    for j in range(shots_N):
        axs[i,j].imshow(shots[j][:,:,i], cmap='gray', vmin=0, vmax=1)
plt.show()







# plt.figure(1)
# plt.plot(emg[:,0],emg[:,1], '*')

# M=emg.transpose().dot(emg)
# M = M/M[0,0]
# print(M)
# lambs, vecs = np.linalg.eigh(M)
# plt.plot(vecs[0,:],vecs[1,:], 'or',linewidth=10)