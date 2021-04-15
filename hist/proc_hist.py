from matplotlib import pyplot as plt
import loadFile
import numpy as np
from learning import learning
import torch
from Net import Net
from Histogram import Hist
# np.set_printoptions(formatter={'float': '{: 0.2f}'.format})

plt.close() 
                        
hist_N = 8  
layer_ind = 6
shots_N = 10
channels_N = 4
plot_shift = 2000
chans = [2,3,4,5]
test_file = '123'


# Извлечение ЭМГ

emg = loadFile.load_data(channels_N,chans, '1504/'+test_file)
emg/=50
emg_size = emg.shape[0]
emg_chunk_size = int(emg_size/shots_N)

plt.figure(0)
plt.plot(emg+np.array([[-x*10 for x in range(4)]]))



# Наделаем снимки из тестового файла

hist = Hist(N = hist_N, lim = 4)
shots=[]
for iter in range(shots_N):
    for t in range(iter*emg_chunk_size, (iter+1)*emg_chunk_size):
        hist.step(emg[t,:]) 
    shots.append(hist.vals.copy())
    
fig, axs = plt.subplots(nrows=hist_N, ncols=shots_N, figsize=(hist_N, hist_N), sharey=True)

for i in range(hist_N):
    for j in range(shots_N):
        axs[i,j].imshow(shots[j][:,:,i], cmap='gray', vmin=0, vmax=1)
plt.show()
    



 
# Подготовим данные для обучения    
 
targs_learn = torch.tensor([[[0,0,0]],[[1,0,0]], 
                          [[0,1,0]],[[0,0,1]],[[1,1,1]]],dtype=torch.float32)


data_learn=[]
for file_name in ['0', '1', '2', '3', '123']:
    # Извлечение ЭМГ
    
    emg = loadFile.load_data(channels_N,chans, '1504/'+file_name)
    emg/=50
    emg_size = emg.shape[0]
    emg_chunk_size = int(emg_size/shots_N)
    
    # Наделаем снимки из тестового файла
    one_class_data = []
    
    hist = Hist(N = hist_N, lim = 4)
    for iter in range(shots_N):
        for t in range(iter*emg_chunk_size, (iter+1)*emg_chunk_size):
            hist.step(emg[t,:]) 
        one_class_data.append(hist.vals.reshape((hist.N*hist.N*hist.N)).copy())
        
    data_learn.append(one_class_data)
data_learn = torch.tensor(data_learn, dtype = torch.float)


# обучим Сетку

net = Net(hist.N*hist.N*hist.N)
learning(net=net, lr=.8,epoches_N=600 , 
         data_learn=data_learn, targs_learn=targs_learn)
print(net(data_learn[4]))
    
    
print("test time!")
#print(net(torch.tensor(data_test[0],dtype=torch.float)))
print('\n\n\n')
#print(net(torch.tensor(data_test[1],dtype=torch.float)))

o1= net(torch.tensor(data_learn[0],dtype=torch.float))   
o2=net(torch.tensor(data_learn[1],dtype=torch.float))

