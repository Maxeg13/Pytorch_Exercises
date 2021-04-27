
    
data_test=[]
for file_name in ['0', '1', '2', '3', '123']:
    # Извлечение ЭМГ
    
    emg = loadFile.load_data(channels_N,chans, fold+file_name)
    emg_size = emg.shape[0]
    emg_chunk_size = int(emg_size/shots_N)
    
    # Наделаем снимки из тестового файла
    one_class_data = []
    
    for iter in range(shots_N):
        for t in range(iter*emg_chunk_size, (iter+1)*emg_chunk_size):
            hist.step(emg[t,:]+[myRand() for x in range(channels_N)]) 

        one_class_data.append(hist.vals.reshape((hist.N*hist.N*hist.N)).copy())
        
    data_test.append(one_class_data)
data_test = torch.tensor(data_test, dtype = torch.float,requires_grad=False)
# ,requires_grad=False


# net = Net(hist.N*hist.N*hist.N)

print(net(data_test[0]))
print(net(data_test[1]))
print(net(data_test[2]))
print(net(data_test[3]))
print(net(data_test[4]))    

# print(net(data_learn[0]))
# print(net(data_learn[1]))
# print(net(data_learn[2]))
# print(net(data_learn[3]))
# print(net(data_learn[4])) 