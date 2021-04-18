import numpy as np
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
from collections import deque
import socket
import numpy as np
import torch
from Histogram import Hist
from Net import Net

plt.style.use('seaborn-pastel')

class Vars:
    def __init__(self,xs,deqs,hist,net):
        self.xs=xs
        self.vals=np.zeros(4)
        self.deqs=deqs
        self.hist=hist
        self.net=net

hist_N=6
frame_i=0

UDP_IP = "127.0.0.1"
UDP_PORT = 6432

sock = socket.socket(socket.AF_INET, # Internet
                     socket.SOCK_DGRAM) # UDP
sock.bind((UDP_IP, UDP_PORT))


x_size=1000
x = np.linspace(0, x_size, x_size) 
y=deque([0 for x in range(x_size)])
fig = plt.figure()
# ax = plt.axes(xlim=(0, x_size),ylim = (-2000,800))
ax = plt.axes(xlim=(0, x_size),ylim = (-1,2))
# plt.autoscale(ax)
line, = ax.plot([], [], lw=2)
 
hist = Hist(N = hist_N, lim = 85)
vs = Vars(xs = np.linspace(0, x_size, x_size), 
              deqs =[y,y.copy(),y.copy(),y.copy()], hist = hist,net=net )
# l=[np.linspace(0, x_size, x_size) , [y,y.copy(),y.copy(),y.copy()] ]

def init():
    bl=23
    line.set_data([], [])
    return line,
def animate(i, vs):
        
    # print("received message: %s" % data)
    # print(bl)
    for iter in range(90):
        data, addr = sock.recvfrom(128) # buffer size is 1024 bytes
        # print(data)
        
        for j,a in enumerate(data):    
           
            byte_ind=j%4
            chan_ind=int((j%16)/4)
            if(byte_ind==0):
                b=np.int32(0)
                b+=np.uint8(a) 
            else:                 
                if(byte_ind ==1):                
                    b+=256*np.uint8(a)   
                else:
                    if(byte_ind==2):
                        b+=65536*np.uint8(a)
                    else:
                        if(byte_ind==3):
                            b+=np.uint8(a)*16777216
                            vs.vals[chan_ind]=b
                            
                            # vs.deqs[chan_ind].append(int(b) )
                            # vs.deqs[chan_ind].popleft()  
                            
            if(j%16==15):
                hist.step([vs.vals[0], vs.vals[1] , vs.vals[3]])
                if(iter%8==0):
                    hist.decr()
                    # vs.deqs[chan_ind].append(int(b) )
                    # vs.deqs[chan_ind].popleft()  
                    
        if(iter%20==0):
            tens=torch.tensor(hist.vals.reshape((hist.N*hist.N*hist.N)).copy(), dtype = torch.float,requires_grad=False)
            print(net(tens).detach().numpy())
            # net(tens).detach().numpy()
            vs.deqs[0].append(net(tens).detach().numpy()[0] )
            vs.deqs[0].popleft()  
          
    line.set_data(vs.xs,vs.deqs[0])                
    return line,
            
    

    
 
anim = FuncAnimation(fig, animate, init_func=init, fargs=(vs,),
                               frames=200, interval=0, blit=True)


   



            
        
    
 