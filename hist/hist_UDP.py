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
        self.shifts=np.array([-17.20, -2.00,  1.70, -13.00])

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
# ax = plt.axes(xlim=(0, x_size),ylim = (-200,200))
# ax = plt.axes(xlim=(0, x_size),ylim = (-1,2))

# !fings
ax = plt.axes(xlim=(0, 5),ylim = (0,2))
# plt.autoscale(ax)
line, = ax.plot([], [], lw=1)
 
# hist = Hist(N = hist_N, lim = 85)
vs = Vars(xs = np.linspace(0, x_size, x_size), 
              deqs =[y,y.copy(),y.copy(),y.copy()], hist = hist,net=net )
# l=[np.linspace(0, x_size, x_size) , [y,y.copy(),y.copy(),y.copy()] ]

def init():
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
                            b+=np.uint8(a)*16777216-vs.shifts[chan_ind]
                            vs.vals[chan_ind]=b
                            
                            vs.deqs[chan_ind].append(int(b) )
                            vs.deqs[chan_ind].popleft()  
                   
                            
                   # to do smth
            if(j%16==15):
                
                if(iter%2==0):
                    vs.hist.step(vs.vals)
                    # vs.deqs[chan_ind].append(int(b) )
                    # vs.deqs[chan_ind].popleft()  
                    
                if(iter%20==0):
                    tens=torch.tensor(vs.hist.vals.reshape((vs.hist.N*vs.hist.N*vs.hist.N)).copy(), dtype = torch.float,requires_grad=False)
                    res = vs.net(tens).detach().numpy()
                    print(res)
                    # net(tens).detach().numpy()
                    # vs.deqs[0].append( vs.net(tens).detach().numpy()[1] )
                    # vs.deqs[0].popleft()  
          
    line.set_data([1,2,2,3,3,4],[res[0],res[0],
                                  res[1],res[1],
                                  res[2],res[2]])  
    # line.set_data(vs.xs,vs.deqs[2])              
    return line,
            
    

    
 
anim = FuncAnimation(fig, animate, init_func=init, fargs=(vs,),
                               frames=200, interval=0, blit=True)


   



            
        
    
 