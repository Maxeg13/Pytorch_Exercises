import numpy as np
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
from collections import deque
import socket
import numpy as np
plt.style.use('seaborn-pastel')

ptr=0
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
ax = plt.axes(xlim=(0, x_size),ylim = (-800,2000))
# plt.autoscale(ax)
line, = ax.plot([], [], lw=2)
 
l=[np.linspace(0, x_size, x_size) , [y,y.copy(),y.copy(),y.copy()] ]

def init():
    line.set_data([], [])
    return line,
def animate(i, l):
    
    
    # print("received message: %s" % data)
    shift=4*0
    for iter in range(100):
        data, addr = sock.recvfrom(100) # buffer size is 1024 bytes
        # print(data)
        for i,a in enumerate(data):    
           
            byte_ind=i%4
            chan_ind=int((i%16)/4)
            if(byte_ind==0):
                b=np.int32(0)
                b+=np.uint8(a) 
                # 
            if(byte_ind ==1):
                # 1
                b+=256*np.uint8(a)
                
            if(byte_ind==2):
                # 1
                b+=65536*np.uint8(a)
            if(byte_ind==3):
                b+=np.uint8(a)*16777216
                # b/=16777216.
                l[1][chan_ind].append(int(b) )
                l[1][chan_ind].popleft()      
                
      
    line.set_data(l[0],l[1][0])                
    return line,
            
    

    
 
anim = FuncAnimation(fig, animate, init_func=init, fargs=(l,),
                               frames=200, interval=0, blit=True)


   



            
        
    
 