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
y=deque(x)
fig = plt.figure()
ax = plt.axes(xlim=(0, x_size),ylim = (-800,2000))
# plt.autoscale(ax)
line, = ax.plot([], [], lw=2)
 
l=[0,0,np.linspace(0, x_size, x_size) , deque(x) ]

def init():
    line.set_data([], [])
    return line,
def animate(i, l):
    
    
    # print("received message: %s" % data)
    shift=4*2
    for iter in range(100):
        data, addr = sock.recvfrom(100) # buffer size is 1024 bytes
        # print(data)
        for a in data:     
            print(type(data))
            if(l[0]==0 and a!=255):
                print('trouble')
            else:
                if(l[0]==(1+shift)):
                    b=np.int32(0)
                    b+=np.uint8(a) 
                    # 
                if(l[0] ==(2+shift)):
                    # 1
                    b+=256*np.uint8(a)
                    
                if(l[0]==(3+shift)):
                    # 1
                    b+=65536*np.uint8(a)
                if(l[0]==(4+shift)):
                    b+=np.uint8(a)*16777216
                    # b/=16777216.
                    l[3].append(int(b) )
                    l[3].popleft()
                                                      
                
                l[0]+=1
                l[0]%=17          
           
                
      
    line.set_data(l[2],l[3])                
    return line,
            
    

    
 
anim = FuncAnimation(fig, animate, init_func=init, fargs=(l,),
                               frames=200, interval=0, blit=True)


   



            
        
    
 