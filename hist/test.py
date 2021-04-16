import numpy as np
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
from collections import deque
plt.style.use('seaborn-pastel')
x = np.linspace(0, 4, 100) 
y=deque(x)
fig = plt.figure()
ax = plt.axes(xlim=(0, 4), ylim=(-2, 2))
line, = ax.plot([], [], lw=3)
 
def init():
    line.set_data([], [])
    return line,
def animate(i):
    
    y.append( np.sin(2 * np.pi * (0.01 * i)))
    y.popleft()
    line.set_data(x, y)
    print(i)
    return line,
    
 
anim = FuncAnimation(fig, animate, init_func=init,
                               frames=200, interval=5, blit=True)
 