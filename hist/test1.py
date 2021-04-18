import numpy as np
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
plt.style.use('seaborn-pastel')
 
l=[0]
fig = plt.figure()
ax = plt.axes(xlim=(0, 4), ylim=(-2, 2))
line, = ax.plot([], [], lw=3)
 
def init():
    line.set_data([], [])
    return line,
def animate(i, l):
    l[0]+=1
    print(l[0])
    # K+=1
    x = np.linspace(0, 4, 1000)
    y = np.sin(2 * np.pi * (x - 0.01 * i))
    line.set_data(x, y)
    return line,
 
K=10
T=20
anim = FuncAnimation(fig, animate, init_func=init, fargs=(l,),
                               frames=200, interval=20, blit=True)
 
 