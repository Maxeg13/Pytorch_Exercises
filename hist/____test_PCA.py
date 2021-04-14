import numpy as np
import matplotlib.pyplot as plt
N=1400

# xs=np.zeros((20,2),dtype=float32)
xs = np.random.rand(N,2)-0.5
for x in xs:
    h=np.random.rand()-0.5
    x[0]+=2.5*h
    x[1]+=1.5*h
    print("{:4.3f}\t{:4.3f}".format(x[0], x[1]))

plt.plot(xs[:,0],xs[:,1], '*')


_, vecs = np.linalg.eig(xs.transpose().dot(xs))
plt.plot(vecs[0,1],vecs[1,1], 'or',linewidth=10)

# xs.dot(vecs[:,1])

