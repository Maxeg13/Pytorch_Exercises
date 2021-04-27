import numpy as np
class Hist:
    def __init__(self, N, lim):
        self.N=N
        self.vals = np.zeros((N, N, N),dtype = float)
        self.cnt=0
        # N:
        self.grid = [(x-(N-2)/2)/N*2*lim for x in range(N-1)]
        self.grid+=[1000000]
    def step(self,x):
        done = False
        self.cnt+=1
        self.cnt%=4
        for i in range(self.N):
            for j in range(self.N):
                for k in range(self.N):                     
                    if(self.grid[i]>x[0] and self.grid[j]>x[1] and self.grid[k]>x[2]):
                        # if(not((i>1 and j>1 and k >1)and(i<4 and j<4 and k <4))):
                        if(not(done)):                            
                            self.vals[i,j,k]+=.07
                            done = True
                    if(self.cnt==0):    
                        self.vals[i,j,k]*=0.98
                        # 78

                    
 