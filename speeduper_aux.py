# -*- coding: utf-8 -*-
"""
Created on Mon Feb  3 14:01:18 2020

@author: chibi
"""
        

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Assuming that we are on a CUDA machine, this should print a CUDA device:

print(device)


from numba import jit, cuda 
import numpy as np 
# to measure exec time 
from timeit import default_timer as timer    
  
# normal function to run on cpu 
def func(a):                                 
    for i in range(10000): 
        a@a.t()     
  
# function optimized to run on gpu  
#@jit(target ="cuda")                          
def func2(a):  
    a=a.to(device)
    for i in range(10000): 
        a@a.t()
if __name__=="__main__": 
    n = 10000000                            
    a = np.ones(n, dtype = np.float64) 
    b = np.ones(n, dtype = np.float32) 
    c=torch.tensor([[1,2,3]],dtype=torch.float)
    c=torch.tensor([[1,2,3]],dtype=torch.float)
    
    start = timer() 
    func(c) 
    print("without GPU:", timer()-start)     
      
    start = timer() 
    func2(c) 
    print("with GPU:", timer()-start) 
 
 