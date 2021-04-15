# -*- coding: utf-8 -*-
"""
Created on Thu Apr 15 20:02:20 2021

@author: DrPepper
"""
import numpy as np

a=np.array([1,3,2.,1.])

for i in range(1000):
    print(i)
    hist.step(a)
    print(net(  torch.tensor(hist.vals.reshape((hist.N*hist.N*hist.N)), dtype = torch.float) ))