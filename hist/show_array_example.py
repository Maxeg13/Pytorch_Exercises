# -*- coding: utf-8 -*-
"""
Created on Tue Apr 13 14:50:35 2021

@author: DrPepper
"""
from matplotlib import pyplot as plt
from PIL import Image
import numpy as np

w, h = 16, 16
data = np.zeros((h, w), dtype=np.float)
data[0:8, 0:8] = [0.6] # red patch in upper left
plt.imshow(data, cmap='gray', vmin=0, vmax=1)