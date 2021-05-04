# -*- coding: utf-8 -*-
"""
Created on Tue May  4 22:04:33 2021

@author: changai
"""
import numpy as np

a6 = [525, 1250, 1325, 675, 1525, 0, 75, 1025, 925, 1375]
a7 = [600, 1325, 1400, 750, 1600, 75, 0, 1100, 1000, 1450]

a8 = [500, 226, 300, 350, 500, 1025, 1100, 0, 100, 350]
a9 = [400, 325, 400, 250, 600, 925, 1000, 100, 0, 450]
a10 = [850, 125, 51, 700, 150, 1375, 1450, 350, 450, 0]

x1 = np.linalg.norm(np.asarray(a6) - np.asarray(a8))
y1 = np.linalg.norm(np.asarray(a6) - np.asarray(a9))
z1 = np.linalg.norm(np.asarray(a6) - np.asarray(a10))
print((x1+y1+z1)/3.)


x2 = np.linalg.norm(np.asarray(a7) - np.asarray(a8))
y2 = np.linalg.norm(np.asarray(a7) - np.asarray(a9))
z2 = np.linalg.norm(np.asarray(a7) - np.asarray(a10))
print((x2+y2+z2)/3.)

print((x1+y1+z1+x2+y2+z2)/6.0)