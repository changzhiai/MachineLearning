# -*- coding: utf-8 -*-
"""
Created on Thu Feb 25 23:42:38 2021

@author: changai
"""

from proj1_1_load_data import *
import matplotlib.pyplot as plt
from scipy.linalg import svd

Y1 = X1 - np.ones((N,1)) * X1.mean(axis=0)
#print(X1.mean(axis=0)) #obtain mean value along column, will get 10 numbers, array(1, M)
#print(Y1)

# PCA by computing SVD of Y
U1,S1,Vh1 = svd(Y1.astype(np.float),full_matrices=False)
Vt1 = Vh1.T

pcs = [0,1,2,3]
legendStrs = ['PC'+str(e+1) for e in pcs]
c = ['r','g','b']
bw = .2
r = np.arange(1,M+1)
print(Vt1)
for i in pcs:    
    plt.bar(r+i*bw, Vt1[:,i], width=bw)
plt.xticks(r+bw, attributeNames1)
plt.xlabel('Attributes')
plt.ylabel('Component coefficients')
plt.legend(legendStrs)
plt.grid()
plt.title('Bejaia: PCA Component Coefficients')
plt.show()