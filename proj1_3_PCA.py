# -*- coding: utf-8 -*-
"""
Created on Thu Feb 25 21:51:44 2021

@author: changai
"""

from proj1_1_load_data import *
import matplotlib.pyplot as plt
from scipy.linalg import svd

Y1 = X1 - np.ones((N,1)) * X1.mean(axis=0)
#print(X1.mean(axis=0)) #obtain mean value along column, will get 10 numbers, array(1, M)
#print(Y1)

#normalizing matrix
Y1 = Y1*(1/np.std(Y1.astype(float),axis=0))

# PCA by computing SVD of Y
U1,S1,V1 = svd(Y1.astype(np.float),full_matrices=False)

# Compute variance explained by principal 
#print(S)
rho1 = (S1*S1) / (S1*S1).sum() 
#print(rho)

threshold = 0.9

# Plot variance explained
plt.figure()
plt.plot(range(1,len(rho1)+1),rho1,'x-')
plt.plot(range(1,len(rho1)+1),np.cumsum(rho1),'o-')
plt.plot([1,len(rho1)],[threshold, threshold],'k--')
plt.title('Variance explained by principal components for Bejaia');
plt.xlabel('Principal component');
plt.ylabel('Variance explained');
plt.legend(['Individual','Cumulative','Threshold'])
plt.grid()
plt.show()

print('run Bjaia')




Y2 = X2 - np.ones((N,1)) * X2.mean(axis=0)
#print(X1.mean(axis=0)) #obtain mean value along column, will get 10 numbers, array(1, M)
#print(Y2)

#normalizing matrix
Y2 = Y2*(1/np.std(Y2.astype(float),axis=0))

# PCA by computing SVD of Y
U2,S2,V2 = svd(Y2.astype(np.float),full_matrices=False)

# Compute variance explained by principal 
#print(S)
rho2 = (S2*S2) / (S2*S2).sum() 
#print(rho)

threshold = 0.9

# Plot variance explained
plt.figure()
plt.plot(range(1,len(rho2)+1),rho2,'x-')
plt.plot(range(1,len(rho2)+1),np.cumsum(rho2),'o-')
plt.plot([1,len(rho2)],[threshold, threshold],'k--')
plt.title('Variance explained by principal components for Sidi');
plt.xlabel('Principal component');
plt.ylabel('Variance explained');
plt.legend(['Individual','Cumulative','Threshold'])
plt.grid()
plt.show()

print('run Sidi')