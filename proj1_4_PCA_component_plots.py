# -*- coding: utf-8 -*-
"""
Created on Thu Feb 25 23:25:32 2021

@author: changai
"""

from proj1_1_load_data import *
from matplotlib.pyplot import figure, plot, title, xlabel, ylabel, show, legend
from scipy.linalg import svd


Y1 = X1 - np.ones((N,1)) * X1.mean(axis=0)
#print(X1.mean(axis=0)) #obtain mean value along column, will get 10 numbers, array(1, M)
#print(Y1)

# PCA by computing SVD of Y
U1,S1,Vh1 = svd(Y1.astype(np.float),full_matrices=False)

Vt1 = Vh1.T    

# Project the centered data onto principal component space
Z1 = Y1 @ Vt1

# Indices of the principal components to be plotted
i = 0
j = 1

# Plot PCA of the data
f = figure()
title('Bejaia data: PCA')
#Z = array(Z)
for c in range(C):
    # select indices belonging to class c:
    class_mask = y1==c
    plot(Z1[class_mask,i], Z1[class_mask,j], 'o', alpha=.9)
legend(classNames)
xlabel('PC{0}'.format(i+1))
ylabel('PC{0}'.format(j+1))

# Output result to screen
show()
print('Ran Bejaia')




Y2 = X2 - np.ones((N,1)) * X2.mean(axis=0)
#print(X1.mean(axis=0)) #obtain mean value along column, will get 10 numbers, array(1, M)
#print(Y1)

# PCA by computing SVD of Y
U2,S2,Vh2 = svd(Y2.astype(np.float),full_matrices=False)

Vt2 = Vh2.T    

# Project the centered data onto principal component space
Z2 = Y2 @ Vt2

# Indices of the principal components to be plotted
i = 0
j = 1

# Plot PCA of the data
f = figure()
title('Sidi data: PCA')
#Z = array(Z)
for c in range(C):
    # select indices belonging to class c:
    class_mask = y1==c
    plot(Z2[class_mask,i], Z2[class_mask,j], 'o', alpha=.9)
legend(classNames)
xlabel('PC{0}'.format(i+1))
ylabel('PC{0}'.format(j+1))

# Output result to screen
show()
print('Ran Sidi')