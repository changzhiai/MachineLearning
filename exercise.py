# -*- coding: utf-8 -*-
"""
Created on Tue May  4 22:04:33 2021

@author: changai
"""
import numpy as np

# ------------------- PCA ------------------
print('\n=============== PCA calculation ================')
sig = [14.4, 8.19, 7.83, 6.91, 6.01]
SumOfSquare = sum(j*j for j in sig)
print(SumOfSquare)
pca = np.zeros(len(sig))
for i, each in enumerate(sig):
    print(i)
    pca[i] = sig[i]**2/SumOfSquare
    print('The variance explained by the {} principal components:{}'.format(i, pca[i]))

print('\n')
print('The variance explained by the first four principal components:{}'.format(pca[0]+pca[1]+pca[2]+pca[3]))


# a6 = [525, 1250, 1325, 675, 1525, 0, 75, 1025, 925, 1375]
# a7 = [600, 1325, 1400, 750, 1600, 75, 0, 1100, 1000, 1450]

# a8 = [500, 226, 300, 350, 500, 1025, 1100, 0, 100, 350]
# a9 = [400, 325, 400, 250, 600, 925, 1000, 100, 0, 450]
# a10 = [850, 125, 51, 700, 150, 1375, 1450, 350, 450, 0]

# x1 = np.linalg.norm(np.asarray(a6) - np.asarray(a8))
# y1 = np.linalg.norm(np.asarray(a6) - np.asarray(a9))
# z1 = np.linalg.norm(np.asarray(a6) - np.asarray(a10))
# print(x1)


# x2 = np.linalg.norm(np.asarray(a7) - np.asarray(a8))
# y2 = np.linalg.norm(np.asarray(a7) - np.asarray(a9))
# z2 = np.linalg.norm(np.asarray(a7) - np.asarray(a10))
# print(x2)

# print((x1+y1+z1+x2+y2+z2)/6.0)

# -------------------AdaBoost classier------------------
print('\n===============start AdaBoost classier================')
from math import e
sigmat = 1./7  # errors
wt = 1./7
at = 1./2 * np.log((1-sigmat)/sigmat)

# for correct
wt1_t = wt * e**(-at)

# for incorrect
wt1_t_in = wt * e**(at)
wt1 = wt1_t_in/[(1-sigmat)* e**(-at) + sigmat* e**(at)]

print(wt1_t)
print(wt1_t_in)

print('update weights for misclassified observations:{}'.format(wt1))



# -------------------sigma function------------------
print('\n===============start sigma function================')
from math import e
print(e)
# x= 418.94-26.12*16
x = -0.93
y = 1/(1+e**(-x))
print(y)


# -------------------GMM function------------------
print('\n===============start GMM function================')

# w1=0.37
# sigma1 = np.sqrt(0.09)
# xx = 6.9
# mu1 = 6.12
# w1=0.29
# sigma1 = np.sqrt(0.13)
# xx = 6.9
# mu1 = 6.55
w1=0.333333333
mu1 = 6.93
sigma1 = np.sqrt(0.12)
xx = 6.9

p1 = w1 * 1/(np.sqrt(2*np.pi*sigma1**2))*e**(-1/(2*sigma1**2)*np.square(xx - mu1))
print(p1)

# print(0.3333333 * 1/((2*np.pi*0.5**2)**5)*e**(-1/(2*0.5**2)*np.square(2.11)))
# print(0.3333333 * 1/((2*np.pi*0.5**2)**5)*e**(-1/(2*0.5**2)*np.square(1.15)))
# print(0.3333333 * 1/((2*np.pi*0.5**2)**5)*e**(-1/(2*0.5**2)*np.square(1.09)))