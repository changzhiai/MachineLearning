# -*- coding: utf-8 -*-
"""
Created on Tue May  4 22:04:33 2021

@author: changai
"""
import numpy as np

# ------------------- PCA ------------------
print('\n=============== PCA calculation ================')
sig = [149, 118, 53, 42, 3]
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
sigmat = 5./7  # errors
wt = 1./7
at = 1./2 * np.log((1-sigmat)/sigmat)

# for correct
wt1_t = wt * e**(-at)

# for incorrect
wt1_t_in = wt * e**(at)
wt1 = wt1_t_in/[(1-sigmat)* e**(-at) + sigmat* e**(at)]

wt2 = wt1_t/[(1-sigmat)* e**(-at) + sigmat* e**(at)]

print(wt1_t)
print(wt1_t_in)

print('update weights for misclassified observations:{}'.format(wt1))
print('update weights for classified observations:{}'.format(wt2))


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
w1=0.48
mu1 = 3.184
sigma1 = 0.0075
xx = 3.19

p1 = w1 * 1/(np.sqrt(2*np.pi*sigma1**2))*e**(-1/(2*sigma1**2)*np.square(xx - mu1))
print(p1)
# print(0.3333333 * 1/((2*np.pi*0.5**2)**5)*e**(-1/(2*0.5**2)*np.square(2.11)))
# print(0.3333333 * 1/((2*np.pi*0.5**2)**5)*e**(-1/(2*0.5**2)*np.square(1.15)))
# print(0.3333333 * 1/((2*np.pi*0.5**2)**5)*e**(-1/(2*0.5**2)*np.square(1.09)))


# -------------------similarity------------------
print('\n===============start similarity================')

from similarity import similarity

x_vec = np.array([1, 0, 1, 0, 0, 1])
y_vec = np.array([1, 0, 1, 0, 1, 0])
# x_vec = np.array([1, 1, 0, 0])
# y_vec = np.array([1, 1, 0, 1])

print('Norm of x_vec-y_vec:',np.linalg.norm(x_vec-y_vec,1))
print('Norm of x_vec-y_vec:',np.linalg.norm(x_vec-y_vec,2))

# Similarity: 'SMC', 'Jaccard', 'ExtendedJaccard', 'Cosine', 'Correlation' 
print('Jaccard:',similarity(x_vec,y_vec,'Jaccard'))
print('SMC:',similarity(x_vec,y_vec,'SMC'))
print('Cos:',similarity(x_vec,y_vec,'cos'))
# print(similarity(x_vec,y_vec,'ext'))
# print(similarity(x_vec,y_vec,'cor'))
print('Ran Exercise 3.2.2')


# -------------------Kernel density estimator------------------
print('\n=============== Kernel Density Estimator================')
xxx = 3.918
mu11 = [-6.35 , -2.677, -3.003]
sigma11 = 2
sum = 0
for mu_i in mu11:
    sum += 1/(np.sqrt(2*np.pi*sigma11**2))*e**(-1/(2*sigma11**2)*np.square(xxx - mu_i))
p_xxx = sum/len(mu11)
print('sum:', sum)
print('p_xxx:', p_xxx)

#Spring2019(25)
#1/4*(-np.log(p_xxx1)-np.log(p_xxx2)-np.log(p_xxx3)-np.log(p_xxx4))
#1/4*(-np.log(0.0004565)-np.log(0.029)-np.log(0.078)-np.log(0.082))


##### In case 
# np.log2(x)

# import math
# number = 74088  # = 42**3
# base = 42
# exponent = math.log(number, base)  # = 3

x222 = np.array([39, 415, -7, -6727, 143])
y333 = np.array([0, -7, 1, 108, -2])
# np.std(x222)
# np.std(y333)
# np.mean(x222)
# np.mean(y333)
print(x222-x222.mean())*(y333-y333.mean())/((x222.std())*(y333.std()))

# array([ 1266.4,  1642.4,  1220.4, -5499.6,  1370.4])
# array([-20., -27., -19.,  88., -22.])