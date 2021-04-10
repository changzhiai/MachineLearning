# -*- coding: utf-8 -*-
"""
Created on Fri Feb 26 10:31:59 2021

@author: changai
"""
from proj1_1_load_data import *

import numpy as np
import scipy.linalg as linalg
from similarity import similarity
import seaborn as sns; sns.set_theme()
import matplotlib.pyplot as plt

M = np.shape(X1)[1]
sim = np.zeros((1, M)).T
cos_sim = np.zeros((M, M)).T
extJacc = np.zeros((M, M)).T
correlation = np.zeros((M, M)).T

for i in range(M):
    x1 = X1[:, i]
    x2 = X2[:, i]
    sim[i] = x1/linalg.norm(x1) @ x2/linalg.norm(x2)  #cosine
    # sim[i] = (x1 @ x2) / (linalg.norm(x1)**2 + linalg.norm(x2)**2 - (x1 @ x2)) #extended Jaccard
    # sim[i] = similarity(x1.astype(np.float).T, x2.astype(np.float).T, 'cos')
    # sim[i] = similarity(x1.astype(np.float).T, x2.astype(np.float).T, 'ext')
    #sim[i] = similarity(x1.astype(np.float).T, x2.astype(np.float).T, 'cor')
    
print('Similarity results:\n {0}'.format(sim))
print('Ran similarity')

#cosine similarity
for i in range(M):
    for j in range(M):
        x1 = X[:, i]
        x2 = X[:, j]
        # print(x1)
        cos_sim[i][j] = x1/linalg.norm(x1) @ x2/linalg.norm(x2)  #cosine similarity
print(cos_sim)

fig, ax = plt.subplots(figsize=(11, 9))
ticklabels = ['Temp', ' RH', ' Ws', 'Rain ', 'FFMC', 'DMC', 'DC', 'ISI', 'BUI', 'FWI']
ax = sns.heatmap(cos_sim, annot=True, xticklabels=ticklabels, yticklabels=ticklabels, linewidths=.5, cmap="YlOrRd")
#sns.set(font_scale=2)
plt.title('Cosine similarity')
plt.xlabel('Both region') 
plt.ylabel('Both region')
plt.show()


#extJacc similarity
for i in range(M):
    for j in range(M):
        x1 = X[:, i]
        x2 = X[:, j]
        # print(x1)
        extJacc[i][j] = (x1 @ x2) / (linalg.norm(x1)**2 + linalg.norm(x2)**2 - (x1 @ x2))  #cosine similarity
print(extJacc)

fig, ax = plt.subplots(figsize=(11, 9))
ticklabels = ['Temp', ' RH', ' Ws', 'Rain ', 'FFMC', 'DMC', 'DC', 'ISI', 'BUI', 'FWI']
ax = sns.heatmap(extJacc, annot=True, xticklabels=ticklabels, yticklabels=ticklabels, linewidths=.5, cmap="YlOrRd")
#sns.set(font_scale=2)
plt.title('extJacc similarity')
plt.xlabel('Both region') 
plt.ylabel('Both region')
plt.show()


#correlation
for i in range(M):
    for j in range(M):
        x1 = np.array([X[:, i].astype(np.float)])
        x2 = np.array([X[:, j].astype(np.float)])
        # print(x1)
        cor = np.cov(x1, x2) / (np.std(x1) * np.std(x2))  #correlation
        correlation[i][j] = cor[0][1]

print(correlation)

fig, ax = plt.subplots(figsize=(11, 9))
ticklabels = ['Temp', ' RH', ' Ws', 'Rain ', 'FFMC', 'DMC', 'DC', 'ISI', 'BUI', 'FWI']
ax = sns.heatmap(correlation, annot=True, xticklabels=ticklabels, yticklabels=ticklabels, linewidths=.5, cmap="YlOrRd")
#sns.set(font_scale=2)
plt.xlabel('Both region') 
plt.ylabel('Both region')
plt.show()


# for i in range(M):
#     x1 = np.array(X1[:, i].astype(np.float))
#     x2 = np.array(X2[:, i].astype(np.float))
#     print(i)
#     print(x1)
#     sim[i] = similarity(x1, x2, 'cor')
    
# print('Similarity results:\n {0}'.format(sim))

# x1 = np.array([X1[:, 8].astype(np.float)])
# x2 = np.array([X2[:, 8].astype(np.float)])
# print(x1)
# sim = similarity(x1, x2, 'cor')
# cor = np.cov(x1, x2) / (np.std(x1) * np.std(x2))
# print(cor)
# print(cor[0][1])
# print(sim)
# print('{:9.5}'.format(sim[0,0]))