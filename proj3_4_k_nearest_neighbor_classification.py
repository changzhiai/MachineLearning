# -*- coding: utf-8 -*-
"""
Created on Fri Apr 16 00:40:21 2021

@author: changai
"""
from proj1_1_load_data import *
from matplotlib.pyplot import figure, plot, xlabel, ylabel, show
import numpy as np
from scipy.io import loadmat
from sklearn.neighbors import KNeighborsClassifier
from sklearn import model_selection

# y = X[:,9].astype('float')
y = y.squeeze()
# y = np.reshape(y,(244,1))
print(y)

X = X.squeeze()
# X = X[:,range(0,8)].astype(float)
X = X.astype(float)
N, M = X.shape
print(X.shape)
X = X - np.ones((N,1)) * X.mean(axis=0)

#normalizing matrix
X = X*(1/np.std(X,axis=0))
print(X)

# attributeNames = attributeNames1[range(0,8)].tolist()
attributeNames = attributeNames1.tolist()
classNames = classNames
C = len(classNames)

# Maximum number of neighbors
L=40

CV = model_selection.LeaveOneOut()
errors = np.zeros((N,L))
i=0
for train_index, test_index in CV.split(X, y):
    print('Crossvalidation fold: {0}/{1}'.format(i+1,N))    
    
    # extract training and test set for current CV fold
    X_train = X[train_index,:]
    y_train = y[train_index]
    X_test = X[test_index,:]
    y_test = y[test_index]

    # Fit classifier and classify the test points (consider 1 to 40 neighbors)
    for l in range(1,L+1):
        knclassifier = KNeighborsClassifier(n_neighbors=l);
        knclassifier.fit(X_train, y_train);
        y_est = knclassifier.predict(X_test);
        errors[i,l-1] = np.sum(y_est[0]!=y_test[0])

    i+=1
    
# Plot the classification error rate
figure()
plot(100*sum(errors,0)/N)
xlabel('Number of neighbors')
ylabel('Classification error rate (%)')
show()