# -*- coding: utf-8 -*-
"""
Created on Fri Apr 16 00:40:21 2021

@author: changai
"""
from proj1_1_load_data import *
from matplotlib.pyplot import figure, plot, xlabel, ylabel, legend, show, boxplot
import numpy as np
from scipy.io import loadmat
from sklearn.neighbors import KNeighborsClassifier
from sklearn import model_selection
from proj3_4_k_nearest_neighbor_classification_validation import KNN_validate

# y = X[:,9].astype('float')
y = y.squeeze()
# y = np.reshape(y,(244,1))
print(y)

X = X.squeeze()
# X = X[:,range(0,8)].astype(float)
X = X.astype(float)
N, M = X.shape

#normalizing matrix
X = X - np.ones((N,1)) * X.mean(axis=0)
X = X*(1/np.std(X,axis=0))
print(X.shape)
print(X)

# attributeNames = attributeNames1[range(0,8)].tolist()
attributeNames = attributeNames1.tolist()
classNames = classNames
C = len(classNames)

# Maximum number of neighbors
L=10

CV = model_selection.LeaveOneOut()
errors = np.empty((N,1))

i=0
opt_Ls = []
for train_index, test_index in CV.split(X, y):
    print('Crossvalidation fold: {0}/{1}'.format(i+1,N))    
    
    # extract training and test set for current CV fold
    X_train = X[train_index,:]
    y_train = y[train_index]
    X_test = X[test_index,:]
    y_test = y[test_index]

    internal_cross_validation = 10
    opt_val_err, opt_L, test_err_vs_L = KNN_validate(X_train, y_train, L, internal_cross_validation)
    opt_Ls.append(opt_L)

    knclassifier = KNeighborsClassifier(n_neighbors=opt_L);
    knclassifier.fit(X_train, y_train);
    y_est = knclassifier.predict(X_test);
    errors[i] = np.sum(y_est[0]!=y_test[0])/N
    
    if i == N-1: 
        f = figure()
        boxplot(errors.T)
        xlabel('Model complexity (max tree depth)')
        ylabel('Test error across CV folds')
        
        f = figure()
        plot(range(N), errors)
        xlabel('Model complexity (max tree depth)')
        ylabel('Error (misclassification rate)')
        legend(['Error_train','Error_test'])
            
        show()
    i+=1 
min_error = np.min(test_err_vs_L)

# Display results
print('logistic regression:')
print('- Testing error: {0}'.format(round(test_err_vs_L.mean()),3))
print('- minimum error: {0}'.format(round(min_error),2))


print('\n +++++++ KNN output ++++++++')
print('optimized tree complixity parameters:', opt_Ls)
print('test errors', errors)

# # Plot the classification error rate
# figure()
# plot(100*sum(errors,0)/N)
# xlabel('Number of neighbors')
# ylabel('Classification error rate (%)')
# show()
