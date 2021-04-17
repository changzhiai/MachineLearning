# -*- coding: utf-8 -*-
"""
Created on Fri Apr 16 00:32:28 2021

@author: changai
"""

from proj1_1_load_data import *
from matplotlib.pyplot import figure, plot, xlabel, ylabel, legend, show, boxplot
from scipy.io import loadmat
from sklearn import model_selection, tree
import numpy as np
from proj3_3_classification_tree_validation import tree_validate

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

# Tree complexity parameter - constraint on maximum depth
tc = np.arange(2, 21, 1)

# K-fold crossvalidation
K = 10
CV = model_selection.KFold(n_splits=K,shuffle=True)

# Initialize variable
Error_train = np.empty((K,1))
Error_test = np.empty((K,1))

k=0
opt_tcs = []
for train_index, test_index in CV.split(X):
    print('Computing CV fold: {0}/{1}..'.format(k+1,K))
    # print(len(train_index), len(test_index))

    # extract training and test set for current CV fold
    X_train, y_train = X[train_index,:], y[train_index]
    X_test, y_test = X[test_index,:], y[test_index]
    
    internal_cross_validation = 10
    opt_val_err, opt_tc, train_err_vs_tc, test_err_vs_tc = tree_validate(X_train, y_train, tc, internal_cross_validation)
    opt_tcs.append(opt_tc)

    # Fit decision tree classifier, Gini split criterion, different pruning levels
    dtc = tree.DecisionTreeClassifier(criterion='gini', max_depth=opt_tc)
    dtc = dtc.fit(X_train,y_train.ravel())
    y_est_test = dtc.predict(X_test)
    y_est_train = dtc.predict(X_train)
    # Evaluate misclassification rate over train/test data (in this CV fold)
    misclass_rate_test = np.sum(y_est_test != y_test) / float(len(y_est_test))
    misclass_rate_train = np.sum(y_est_train != y_train) / float(len(y_est_train))
    Error_test[k], Error_train[k] = misclass_rate_test, misclass_rate_train

    if k == K-1: 
        f = figure()
        print(Error_test)
        boxplot(Error_test.T)
        xlabel('Model complexity (max tree depth)')
        ylabel('Test error across CV folds, K={0})'.format(K))
        
        f = figure()
        plot(tc, train_err_vs_tc)
        plot(tc, test_err_vs_tc)
        xlabel('Model complexity (max tree depth)')
        ylabel('Error (misclassification rate, CV K={0})'.format(K))
        legend(['Error_train','Error_test'])
            
        show()
    k+=1

min_error = np.min(Error_test)

# Display results
print('logistic regression:')
print('- Training error: {0}'.format(round(Error_train.mean()),3))
print('- Test error:     {0}'.format(Error_test.mean()))
print('- minimum error:     {0}'.format(round(min_error),2))


print('\n +++++++ classification tree output ++++++++')
print('optimized tree complixity parameters:', opt_tcs)
print('test errors', Error_test)

