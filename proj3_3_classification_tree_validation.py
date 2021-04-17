# -*- coding: utf-8 -*-
"""
Created on Fri Apr 16 13:50:25 2021

@author: changai
"""

import numpy as np
from scipy.io import loadmat
from sklearn.linear_model import LogisticRegression
from sklearn import model_selection, tree

def tree_validate(X,y,tc,cvf=10):
    
    print("========== inner loop start ================")
    
    M = X.shape[1]
    # w = np.empty((M,cvf,len(lambda_interval)))
    Error_test = np.empty((cvf,len(tc)))
    Error_train = np.empty((cvf,len(tc)))

    # K-fold crossvalidation
    CV = model_selection.KFold(cvf, shuffle=True)
    
    f = 0
    for train_index, test_index in CV.split(X,y):
        print('\nInner crossvalidation fold: {0}/{1}'.format(f+1,cvf))
        X_train = X[train_index]
        y_train = y[train_index]
        X_test = X[test_index]
        y_test = y[test_index]
        
        for i, t in enumerate(tc):
            # Fit decision tree classifier, Gini split criterion, different pruning levels
            dtc = tree.DecisionTreeClassifier(criterion='gini', max_depth=t)
            dtc = dtc.fit(X_train,y_train.ravel())
            y_est_test = dtc.predict(X_test)
            y_est_train = dtc.predict(X_train)
            # Evaluate misclassification rate over train/test data (in this CV fold)
            misclass_rate_test = np.sum(y_est_test != y_test) / float(len(y_est_test))
            misclass_rate_train = np.sum(y_est_train != y_train) / float(len(y_est_train))
            Error_test[f,i], Error_train[f,i] = misclass_rate_test, misclass_rate_train
   
        f=f+1
    
    opt_val_err = np.min(np.mean(Error_train,axis=0))
    opt_tc = tc[np.argmin(np.mean(Error_train,axis=0))]
    train_err_vs_tc = np.mean(Error_train,axis=0)
    test_err_vs_tc = np.mean(Error_test,axis=0)
            
    print("========== inner loop end ================")
    
    return opt_val_err, opt_tc, train_err_vs_tc, test_err_vs_tc