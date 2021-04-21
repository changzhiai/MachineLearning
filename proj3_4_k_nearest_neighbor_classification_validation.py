# -*- coding: utf-8 -*-
"""
Created on Sat Apr 17 00:42:38 2021

@author: changai
"""

import numpy as np
from scipy.io import loadmat
from sklearn.linear_model import LogisticRegression
from sklearn import model_selection, tree
from sklearn.neighbors import KNeighborsClassifier

def KNN_validate(X,y,L,cvf=10):
    
    print("========== inner loop start ================")
    
    N, M = X.shape
    # w = np.empty((M,cvf,len(lambda_interval)))

    errors = np.empty((cvf,L))

    # K-fold crossvalidation
    CV = model_selection.KFold(cvf, shuffle=True)
    
    f = 0
    for train_index, test_index in CV.split(X,y):
        print('\nInner crossvalidation fold: {0}/{1}'.format(f+1,cvf))
        X_train = X[train_index]
        y_train = y[train_index]
        X_test = X[test_index]
        y_test = y[test_index]
        
        for l in range(1,L+1):
            knclassifier = KNeighborsClassifier(n_neighbors=l);
            knclassifier.fit(X_train, y_train);
            y_est = knclassifier.predict(X_test);
            errors[f,l-1] = np.sum(y_est[0]!=y_test[0])/N
        
        f=f+1
    
    # print(errors)
    opt_val_err = np.min(np.mean(errors,axis=0))
    opt_L = np.argmin(np.mean(errors,axis=0))+1
    print(opt_L)
    print(opt_val_err)
    test_err_vs_L = np.mean(errors,axis=0)
    print(test_err_vs_L)
            
    print("========== inner loop end ================")
    
    return opt_val_err, opt_L, test_err_vs_L