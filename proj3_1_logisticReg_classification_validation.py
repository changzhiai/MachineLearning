# -*- coding: utf-8 -*-
"""
Created on Fri Apr 16 10:15:19 2021

@author: changai
"""

import numpy as np
from scipy.io import loadmat
from sklearn.linear_model import LogisticRegression
from sklearn import model_selection

def LogReg_validate(X,y,lambda_interval,cvf=10):
    
    print("========== inner loop start ================")
    
    M = X.shape[1]
    # w = np.empty((M,cvf,len(lambda_interval)))
    train_error_rate = np.empty((cvf,len(lambda_interval)))
    test_error_rate = np.empty((cvf,len(lambda_interval)))
    coefficient_norm = np.empty((cvf,len(lambda_interval)))

    # K-fold crossvalidation
    CV = model_selection.KFold(cvf, shuffle=True)
    
    f = 0
    for train_index, test_index in CV.split(X,y):
        print('\nInner crossvalidation fold: {0}/{1}'.format(f+1,cvf))
        X_train = X[train_index]
        y_train = y[train_index]
        X_test = X[test_index]
        y_test = y[test_index]
        
        for k in range(0, len(lambda_interval)):
            mdl = LogisticRegression(penalty='l2', C=1/lambda_interval[k] )
            
            mdl.fit(X_train, y_train)
        
            y_train_est = mdl.predict(X_train).T
            y_test_est = mdl.predict(X_test).T
            
            train_error_rate[f,k] = np.sum(y_train_est != y_train) / len(y_train)
            test_error_rate[f,k] = np.sum(y_test_est != y_test) / len(y_test)
        
            w_est = mdl.coef_[0] 
            coefficient_norm[f,k] = np.sqrt(np.sum(w_est**2))
   
        f=f+1
    
    opt_val_err = np.min(np.mean(train_error_rate,axis=0))
    opt_lambda_interval = lambda_interval[np.argmin(np.mean(train_error_rate,axis=0))]
    train_err_vs_lambda = np.mean(train_error_rate,axis=0)
    test_err_vs_lambda = np.mean(test_error_rate,axis=0)
    mean_w_vs_lambda = np.squeeze(np.mean(coefficient_norm,axis=0))
    
    # min_error = np.min(test_error_rate)
    # opt_lambda_idx = np.argmin(test_error_rate)
    # opt_lambda = lambda_interval[opt_lambda_idx]
            
    print("========== inner loop end ================")
    
    return opt_val_err, opt_lambda_interval, train_err_vs_lambda, test_err_vs_lambda, mean_w_vs_lambda