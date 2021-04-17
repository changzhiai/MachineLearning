# -*- coding: utf-8 -*-
"""
Created on Fri Apr 16 00:47:48 2021

@author: changai
"""

import numpy as np
from scipy.io import loadmat
from sklearn.linear_model import LogisticRegression
from sklearn import model_selection, tree
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB


def NB_validate(X,y,alpha,cvf=10):
    
    print("========== inner loop start ================")
    
    N, M = X.shape
    # w = np.empty((M,cvf,len(lambda_interval)))
    fit_prior = True
    errors = np.empty((cvf,len(alpha)))

    # K-fold crossvalidation
    CV = model_selection.KFold(cvf, shuffle=True)
    
    f = 0
    for train_index, test_index in CV.split(X,y):
        print('\nInner crossvalidation fold: {0}/{1}'.format(f+1,cvf))
        X_train = X[train_index]
        y_train = y[train_index]
        X_test = X[test_index]
        y_test = y[test_index]
        
        for i, t in enumerate(alpha):
            nb_classifier = MultinomialNB(alpha=t,
                                      fit_prior=fit_prior)
            nb_classifier.fit(X_train, y_train)
            y_est_prob = nb_classifier.predict_proba(X_test)
            y_est = np.argmax(y_est_prob,1)
            # print(y_est.shape, y_test.shape)
            errors[f,i] = np.sum(y_est!=y_test,dtype=float)/y_test.shape[0]
        
        f=f+1
    
    # print(errors)
    opt_val_err = np.min(np.mean(errors,axis=0))
    opt_alpha = np.argmin(np.mean(errors,axis=0))+1
    print(opt_alpha)
    print(opt_val_err)
    test_err_vs_alpha = np.mean(errors,axis=0)
    print(test_err_vs_alpha)
            
    print("========== inner loop end ================")
    
    return opt_val_err, opt_alpha, test_err_vs_alpha