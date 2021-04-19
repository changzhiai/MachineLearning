# -*- coding: utf-8 -*-
"""
Created on Sun Apr 18 16:29:33 2021

@author: changai
"""

from proj1_1_load_data import *
from matplotlib.pylab import (figure, semilogx, loglog, xlabel, ylabel, legend, 
                           title, subplot, show, grid, xticks, yticks)
import sklearn.linear_model as lm
from sklearn import model_selection
from toolbox_02450 import rlr_validate

# Values of lambda
lambdas = np.power(10.,range(-5,9))


train_err_vs_lambda = np.empty((len(lambdas), len(range(0,9))))
test_err_vs_lambda = np.empty((len(lambdas),len(range(0,9))))
for i in range(0,9):
    
    y = X1[:,i].astype('float')
    y = y.squeeze()
    
    X = np.concatenate((X1[:,range(0,i)], X1[:,range(i+1, 9)]), 1).astype('float')
    # print(X)
    # X = X[:,range(0,i)].astype(float) ## select only metereologcal datas
    N,M = X.shape 
    
    #normalizing matrix
    X = X - np.ones((N,1)) * X.mean(axis=0)
    X = X*(1/np.std(X,axis=0))
    
    # Add offset attribute
    X = np.concatenate((np.ones((X.shape[0],1)),X),1).astype('float')
    print(X.shape)
    attributeNames = [u'Offset']+attributeNames1
    M = M+1

    K = 10
    CV = model_selection.KFold(K, shuffle=True)
    
    Error_train = np.empty((K,len(lambdas)))
    Error_test = np.empty((K,len(lambdas)))
    Error_train_rlr = np.empty((K,len(lambdas)))
    Error_test_rlr = np.empty((K,len(lambdas)))
    Error_train_nofeatures = np.empty((K,len(lambdas)))
    Error_test_nofeatures = np.empty((K,len(lambdas)))
    w_rlr = np.empty((M,K,len(lambdas)))
    mu = np.empty((K, M-1))
    sigma = np.empty((K, M-1))
    w_noreg = np.empty((M,K,len(lambdas)))
    
    for l in range(0,len(lambdas)):
        k=0
        for train_index, test_index in CV.split(X,y):
            X_train = X[train_index]
            y_train = y[train_index]
            X_test = X[test_index]
            y_test = y[test_index]
            
            # Standardize the training and set set based on training set moments
            mu = np.mean(X_train[:, 1:], 0)
            sigma = np.std(X_train[:, 1:], 0)
            # print('mu.shape', mu.shape)
            
            X_train[:, 1:] = (X_train[:, 1:] - mu) / sigma
            X_test[:, 1:] = (X_test[:, 1:] - mu) / sigma
            
            # precompute terms
            Xty = X_train.T @ y_train
            XtX = X_train.T @ X_train
            
            lambdaI = lambdas[l] * np.eye(M)
            lambdaI[0,0] = 0 # remove bias regularization
            w_rlr[:,k,l] = np.linalg.solve(XtX+lambdaI,Xty).squeeze()
            
            # Compute mean squared error with regularization with optimal lambda
            Error_train_rlr[k,l] = np.square(y_train-X_train @ w_rlr[:,k,l]).sum(axis=0)/y_train.shape[0]
            Error_test_rlr[k,l] = np.square(y_test-X_test @ w_rlr[:,k,l]).sum(axis=0)/y_test.shape[0]

            # Estimate weights for unregularized linear regression, on entire training set
            w_noreg[:,k,l] = np.linalg.solve(XtX,Xty).squeeze()
            # Compute mean squared error without regularization
            Error_train[k,l] = np.square(y_train-X_train @ w_noreg[:,k,l]).sum(axis=0)/y_train.shape[0]
            Error_test[k,l] = np.square(y_test-X_test @ w_noreg[:,k,l]).sum(axis=0)/y_test.shape[0]
            
            k+=1
    
    train_err_vs_lambda[:,i] = np.mean(Error_train_rlr,axis=0)
    test_err_vs_lambda[:,i] = np.mean(Error_test_rlr,axis=0)
    # mean_w_vs_lambda = np.squeeze(np.mean(w_rlr,axis=1))
    
    # train_err_noreg_vs_lambda = np.mean(Error_train,axis=0)
    # test_err_noreg_vs_lambda = np.mean(Error_test,axis=0)
    # mean_w_noreg_vs_lambda = np.squeeze(np.mean(w_noreg,axis=1))
    
    # figure(figsize=(12,8))
    # subplot(1,2,1)
    # print(mean_w_vs_lambda.shape)
    # semilogx(lambdas,mean_w_vs_lambda.T[:,1:],'.-') # Don't plot the bias term
    # xlabel('Regularization factor')
    # ylabel('Mean Coefficient Values')
    # grid()
    # # You can choose to display the legend, but it's omitted for a cleaner 
    # # plot, since there are many attributes
    # #legend(attributeNames[1:], loc='best')
attributeNames = attributeNames1[range(0,9)].tolist()

figure(figsize=(12,10))
M = 3
x=0
for m1 in range(M):
    for m2 in range(M):
        subplot(M, M, m1*M + m2 + 1)
        title('linear regression (y: {})'.format(attributeNames[x]))
        # print(train_err_vs_lambda)
        # print(test_err_vs_lambda)
        loglog(lambdas,train_err_vs_lambda[:,x].T,'b.-',lambdas,test_err_vs_lambda[:,x].T,'r.-')
        # xlabel('Regularization factor')
        # ylabel('Squared error (crossvalidation)')
        legend(['Train error','Validation error'])
        grid()
        if m1==M-1:
            xlabel('Regularization factor')
        else:
            xticks([])
        if m2==0:
            ylabel('Squared error (crossvalidation)')
        else:
            ylabel('')
        
        x+=1
    # subplot(1,2,2)
    # title('regularized linear regression')
    # print(train_err_vs_lambda)
    # print(test_err_vs_lambda)
    # loglog(lambdas,train_err_vs_lambda.T,'b.-',lambdas,test_err_vs_lambda.T,'r.-')
    # xlabel('Regularization factor')
    # ylabel('Squared error (crossvalidation)')
    # legend(['Train error','Validation error'])
    # grid()
    
    
    
    # figure(figsize=(12,8))
    # subplot(1,2,1)
    # print(mean_w_vs_lambda.shape)
    # semilogx(lambdas,mean_w_noreg_vs_lambda.T[:,1:],'.-') # Don't plot the bias term
    # xlabel('Regularization factor')
    # ylabel('Mean Coefficient Values')
    # grid()
    # # You can choose to display the legend, but it's omitted for a cleaner 
    # # plot, since there are many attributes
    # #legend(attributeNames[1:], loc='best')
    
    # subplot(1,2,2)
    # title('unregularized linear regression')
    # loglog(lambdas,train_err_noreg_vs_lambda.T,'b.-',lambdas,test_err_noreg_vs_lambda.T,'r.-')
    # xlabel('Regularization factor')
    # ylabel('Squared error')
    # legend(['Train error','Validation error'])
    # grid()
    
    # show()
    
    
    # Display results
    # print('Linear regression without feature selection:')
    # print('- Training error: {0}'.format(Error_train.mean()))
    # print('- Test error:     {0}'.format(Error_test.mean()))
    # print('- R^2 train:     {0}'.format((Error_train_nofeatures.sum()-Error_train.sum())/Error_train_nofeatures.sum()))
    # print('- R^2 test:     {0}\n'.format((Error_test_nofeatures.sum()-Error_test.sum())/Error_test_nofeatures.sum()))
    # print('Regularized linear regression:')
    # print('- Training error: {0}'.format(Error_train_rlr.mean()))
    # print('- Test error:     {0}'.format(Error_test_rlr.mean()))
    # print('- R^2 train:     {0}'.format((Error_train_nofeatures.sum()-Error_train_rlr.sum())/Error_train_nofeatures.sum()))
    # print('- R^2 test:     {0}\n'.format((Error_test_nofeatures.sum()-Error_test_rlr.sum())/Error_test_nofeatures.sum()))
    
    # print('Weights in last fold:')
    # for m in range(M):
    #     print('{:>15} {:>15}'.format(attributeNames[m], np.round(w_rlr[m,-1],2)))
    
    # print('Ran Exercise 8.1.1')