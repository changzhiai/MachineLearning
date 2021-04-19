# -*- coding: utf-8 -*-
"""
Created on Wed Apr 14 15:56:13 2021

@author: changai
"""

from proj1_1_load_data import *
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from toolbox_02450 import rocplot, confmatplot
from proj3_1_logisticReg_classification_validation import LogReg_validate
from sklearn import model_selection
from matplotlib.pylab import (figure, semilogx, loglog, xlabel, ylabel, legend, 
                           title, subplot, show, grid)

font_size = 15
plt.rcParams.update({'font.size': font_size})

# y = X[:,9].astype('float')
y = y1.squeeze()
# y = np.reshape(y,(244,1))
print(y)

X = X1.squeeze()
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

# Create crossvalidation partition for evaluation
# using stratification and 95 pct. split between training and test 
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.95, stratify=y)

# K-fold crossvalidation
K = 10                   # only five folds to speed up this example
CV = model_selection.KFold(K, shuffle=True)

# Fit regularized logistic regression model to training data to predict 
# the type of wine
lambda_interval = np.logspace(-8, 2, 50)
# train_error_rate = np.zeros(len(lambda_interval))
# test_error_rate = np.zeros(len(lambda_interval))
# coefficient_norm = np.zeros(len(lambda_interval))
train_error_rate = np.empty((K,1))
test_error_rate = np.empty((K,1))
coefficient_norm = np.empty((K,1))
Error_train_nofeatures = np.empty((K,1))
Error_test_nofeatures = np.empty((K,1))

opt_lambdas = []
for k, (train_index, test_index) in enumerate(CV.split(X,y)): 
    print('\nCrossvalidation fold: {0}/{1}'.format(k+1,K))
    X_train = X[train_index]
    y_train = y[train_index]
    X_test = X[test_index]
    y_test = y[test_index]
    internal_cross_validation = 10
    
    opt_val_err, opt_lambda_interval, train_err_vs_lambda, test_err_vs_lambda, mean_w_vs_lambda = LogReg_validate(X_train, y_train, lambda_interval, internal_cross_validation)
    opt_lambdas.append(opt_lambda_interval)
    
    #baseline for classification
    if y_train.tolist().count(0) > y_train.tolist().count(1):  
        y_est = 0
    else:
        y_est = 1
    Error_train_nofeatures[k] = np.sum(y_est != y_train) / len(y_train)
    Error_test_nofeatures[k] = np.sum(y_est != y_test) / len(y_test)
    
    #logistic regression for classification
    mdl = LogisticRegression(penalty='l2', C=1/opt_lambda_interval)
    
    mdl.fit(X_train, y_train)

    y_train_est = mdl.predict(X_train).T
    y_test_est = mdl.predict(X_test).T
    
    train_error_rate[k] = np.sum(y_train_est != y_train) / len(y_train)
    test_error_rate[k] = np.sum(y_test_est != y_test) / len(y_test)

    w_est = mdl.coef_[0] 
    # print(w_est)
    coefficient_norm[k] = np.sqrt(np.sum(w_est**2))
    
    # Display the results for the last cross-validation fold
    if k == K-1:
        figure(k, figsize=(12,8))
        subplot(1,2,1)
        print(mean_w_vs_lambda.shape)
        semilogx(lambda_interval,mean_w_vs_lambda.T,'.-') # Don't plot the bias term
        xlabel('Regularization factor')
        ylabel('Mean Coefficient Values')
        grid()
        # You can choose to display the legend, but it's omitted for a cleaner 
        # plot, since there are many attributes
        #legend(attributeNames[1:], loc='best')
        
        subplot(1,2,2)
        title('Optimal lambda: 1e{0}'.format(np.log10(opt_lambda_interval)))
        loglog(lambda_interval,train_err_vs_lambda.T,'b.-',lambda_interval,test_err_vs_lambda.T,'r.-')
        xlabel('Regularization factor')
        ylabel('Squared error (crossvalidation)')
        legend(['Train error','Validation error'])
        grid()
    
    # To inspect the used indices, use these print statements
    #print('Cross validation fold {0}/{1}:'.format(k+1,K))
    #print('Train indices: {0}'.format(train_index))
    #print('Test indices: {0}\n'.format(test_index))

    k+=1
    
min_error = np.min(test_error_rate)

print('logistic regression:')
print('- Training error: {0}'.format(train_error_rate.mean()))
print('- Test error:     {0}'.format(test_error_rate.mean()))
print('- minimum error:     {0}'.format(min_error))

print('\n +++++++ baseline for classification output ++++++++')
print('test errors', Error_test_nofeatures)

print('\n +++++++ logistic regression for classification output ++++++++')
print('optimized intervel lambdas:', opt_lambdas)
print('test errors', test_error_rate)
# opt_lambda_idx = np.argmin(test_error_rate)
# opt_lambda = lambda_interval[opt_lambda_idx]

# plt.figure(figsize=(8,8))
# #plt.plot(np.log10(lambda_interval), train_error_rate*100)
# #plt.plot(np.log10(lambda_interval), test_error_rate*100)
# #plt.plot(np.log10(opt_lambda), min_error*100, 'o')
# plt.semilogx(range(K), train_error_rate*100)
# plt.semilogx(range(K), test_error_rate*100)
# # plt.semilogx(opt_lambda, min_error*100, 'o')
# # plt.text(1e-8, 3, "Minimum test error: " + str(np.round(min_error*100,2)) + ' % at 1e' + str(np.round(np.log10(opt_lambda),2)))
# plt.xlabel('Regularization strength, $\log_{10}(\lambda)$')
# plt.ylabel('Error rate (%)')
# plt.title('Classification error')
# plt.legend(['Training error','Test error','Test minimum'],loc='upper right')
# plt.ylim([0, 10])
# plt.grid()
# plt.show()    

# plt.figure(figsize=(8,8))
# plt.semilogx(range(K), coefficient_norm,'k')
# plt.ylabel('L2 Norm')
# plt.xlabel('Regularization strength, $\log_{10}(\lambda)$')
# plt.title('Parameter vector L2 norm')
# plt.grid()
# plt.show()    
# show()
# Display results

# print('Weights in last fold:')
# for m in range(M):
#     print('{:>15} {:>15}'.format(attributeNames[m], np.round(w_rlr[m,-1],2)))
