# -*- coding: utf-8 -*-
"""
Created on Wed Apr 14 15:56:13 2021

@author: changai
"""

from proj1_1_load_data import *
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from toolbox_02450 import rocplot, confmatplot

font_size = 15
plt.rcParams.update({'font.size': font_size})

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

# Create crossvalidation partition for evaluation
# using stratification and 95 pct. split between training and test 
K = 20
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.95, stratify=y)
# Try to change the test_size to e.g. 50 % and 99 % - how does that change the 
# effect of regularization? How does differetn runs of  test_size=.99 compare 
# to eachother?

# # Standardize the training and set set based on training set mean and std
# mu = np.mean(X_train, 0)
# sigma = np.std(X_train, 0)

# X_train = (X_train - mu) / sigma
# X_test = (X_test - mu) / sigma

# Fit regularized logistic regression model to training data to predict 
# the type of wine
lambda_interval = np.logspace(-8, 2, 50)
train_error_rate = np.zeros(len(lambda_interval))
test_error_rate = np.zeros(len(lambda_interval))
coefficient_norm = np.zeros(len(lambda_interval))
for k in range(0, len(lambda_interval)):
    mdl = LogisticRegression(penalty='l2', C=1/lambda_interval[k] )
    
    mdl.fit(X_train, y_train)

    y_train_est = mdl.predict(X_train).T
    y_test_est = mdl.predict(X_test).T
    
    train_error_rate[k] = np.sum(y_train_est != y_train) / len(y_train)
    test_error_rate[k] = np.sum(y_test_est != y_test) / len(y_test)

    w_est = mdl.coef_[0] 
    coefficient_norm[k] = np.sqrt(np.sum(w_est**2))

min_error = np.min(test_error_rate)
opt_lambda_idx = np.argmin(test_error_rate)
opt_lambda = lambda_interval[opt_lambda_idx]

plt.figure(figsize=(8,8))
#plt.plot(np.log10(lambda_interval), train_error_rate*100)
#plt.plot(np.log10(lambda_interval), test_error_rate*100)
#plt.plot(np.log10(opt_lambda), min_error*100, 'o')
plt.semilogx(lambda_interval, train_error_rate*100)
plt.semilogx(lambda_interval, test_error_rate*100)
plt.semilogx(opt_lambda, min_error*100, 'o')
plt.text(1e-8, 3, "Minimum test error: " + str(np.round(min_error*100,2)) + ' % at 1e' + str(np.round(np.log10(opt_lambda),2)))
plt.xlabel('Regularization strength, $\log_{10}(\lambda)$')
plt.ylabel('Error rate (%)')
plt.title('Classification error')
plt.legend(['Training error','Test error','Test minimum'],loc='upper right')
plt.ylim([0, 20])
plt.grid()
plt.show()    

plt.figure(figsize=(8,8))
plt.semilogx(lambda_interval, coefficient_norm,'k')
plt.ylabel('L2 Norm')
plt.xlabel('Regularization strength, $\log_{10}(\lambda)$')
plt.title('Parameter vector L2 norm')
plt.grid()
plt.show()    
