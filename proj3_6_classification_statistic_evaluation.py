# -*- coding: utf-8 -*-
"""
Created on Sun Apr 18 01:12:15 2021

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
from proj3_2_ANN_classification_validation import ANN_validate
import torch
from toolbox_02450 import train_neural_net, draw_neural_net
from toolbox_02450 import *

y = y.squeeze()
# y = np.reshape(y,(244,1))
print(y)

X = X.squeeze()
X = X.astype(float)
N, M = X.shape

#normalizing matrix
X = X - np.ones((N,1)) * X.mean(axis=0)
X = X*(1/np.std(X,axis=0))
print(X.shape)
print(X)

attributeNames = attributeNames1.tolist()
classNames = classNames
C = len(classNames)

# K-fold crossvalidation
K = 5
CV = model_selection.KFold(K, shuffle=True)

# Initialize variables for baseline and logistic regression for classification
train_error_rate = np.empty((K,1))
test_error_rate = np.empty((K,1))
coefficient_norm = np.empty((K,1))
Error_train_nofeatures = np.empty((K,1))
Error_test_nofeatures = np.empty((K,1))
opt_lambdas = []

# Initialize variables for ANN classification
n_replicates = 2        # number of networks trained in each k-fold
max_iter = 10000         # stop criterion 2 (max epochs in training)
opt_val_errs = []
opt_hidden_units = []
errors = [] # make a list for storing generalizaition error in each loop
error_rate = np.empty(K)

yhat = []
y_true = []
rAB = []
rBC = []
rAC = []

for k, (train_index, test_index) in enumerate(CV.split(X,y)):
    
    print('\nCrossvalidation fold: {0}/{1}'.format(k+1,K))
    X_train = X[train_index]
    y_train = y[train_index]
    X_test = X[test_index]
    y_test = y[test_index]
    y_true.append(y_test.reshape(-1,1))
    
    internal_cross_validation = 5
    
    ##### baseline for classification #####
    if y_train.tolist().count(0) > y_train.tolist().count(1):  
        y_est = 0
    else:
        y_est = 1
    Error_train_nofeatures[k] = np.sum(y_est != y_train) / len(y_train)
    Error_test_nofeatures[k] = np.sum(y_est != y_test) / len(y_test)
    yhatA = np.ones((len(y_test),1)) * y_est
    
   
    ##### logistic regression for classification #####
    lambda_interval = np.logspace(-8, 2, 50)
    opt_val_err, opt_lambda_interval, train_err_vs_lambda, test_err_vs_lambda, mean_w_vs_lambda = LogReg_validate(X_train, y_train, lambda_interval, internal_cross_validation)
    opt_lambdas.append(opt_lambda_interval)

    mdl = LogisticRegression(penalty='l2', C=1/opt_lambda_interval)
    mdl.fit(X_train, y_train)

    y_train_est = mdl.predict(X_train).T
    y_test_est = mdl.predict(X_test).T
    
    train_error_rate[k] = np.sum(y_train_est != y_train) / len(y_train)
    test_error_rate[k] = np.sum(y_test_est != y_test) / len(y_test)
    
    yhatB = np.reshape(y_test_est,(-1,1))
    # w_est = mdl.coef_[0] 
    # # print(w_est)
    # coefficient_norm[k] = np.sqrt(np.sum(w_est**2))
    
    
    ##### ANN classification #####
    n_hidden_units = range(1, 10)
    # internal_cross_validation = 10
    y_train = np.reshape(y_train,(-1,1))
    y_test = np.reshape(y_test,(-1,1))
    opt_val_err, opt_hidden_unit = ANN_validate(X_train, y_train, n_hidden_units, internal_cross_validation)
    opt_val_errs.append(opt_val_err)
    opt_hidden_units.append(opt_hidden_unit)
    
    # Extract training and test set for current CV fold, convert to tensors
    X_train_tensor = torch.Tensor(X[train_index,:])
    y_train_tensor = torch.Tensor(np.reshape(y[train_index],(-1,1)))
    X_test_tensor = torch.Tensor(X[test_index,:])
    y_test_tensor = torch.Tensor(np.reshape(y[test_index],(-1,1)))
    
    # Define the model, see also Exercise 8.2.2-script for more information.
    model = lambda: torch.nn.Sequential(
                        torch.nn.Linear(M, opt_hidden_unit), #M features to H hiden units
                        torch.nn.Tanh(),   # 1st transfer function,
                        torch.nn.Linear(opt_hidden_unit, 1), # H hidden units to 1 output neuron
                        torch.nn.Sigmoid() # final tranfer function
                        )
    
    loss_fn = torch.nn.BCELoss()

    print('Training model of type:\n\n{}\n'.format(str(model())))
    # Train the net on training data
    net, final_loss, learning_curve = train_neural_net(model,
                                                       loss_fn,
                                                       X=X_train_tensor,
                                                       y=y_train_tensor,
                                                       n_replicates=n_replicates,
                                                       max_iter=max_iter)
    
    print('\n\tBest loss: {}\n'.format(final_loss))
    
    # Determine estimated class labels for test set
    y_sigmoid = net(X_test_tensor)
    y_test_est = (y_sigmoid>.5).type(dtype=torch.uint8)

    # Determine errors and errors
    y_test_ANN = y_test_tensor.type(dtype=torch.uint8)

    e = y_test_est != y_test_ANN
    error_rate[k] = (sum(e).type(torch.float)/len(y_test_ANN)).data.numpy()
    # errors.append(error_rate) # store error rate for current CV fold 
    
    yhatC = y_test_est.detach().numpy()
    
    yhatA = yhatA.astype(float)
    yhatB = yhatB.astype(float)
    yhatC = yhatC.astype(float)
    
    loss = 2
    yhat.append( np.concatenate([yhatA, yhatB, yhatC], axis=1) )
    rAB.append( np.mean( np.abs( yhatA-y_test ) ** loss - np.abs( yhatB-y_test) ** loss ) )
    rBC.append( np.mean( np.abs( yhatB-y_test ) ** loss - np.abs( yhatC-y_test) ** loss ) )
    rAC.append( np.mean( np.abs( yhatA-y_test ) ** loss - np.abs( yhatC-y_test) ** loss ) )

    k+=1
    
min_error = np.min(test_error_rate)


print('\n +++++++ baseline for classification output ++++++++')
print('test errors', Error_test_nofeatures)

print('\n +++++++ logistic regression for classification output ++++++++')
print('optimized intervel lambdas:', opt_lambdas)
print('test errors', test_error_rate)

print('\n +++++++ ANN classification output ++++++++')
print('Optimized hidden units: {}'.format(opt_hidden_units))
print('test errors: {}'.format(error_rate))

# setup II
alpha = 0.05
rho = 1/K
p_AB_setupII, CI_AB_setupII = correlated_ttest(rAB, rho, alpha=alpha)
p_BC_setupII, CI_BC_setupII = correlated_ttest(rBC, rho, alpha=alpha)
p_AC_setupII, CI_AC_setupII = correlated_ttest(rAC, rho, alpha=alpha)

print('\n +++++++ p value and confidence intervel ++++++++')
print('Baseline vs. logistic regression: {},\n {}'.format(p_AB_setupII, CI_AB_setupII))
print('logistic regression vs. ANN regression: {},\n {}'.format(p_BC_setupII, CI_BC_setupII))
print('Baseline vs ANN regression: {},{}'.format(p_AC_setupII, CI_AC_setupII))

# setup I
alpha = 0.05
y_true = np.concatenate(y_true)[:,0]
yhat = np.concatenate(yhat)

zA = np.abs(y_true - yhat[:,0] ) ** loss
zB = np.abs(y_true - yhat[:,1] ) ** loss
zC = np.abs(y_true - yhat[:,2] ) ** loss
zAB = zA - zB
zBC = zB - zC
zAC = zA - zC

CI_AB_setupI = st.t.interval(1 - alpha, len(zAB) - 1, loc=np.mean(zAB), scale=st.sem(zAB))  # Confidence interval
p_AB_setupI = st.t.cdf(-np.abs(np.mean(zAB)) / st.sem(zAB), df=len(zAB) - 1)  # p-value

CI_BC_setupI = st.t.interval(1 - alpha, len(zBC) - 1, loc=np.mean(zBC), scale=st.sem(zBC))  # Confidence interval
p_BC_setupI = st.t.cdf(-np.abs(np.mean(zBC)) / st.sem(zBC), df=len(zBC) - 1)  # p-value

CI_AC_setupI = st.t.interval(1 - alpha, len(zAC) - 1, loc=np.mean(zAC), scale=st.sem(zAC))  # Confidence interval
p_AC_setupI = st.t.cdf(-np.abs(np.mean(zAC)) / st.sem(zAC), df=len(zAC) - 1)  # p-value

print('\n +++++++ p value and confidence intervel for setup I ++++++++')
print('Baseline vs. linear regression: {},\n {}'.format(p_AB_setupI, CI_AB_setupI))
print('linear regression vs. ANN regression: {},\n {}'.format(p_BC_setupI, CI_BC_setupI))
print('Baseline vs ANN regression: {},{}'.format(p_AC_setupI, CI_AC_setupI))


