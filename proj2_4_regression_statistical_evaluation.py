# -*- coding: utf-8 -*-
"""
Created on Sat Apr 17 22:34:57 2021

@author: changai
"""

from proj1_1_load_data import *
from matplotlib.pylab import (figure, semilogx, loglog, xlabel, ylabel, legend, 
                           title, subplot, show, grid)
import numpy as np
import sklearn.linear_model as lm
from sklearn import model_selection
from toolbox_02450 import rlr_validate
import matplotlib.pyplot as plt
import torch
from toolbox_02450 import train_neural_net, draw_neural_net
from scipy import stats
from proj2_3_ANN_regression_validation import ANN_validate
from toolbox_02450 import *

y = X2[:,9].astype('float')
y = y.squeeze()

X = X2[:,range(0,9)].astype(float) ## select only metereologcal datas
N,M = X.shape
#normalizing matrix
X = X - np.ones((N,1)) * X.mean(axis=0)
X = X*(1/np.std(X,axis=0))

# Add offset attribute
X = np.concatenate((np.ones((X.shape[0],1)),X),1).astype('float')
print(X.shape)
attributeNames = [u'Offset']+attributeNames1
M = M+1

attributeNames = attributeNames1[range(0,9)].tolist()

## Crossvalidation
K = 10
CV = model_selection.KFold(K, shuffle=True)

# Initialize variables for linear regression
Error_train = np.empty((K,1))
Error_test = np.empty((K,1))
Error_train_rlr = np.empty((K,1))
Error_test_rlr = np.empty((K,1))
Error_train_nofeatures = np.empty((K,1))
Error_test_nofeatures = np.empty((K,1))
w_rlr = np.empty((M,K))
mu = np.empty((K, M-1))
sigma = np.empty((K, M-1))
w_noreg = np.empty((M,K))
opt_lambdas = []

# Initialize variables for baseline and ANN regression
n_hidden_units = range(1, 11)

n_replicates = 2        # number of networks trained in each k-fold
max_iter = 10000         # stop criterion 2 (max epochs in training)
opt_hidden_units = []
# errors = [] # make a list for storing generalizaition error in each loop
mse = np.empty(K)


k=0
yhat = []
y_true = []
rAB = []
rBC = []
rAC = []
for train_index, test_index in CV.split(X,y):
    print('\nCrossvalidation fold: {0}/{1}'.format(k+1,K))
    
    # extract training and test set for current CV fold
    X_train = X[train_index]
    y_train = y[train_index]
    X_test = X[test_index]
    y_test = y[test_index]
    y_true.append(y_test.reshape(-1,1))
    
    internal_cross_validation = 10
    
    mu[k, :] = np.mean(X_train[:, 1:], 0)
    sigma[k, :] = np.std(X_train[:, 1:], 0)
    X_train[:, 1:] = (X_train[:, 1:] - mu[k, :] ) / sigma[k, :] 
    X_test[:, 1:] = (X_test[:, 1:] - mu[k, :] ) / sigma[k, :] 
    
    ##### Baseline part #####
    Error_train_nofeatures[k] = np.square(y_train-y_train.mean()).sum(axis=0)/y_train.shape[0]
    Error_test_nofeatures[k] = np.square(y_test-y_test.mean()).sum(axis=0)/y_test.shape[0]
    yhatA = np.ones((len(y_test),1)) * y_test.mean()

    
    ##### linear regression part #####
    lambdas = np.power(10.,range(-5,9))
    opt_val_err1, opt_lambda, mean_w_vs_lambda, train_err_vs_lambda, test_err_vs_lambda = rlr_validate(X_train, y_train, lambdas, internal_cross_validation)
    opt_lambdas.append(opt_lambda)

    Xty = X_train.T @ y_train
    XtX = X_train.T @ X_train

    # Estimate weights for the optimal value of lambda, on entire training set
    lambdaI = opt_lambda * np.eye(M)
    lambdaI[0,0] = 0 # Do no regularize the bias term
    w_rlr[:,k] = np.linalg.solve(XtX+lambdaI,Xty).squeeze()
    # Compute mean squared error with regularization with optimal lambda
    Error_train_rlr[k] = np.square(y_train-X_train @ w_rlr[:,k]).sum(axis=0)/y_train.shape[0]
    Error_test_rlr[k] = np.square(y_test-X_test @ w_rlr[:,k]).sum(axis=0)/y_test.shape[0]
    
    yhatB = np.reshape(X_test @ w_rlr[:,k],(-1,1))
    # # Estimate weights for unregularized linear regression, on entire training set
    # w_noreg[:,k] = np.linalg.solve(XtX,Xty).squeeze()
    # # Compute mean squared error without regularization
    # Error_train[k] = np.square(y_train-X_train @ w_noreg[:,k]).sum(axis=0)/y_train.shape[0]
    # Error_test[k] = np.square(y_test-X_test @ w_noreg[:,k]).sum(axis=0)/y_test.shape[0]
    # k+=1
    
    
    ##### ANN regression part #####
    # n_hidden_units = range(1, 2)
    # internal_cross_validation = 10
    y_train = np.reshape(y_train,(-1,1))
    y_test = np.reshape(y_test,(-1,1))
    opt_val_err2, opt_hidden_unit = ANN_validate(X_train, y_train, n_hidden_units, internal_cross_validation)
    # opt_val_err2, opt_hidden_unit = ANN_validate(X_train, y_train, n_hidden_units, internal_cross_validation)
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
                        # torch.nn.Sigmoid() # final tranfer function
                        )
    loss_fn = torch.nn.MSELoss()

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
    y_test_est = net(X_test_tensor)
    
    # Determine errors and errors
    se = (y_test_est.float()-y_test_tensor.float())**2 # squared error
    mse[k] = (sum(se).type(torch.float)/len(y_test_tensor)).data.numpy() #mean
    # mse[k] = (sum(se).type(torch.float)/len(y_test)).data.numpy() #mean
    # errors.append(mse) # store error rate for current CV fold 
    
    yhatC = y_test_est.detach().numpy()
    
    loss = 2
    yhat.append( np.concatenate([yhatA, yhatB, yhatC], axis=1) )
    rAB.append( np.mean( np.abs( yhatA-y_test ) ** loss - np.abs( yhatB-y_test) ** loss ) )
    rBC.append( np.mean( np.abs( yhatB-y_test ) ** loss - np.abs( yhatC-y_test) ** loss ) )
    rAC.append( np.mean( np.abs( yhatA-y_test ) ** loss - np.abs( yhatC-y_test) ** loss ) )
    
    k+=1

    
print('\n +++++++ baseline output ++++++++')
print('test errors: {}'.format(Error_test_nofeatures.squeeze()))

print('\n +++++++ linear regression output ++++++++')
print('Optimized lambdas: {}'.format(opt_lambdas))
print('test errors: {}'.format(Error_test_rlr.squeeze()))

print('\n +++++++ ANN regression output ++++++++')
print('Optimized hidden units: {}'.format(opt_hidden_units))
print('test errors: {}'.format(mse))
# print('test errors: {}'.format(round(100*np.mean(errors),4)))

# setup II
alpha = 0.05
rho = 1/K
p_AB_setupII, CI_AB_setupII = correlated_ttest(rAB, rho, alpha=alpha)
p_BC_setupII, CI_BC_setupII = correlated_ttest(rBC, rho, alpha=alpha)
p_AC_setupII, CI_AC_setupII = correlated_ttest(rAC, rho, alpha=alpha)

print('\n +++++++ p value and confidence intervel for setup II ++++++++')
print('Baseline vs. linear regression: {},\n {}'.format(p_AB_setupII, CI_AB_setupII))
print('linear regression vs. ANN regression: {},\n {}'.format(p_BC_setupII, CI_BC_setupII))
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

