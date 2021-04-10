# -*- coding: utf-8 -*-
"""
Created on Fri Apr  9 15:34:14 2021

@author: changai
"""

from proj1_1_load_data import *
from matplotlib.pylab import (figure, semilogx, loglog, xlabel, ylabel, legend, 
                           title, subplot, show, grid)
import sklearn.linear_model as lm
from sklearn import model_selection
from toolbox_02450 import rlr_validate


Y1 = X1 - np.ones((N,1)) * X1.mean(axis=0)
#print(X1.mean(axis=0)) #obtain mean value along column, will get 10 numbers, array(1, M)
#print(Y1)

#normalizing matrix
X = Y1*(1/np.std(Y1.astype(float),axis=0))

# Add offset attribute
X = np.concatenate((np.ones((X.shape[0],1)),X),1)
print(X.shape)
attributeNames = [u'Offset']+attributeNames1
M = M+1

test_proportion = 0.2
X_train, X_test, y_train, y_test = model_selection.train_test_split(X,y1,test_size=test_proportion)


# Values of lambda
lambdas = np.power(10.,range(-5,9))

internal_cross_validation = 10    

opt_val_err, opt_lambda, mean_w_vs_lambda, train_err_vs_lambda, test_err_vs_lambda = rlr_validate(X_train.astype(float), y_train.astype(float), lambdas, internal_cross_validation)


figure(figsize=(12,8))
subplot(1,2,1)
print(mean_w_vs_lambda.shape)
semilogx(lambdas,mean_w_vs_lambda.T[:,1:],'.-') # Don't plot the bias term
xlabel('Regularization factor')
ylabel('Mean Coefficient Values')
grid()
# You can choose to display the legend, but it's omitted for a cleaner 
# plot, since there are many attributes
#legend(attributeNames[1:], loc='best')

subplot(1,2,2)
title('Optimal lambda: 1e{0}'.format(np.log10(opt_lambda)))
loglog(lambdas,train_err_vs_lambda.T,'b.-',lambdas,test_err_vs_lambda.T,'r.-')
xlabel('Regularization factor')
ylabel('Squared error (crossvalidation)')
legend(['Train error','Validation error'])
grid()
    


show()
# Display results
print('Linear regression without feature selection:')
print('- Training error: {0}'.format(Error_train.mean()))
print('- Test error:     {0}'.format(Error_test.mean()))
print('- R^2 train:     {0}'.format((Error_train_nofeatures.sum()-Error_train.sum())/Error_train_nofeatures.sum()))
print('- R^2 test:     {0}\n'.format((Error_test_nofeatures.sum()-Error_test.sum())/Error_test_nofeatures.sum()))
print('Regularized linear regression:')
print('- Training error: {0}'.format(Error_train_rlr.mean()))
print('- Test error:     {0}'.format(Error_test_rlr.mean()))
print('- R^2 train:     {0}'.format((Error_train_nofeatures.sum()-Error_train_rlr.sum())/Error_train_nofeatures.sum()))
print('- R^2 test:     {0}\n'.format((Error_test_nofeatures.sum()-Error_test_rlr.sum())/Error_test_nofeatures.sum()))

print('Weights in last fold:')
for m in range(M):
    print('{:>15} {:>15}'.format(attributeNames[m], np.round(w_rlr[m,-1],2)))

print('Ran Exercise 8.1.1')