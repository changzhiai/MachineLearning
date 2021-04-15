# -*- coding: utf-8 -*-
"""
Created on Thu Apr 15 23:04:35 2021

@author: changai
"""

import numpy as np
from sklearn import model_selection
from toolbox_02450 import train_neural_net, draw_neural_net
import torch

def ANN_validate(X,y,units,cvf=10):
    
    print("========== inner loop start ================")
    # w = np.empty((M,cvf,len(units)))
    # train_error = np.empty((cvf,len(units)))
    # test_error = np.empty((cvf,len(units)))
    # e = np.empty((cvf,len(units)))
    error_rate = np.empty((cvf,len(units)))

    # y = y.squeeze()
    
    n_replicates = 2        # number of networks trained in each k-fold
    max_iter = 10000         # stop criterion 2 (max epochs in training)
    
    CV = model_selection.KFold(cvf, shuffle=True)
    M = X.shape[1]
    
    f = 0
    for train_index, test_index in CV.split(X,y):
        print('\nInner crossvalidation fold: {0}/{1}'.format(f+1,cvf))
        # X_train = X[train_index]
        # y_train = y[train_index]
        # X_test = X[test_index]
        # y_test = y[test_index]
        
        # mu = np.mean(X_train, 0)
        # sigma = np.std(X_train, 0)
        # X_train = (X_train - mu) / sigma
        # X_test = (X_test - mu) / sigma
        
        X_train = torch.Tensor(X[train_index,:])
        y_train = torch.Tensor(y[train_index])
        X_test = torch.Tensor(X[test_index,:])
        y_test = torch.Tensor(y[test_index])
        
        # Standardize the training and set set based on training set moments
       
        
        for n in range(0,len(units)):
            # # Compute parameters for current value of lambda and current CV fold
            # # note: "linalg.lstsq(a,b)" is substitue for Matlab's left division operator "\"
            # lambdaI = lambdas[l] * np.eye(M)
            # lambdaI[0,0] = 0 # remove bias regularization
            # w[:,f,l] = np.linalg.solve(XtX+lambdaI,Xty).squeeze()
            # # Evaluate training and test performance
            # train_error[f,l] = np.power(y_train-X_train @ w[:,f,l].T,2).mean(axis=0)
            # test_error[f,l] = np.power(y_test-X_test @ w[:,f,l].T,2).mean(axis=0)
            
            model = lambda: torch.nn.Sequential(
                        torch.nn.Linear(M, units[n]), #M features to H hiden units
                        torch.nn.Tanh(),   # 1st transfer function,
                        torch.nn.Linear(units[n], 1), # H hidden units to 1 output neuron
                        torch.nn.Sigmoid() # final tranfer function
                        )
            loss_fn = torch.nn.BCELoss()

            # Train the net on training data
            net, final_loss, learning_curve = train_neural_net(model,
                                                               loss_fn,
                                                               X=X_train,
                                                               y=y_train,
                                                               n_replicates=n_replicates,
                                                               max_iter=max_iter)
            
            print('\n\tBest loss: {}\n'.format(final_loss))
            
            # Determine estimated class labels for test set
            y_sigmoid = net(X_test)
            y_test_est = (y_sigmoid>.5).type(dtype=torch.uint8)
        
            # Determine errors and errors
            y_test = y_test.type(dtype=torch.uint8)
        
            e = y_test_est != y_test
            error_rate[f,n] = (sum(e).type(torch.float)/len(y_test)).data.numpy()
            # errors.append(error_rate) # store error rate for current CV fold 
            
            # Display the learning curve for the best net in the current fold
            # h, = summaries_axes[0].plot(learning_curve, color=color_list[k])
            # h.set_label('CV fold {0}'.format(k+1))
            # summaries_axes[0].set_xlabel('Iterations')
            # summaries_axes[0].set_xlim((0, max_iter))
            # summaries_axes[0].set_ylabel('Loss')
            # summaries_axes[0].set_title('Learning curves')
    
        f=f+1
    
    print(error_rate)
    opt_val_err = np.min(np.mean(error_rate,axis=0))
    opt_units = units[np.argmin(np.mean(error_rate,axis=0))]
    # train_err_vs_lambda = np.mean(train_error,axis=0)
    # test_err_vs_lambda = np.mean(test_error,axis=0)
    # mean_w_vs_lambda = np.squeeze(np.mean(w,axis=1))
    print("========== inner loop end ================")
    
    return opt_val_err, opt_units