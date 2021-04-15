# -*- coding: utf-8 -*-
"""
Created on Sat Apr 10 23:56:35 2021

@author: changai
"""
from proj1_1_load_data import *
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat
import torch
from sklearn import model_selection
from toolbox_02450 import train_neural_net, draw_neural_net
from scipy import stats

y = X[:,9].astype('float')
y = y.squeeze()
y = np.reshape(y,(244,1))

X = X.squeeze()
X = X[:,range(0,8)].astype(float)
X = X.astype(float)

N,M = X.shape 

X = X - np.ones((N,1)) * X.mean(axis=0)
#print(X1.mean(axis=0)) #obtain mean value along column, will get 10 numbers, array(1, M)
#print(Y1)

#normalizing matrix
X = X*(1/np.std(X,axis=0))

# Add offset attribute
# X = np.concatenate((np.ones((X.shape[0],1)),X),1).astype('float')
# print(X.shape)
# attributeNames = [u'Offset']+attributeNames1
# M = M+1

#Downsample: X = X[1:20,:] y = y[1:20,:]
N, M = X.shape
print(X.shape)
# # print(np.reshape(y,(244,1)))
# # print(np.asmatrix(y).shape)
# print(attributeNames1.tolist())
# assert False

attributeNames = attributeNames1[range(0,8)].tolist()
# Normalize data
# X = stats.zscore(X);

for n_hidd in range(1,10):
    
    # Parameters for neural network classifier
    n_hidden_units = n_hidd     # number of hidden units
    n_replicates = 2        # number of networks trained in each k-fold
    max_iter = 10000         # stop criterion 2 (max epochs in training)
    
    # K-fold crossvalidation
    K = 3                   # only five folds to speed up this example
    CV = model_selection.KFold(K, shuffle=True)
    # Make figure for holding summaries (errors and learning curves)
    summaries, summaries_axes = plt.subplots(1,2, figsize=(10,5))
    # Make a list for storing assigned color of learning curve for up to K=10
    color_list = ['tab:orange', 'tab:green', 'tab:purple', 'tab:brown', 'tab:pink',
                  'tab:gray', 'tab:olive', 'tab:cyan', 'tab:red', 'tab:blue']
    
    # Define the model, see also Exercise 8.2.2-script for more information.
    model = lambda: torch.nn.Sequential(
                        torch.nn.Linear(M, n_hidden_units), #M features to H hiden units
                        torch.nn.Tanh(),   # 1st transfer function,
                        torch.nn.Linear(n_hidden_units, 1), # H hidden units to 1 output neuron
                        # torch.nn.Sigmoid() # final tranfer function
                        )
    loss_fn = torch.nn.MSELoss()
    
    print('Training model of type:\n\n{}\n'.format(str(model())))
    errors = [] # make a list for storing generalizaition error in each loop
    for k, (train_index, test_index) in enumerate(CV.split(X,y)): 
        print('\nCrossvalidation fold: {0}/{1}'.format(k+1,K))    
        
        # Extract training and test set for current CV fold, convert to tensors
        X_train = torch.Tensor(X[train_index,:])
        y_train = torch.Tensor(y[train_index])
        X_test = torch.Tensor(X[test_index,:])
        y_test = torch.Tensor(y[test_index])
        
        # Train the net on training data
        net, final_loss, learning_curve = train_neural_net(model,
                                                           loss_fn,
                                                           X=X_train,
                                                           y=y_train,
                                                           n_replicates=n_replicates,
                                                           max_iter=max_iter)
        
        print('\n\tBest loss: {}\n'.format(final_loss))
        
        # Determine estimated class labels for test set
        y_test_est = net(X_test)
        
        # Determine errors and errors
        se = (y_test_est.float()-y_test.float())**2 # squared error
        mse = (sum(se).type(torch.float)/len(y_test)).data.numpy() #mean
        errors.append(mse) # store error rate for current CV fold 
        
        # Display the learning curve for the best net in the current fold
        h, = summaries_axes[0].plot(learning_curve, color=color_list[k])
        h.set_label('CV fold {0}'.format(k+1))
        summaries_axes[0].set_xlabel('Iterations')
        summaries_axes[0].set_xlim((0, max_iter))
        summaries_axes[0].set_ylabel('Loss')
        summaries_axes[0].set_title('Learning curves: {} hidden unit(s)'.format(n_hidd))
        
    # Display the MSE across folds
    summaries_axes[1].bar(np.arange(1, K+1), np.squeeze(np.asarray(errors)), color=color_list)
    summaries_axes[1].set_xlabel('Fold')
    summaries_axes[1].set_xticks(np.arange(1, K+1))
    summaries_axes[1].set_ylabel('MSE')
    summaries_axes[1].set_title('Test mean-squared-error')
        
    print('Diagram of best neural net in last fold:')
    weights = [net[i].weight.data.numpy().T for i in [0,2]]
    biases = [net[i].bias.data.numpy() for i in [0,2]]
    tf =  [str(net[i]) for i in [1,2]]
    draw_neural_net(weights, biases, tf, attribute_names=attributeNames)
    
    # Print the average classification error rate
    print('\nEstimated generalization error, RMSE: {0}'.format(round(np.sqrt(np.mean(errors)), 4)))
    
    # When dealing with regression outputs, a simple way of looking at the quality
    # of predictions visually is by plotting the estimated value as a function of 
    # the true/known value - these values should all be along a straight line "y=x", 
    # and if the points are above the line, the model overestimates, whereas if the
    # points are below the y=x line, then the model underestimates the value
    plt.figure(figsize=(10,10))
    y_est = y_test_est.data.numpy(); y_true = y_test.data.numpy()
    axis_range = [np.min([y_est, y_true])-1,np.max([y_est, y_true])+1]
    plt.plot(axis_range,axis_range,'k--')
    plt.plot(y_true, y_est,'ob',alpha=.25)
    plt.legend(['Perfect estimation','Model estimations'])
    plt.title('Alcohol content: estimated versus true value (for last CV-fold)')
    plt.ylim(axis_range); plt.xlim(axis_range)
    plt.xlabel('True value')
    plt.ylabel('Estimated value')
    plt.grid()
    
    plt.show()    
