# -*- coding: utf-8 -*-
"""
Created on Fri Apr 16 00:47:48 2021

@author: changai
"""
from proj1_1_load_data import *
from sklearn.naive_bayes import MultinomialNB
from sklearn import model_selection
from sklearn.preprocessing import OneHotEncoder
from matplotlib.pyplot import figure, plot, xlabel, ylabel, legend, show, boxplot
from proj3_5_naive_Bayes_classification_validation import NB_validate

# y = X[:,9].astype('float')
y = y.squeeze()
# y = np.reshape(y,(244,1))
print(y)

X = X.squeeze()
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

np.random.seed(2450)

# Naive Bayes classifier parameters
# alpha = 1 # pseudo-count, additive parameter (Laplace correction if 1.0 or Lidtstone smoothing otherwise)
alpha = np.power(10.,range(-5,9))
fit_prior = True   # uniform prior (change to True to estimate prior from data)

# K-fold crossvalidation
K = 10
CV = model_selection.KFold(n_splits=K,shuffle=True)

X = OneHotEncoder().fit_transform(X=X)
# print(X.shape)


k=0
opt_alphas= []
errors = np.zeros(K)
for train_index, test_index in CV.split(X):
    #print('Crossvalidation fold: {0}/{1}'.format(k+1,K))

    # extract training and test set for current CV fold
    X_train = X[train_index,:]
    y_train = y[train_index]
    X_test = X[test_index,:]
    y_test = y[test_index]
    
    internal_cross_validation = 10
    opt_val_err, opt_alpha, test_err_vs_alpha = NB_validate(X_train, y_train, alpha, internal_cross_validation)
    opt_alphas.append(opt_alpha)
    
    nb_classifier = MultinomialNB(alpha=opt_alpha,
                                  fit_prior=fit_prior)
    nb_classifier.fit(X_train, y_train)
    y_est_prob = nb_classifier.predict_proba(X_test)
    y_est = np.argmax(y_est_prob,1)
    print(y_est.shape, y_test.shape)

    errors[k] = np.sum(y_est!=y_test,dtype=float)/y_test.shape[0]
    if k == K-1: 
        # f = figure()
        # print(errors)
        # boxplot(errors.T)
        # xlabel('Model complexity (max tree depth)')
        # ylabel('Test error across CV folds')
        
        f = figure()
        plot(range(K), errors)
        xlabel('Model complexity (max tree depth)')
        ylabel('Error (misclassification rate)')
        legend(['Error_train','Error_test'])
            
        show()
    
    k+=1

print('Error rate: {0}%'.format(100*np.mean(errors)))

print('\n +++++++ KNN output ++++++++')
print('optimized alpha:', opt_alphas)
print('test errors', errors)

# figure()
# plot(range(K), 100*errors)
# xlabel('Numbers of K')
# ylabel('Classification error rate (%)')
# show()