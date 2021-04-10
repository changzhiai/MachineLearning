# -*- coding: utf-8 -*-
"""
Created on Thu Feb 25 21:12:48 2021

@author: changai
"""

from proj1_1_load_data import *
from matplotlib.pyplot import figure, plot, title, legend, xlabel, ylabel, show

# Data attributes to be plotted
i = 0
j = 1

title('Bejaia data')
plot(X1[:, i], X1[:, j], 'o')


figure()
title('Bejaia data')
for c in range(C):
    # select indices belonging to class c:
    class_mask = y1==c
    plot(X1[class_mask,i], X1[class_mask,j], 'o',alpha=.9)

legend(classNames)
xlabel(attributeNames1[i])
ylabel(attributeNames1[j])
# Output result to screen
show()
print('Show Bejaia data')



title('Sidi data')
plot(X2[:, i], X2[:, j], 'o')

figure()
title('Sidi data')
for c in range(C):
    # select indices belonging to class c:
    class_mask = y2==c
    plot(X2[class_mask,i], X2[class_mask,j], 'o',alpha=.9)

legend(classNames)
xlabel(attributeNames2[i])
ylabel(attributeNames2[j])
# Output result to screen
show()
print('Show Sidi data')



# visualization
from matplotlib.pyplot import (figure, subplot, plot, xlabel, ylabel, 
                               xticks, yticks,legend,show)

figure(figsize=(12,10))
for m1 in range(M):
    for m2 in range(M):
        subplot(M, M, m1*M + m2 + 1)
        for c in range(C):
            class_mask = (y==c)
            plot(np.array(X[class_mask,m2]), np.array(X[class_mask,m1]), '.')
            if m1==M-1:
                xlabel(attributeNames1[m2])
            else:
                xticks([])
            if m2==0:
                ylabel(attributeNames1[m1])
            else:
                yticks([])
            #ylim(0,X.max()*1.1)
            #xlim(0,X.max()*1.1)
legend(['Not fire', 'Fire'])

show()


