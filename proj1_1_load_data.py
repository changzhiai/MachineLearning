# -*- coding: utf-8 -*-
"""
Created on Thu Feb 25 20:04:22 2021

@author: changai
"""

import numpy as np
import pandas as pd

filename1 = './Bejaia.csv'
filename2 = './Sidi.csv'

df1 = pd.read_csv(filename1)
df2 = pd.read_csv(filename2)

raw_data1 = df1.values
raw_data2 = df2.values
#print(raw_data1)
#print(raw_data2)

cols = range(3, 13) 
X1 = raw_data1[:, cols]
X2 = raw_data2[:, cols]
X = np.concatenate((X1, X2))

attributeNames1 = np.asarray(df1.columns[cols])
attributeNames2 = np.asarray(df2.columns[cols])
print(attributeNames1)
print(attributeNames2)

classLabels1_raw = raw_data1[:,-1]
classLabels2_raw = raw_data2[:,-1]
#print(classLabels1_raw)

classLabels1 = [x1.strip() for x1 in classLabels1_raw]  #remove the spaces of beginning and end
classLabels2 = [x2.strip() for x2 in classLabels2_raw]
#print(classLabels1)

classNames = np.unique(classLabels1)
print(classNames)
classDict = dict(zip(classNames,[1, 0])) #{'fire': 1, 'not fire': 0}
#classDict = dict(zip(['not fire', 'fire'],[0, 1])) #{'fire': 1, 'not fire': 0}

y1 = np.array([classDict[cl] for cl in classLabels1])
y2 = np.array([classDict[cl] for cl in classLabels2])
y = np.concatenate((y1, y2))

print(X1.shape)
print(X2.shape)
N, M = X1.shape
C = len(classNames)