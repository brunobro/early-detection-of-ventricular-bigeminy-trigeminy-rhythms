'''
Count the sequence of heart rhythms '(N before (B' and '(B before (N'
author: Bruno Rodrigues de Oliveira
contact: bruno@editorapantanal.com.br
ORCID: https://orcid.org/0000-0002-1037-6541
'''
import numpy as np
import pandas as pd

'''
Machine Learning models training/test
'''

#Read dataset
data_X = np.array(pd.read_csv('data/data_X.csv', dtype=str))
data_y = np.array(pd.read_csv('data/data_y.csv', dtype=str))
data_y = data_y.ravel() #convert for avoid warning error

BN = 0
NB = 0
NT = 0
TN = 0

for i, dx in enumerate(data_X):
    dy = data_y[i]
    
    if '(N' in dx:
        if dy == '(B':
            NB += 1
        elif dy == '(T':
            NT += 1
    
    if '(B' in dx:
        if dy == '(N':
            BN += 1
    
    if '(T' in dx:
        if dy == '(N':
            TN += 1
            
        
    
print('Heart rhythms sequence')
print('(N before (B', NB)
print('(B before (N', BN)
print('(N before (T', NT)
print('(T before (N', TN)
