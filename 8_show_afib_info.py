'''
Show information over (AFIB rhythms, the greater source of errors
author: Bruno Rodrigues de Oliveira
contact: bruno@editorapantanal.com.br
ORCID: https://orcid.org/0000-0002-1037-6541
'''

import numpy as np
import pandas as pd
from sklearn.model_selection import cross_validate
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import OrdinalEncoder
from sklearn.ensemble import RandomForestClassifier
import params

#Read dataset
data_X = np.array(pd.read_csv('data/data_X.csv', dtype=str))
data_y = np.array(pd.read_csv('data/data_y.csv', dtype=str))

#Compute the amount of sequences
# 1. (AFIB > (B or (T
# 2. (AFIB > (N

AFIB_BT = 0
AFIB_N  = 0

for i, rhythm_prev in enumerate(data_X[:,2]):
	if rhythm_prev == '(AFIB':
		if data_y[i] == '(N':
			AFIB_N += 1
		else:
			AFIB_BT += 1

print('Sequences')
print('\t(AFIB > (B or (T: %.d' % AFIB_BT)
print('\t(AFIB > (N: %.d' % AFIB_N)