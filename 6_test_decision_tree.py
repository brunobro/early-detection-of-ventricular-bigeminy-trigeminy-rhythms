'''
Teste decision tree as algorithm
author: Bruno Rodrigues de Oliveira
contact: bruno@editorapantanal.com.br
ORCID: https://orcid.org/0000-0002-1037-6541
'''
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

'''
Machine Learning models training/test
'''

#Read dataset
data_X = np.array(pd.read_csv('data/data_X.csv', dtype=str))
data_y = np.array(pd.read_csv('data/data_y.csv', dtype=str))
data_y = data_y.ravel() #convert for avoid warning error

#Performance metrics
def acc(cm):
	#accuracy
	return (cm[0][0] + cm[1][1])/(cm[0][0] + cm[0][1] + cm[1][0] + cm[1][1]) #(TN + TP)/(TN + FP + FN + TP)

def se(cm):
	#sensitivity
	return cm[1][1]/(cm[1][1] + cm[1][0]) # TP/(TP + FN)

def sp(cm):
	#specificity
	return cm[0][0]/(cm[0][0] + cm[0][1]) # TN/(TN + FP)

#Divide dataset
X_train, X_test, y_train, y_test = train_test_split(data_X, data_y, test_size=0.3, random_state=0)

#List of measures
acc_l = []
se_l  = []
sp_l  = []

y_pred = []

for x_test in X_test:
    
    #Decision obtained from Tree structure
    if x_test[2] in ['(AB', '(AFIB', '(B', '(IVR']:
        y_pred.append('(N')
    else:
        if x_test[2] in ['(N', '(NOD', '(P']:
            y_pred.append('(B')
        else:
            y_pred.append('(N')
 
#Calculate metrics
cm = confusion_matrix(y_test, y_pred)

#Accuracy
acc_c = acc(cm)
print('\n\tAccuracy: %.4f' % (acc_c))
      	
#Sensitivity
se_c = se(cm)
print('\n\tSensitivity: %.4f' % (se_c))
      	
#Specificity
sp_c = sp(cm)
print('\n\tSpecificity: %.4f' % (sp_c))