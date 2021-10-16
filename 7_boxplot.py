'''
Plot a boxplot with results of individual trees
author: Bruno Rodrigues de Oliveira
contact: bruno@editorapantanal.com.br
ORCID: https://orcid.org/0000-0002-1037-6541
'''
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import OrdinalEncoder
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import params

'''
Machine Learning models training/test
'''

#Read dataset
data_X = np.array(pd.read_csv('data/data_X.csv', dtype=str))
data_y = np.array(pd.read_csv('data/data_y.csv', dtype=str))
data_y = data_y.ravel() #convert for avoid warning error

#Encodes the rhythm data (input)
Enc_X       = OrdinalEncoder().fit(data_X[:,2].reshape(-1, 1))
data_X[:,2] = Enc_X.transform(data_X[:,2].reshape(-1, 1))[:,0]
data_X      = data_X.astype(np.float)

#Encodes the rhythm data (output)
data_y = params.change_codes(data_y)
data_y = OrdinalEncoder().fit_transform(data_y.reshape(-1, 1)).ravel()

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

#Model fit and show a singular decision tree and their accuracy and tree structure
clf = RandomForestClassifier(random_state=0).fit(X_train, y_train)

#DT fitted
trees = clf.estimators_

#List of measures
acc_l = []
se_l  = []
sp_l  = []

for tree in trees:
    y_pred = tree.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)

    #Accuracy
    acc_l.append(acc(cm))
            
    #Sensitivity
    se_l.append(se(cm))
            
    #Specificity
    sp_l.append(sp(cm))


plt.figure(dpi=150)
plt.boxplot([acc_l, se_l, sp_l])
plt.ylabel('Performance')
plt.xticks([1, 2, 3], ['$A_{cc}$', '$S_e$', '$S_p$'])
plt.grid()
plt.savefig('boxplot.png', dpi=150)
plt.show()

