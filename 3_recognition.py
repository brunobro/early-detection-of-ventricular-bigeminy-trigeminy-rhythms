'''
Machine Learning models training/test
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
data_y = data_y.ravel() #convert for avoid warning error

'''
Show dataset information
'''
print('Total of predicted rhythms')
for rhythm in params.rhythms_considered:
	print('Rhythm: ', rhythm)
	print(np.count_nonzero(data_y  == rhythm))

Nbeats = data_X[:,0].astype(np.float)
Abeats = data_X[:,1].astype(np.float)

print('Normal beats: ', np.mean(Nbeats), ' ', np.var(Nbeats))
print('Abnormal beats: ', np.mean(Abeats), ' ', np.var(Abeats))

print('Total of previous rhythms')
for rhythm in params.rhythms_available:
	print(rhythm, ': ', np.count_nonzero(data_X  == rhythm))

#Encodes the data
Enc_X  = OrdinalEncoder().fit(data_X)
data_X = Enc_X.transform(data_X)

data_y = params.change_codes(data_y)

#5-fold cross-validation
skf = StratifiedKFold(n_splits=5)

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

acc_l = []

se_l = []

sp_l = []

i = 1

for train_index, test_index in skf.split(data_X, data_y):
	
	print('\n>>> Fold ', i, ' <<<')
	i += 1

	X_train, X_test = data_X[train_index], data_X[test_index]
	y_train, y_test = data_y[train_index], data_y[test_index]

	#Random Forest learning
	clf = RandomForestClassifier(random_state=0).fit(X_train, y_train)

	#Done predictions
	y_pred = clf.predict(X_test) #for test dataset
	#y_pred = clf.predict(X_train) #for training dataset

	#Confusion matrix
	cm = confusion_matrix(y_test, y_pred, labels=['N', 'P']) #for test dataset
	#cm = confusion_matrix(y_train, y_pred, labels=['N', 'P']) #for training dataset

	#Performance measures

	#Accuracy
	acc_c = acc(cm)

	print('\n\tAccuracy: %.4f' % (acc_c))

	acc_l.append(acc_c)
	
	#Sensitivity
	se_c = se(cm)

	print('\n\tSensitivity: %.4f' % (se_c))

	se_l.append(se_c)
	
	#Specificity
	sp_c = sp(cm)
	
	print('\n\tSpecificity: %.4f' % (sp_c))

	sp_l.append(sp_c)

print('\nAverage Accuracy: %.4f' % (np.mean(acc_l)))
print('\nAverage Sensitivity: %.4f' % (np.mean(se_l)))
print('\nAverage Specificity: %.4f' % (np.mean(sp_l)))

'''
#Cross-validation
cv_results = cross_validate(clf, data_X, data_y, cv=10, return_train_score=True, scoring=['accuracy', 'precision_macro', 'recall_macro'])
print('\nTraining')
print('Accuray', cv_results['train_accuracy'])
print('Precision', cv_results['train_precision_macro'])
print('Sensitivity', cv_results['train_recall_macro'])

print('\nTraining - Average')
print('Accuray', np.mean(cv_results['train_accuracy']))
print('Precision', np.mean(cv_results['train_precision_macro']))
print('Sensitivity', np.mean(cv_results['train_recall_macro']))

print('\nTest')
print('Accuray', cv_results['test_accuracy'])
print('Precision', cv_results['test_precision_macro'])
print('Sensitivity', cv_results['test_recall_macro'])

print('\nTest - Average')
print('Accuray', np.mean(cv_results['test_accuracy']))
print('Precision', np.mean(cv_results['test_precision_macro']))
print('Sensitivity', np.mean(cv_results['test_recall_macro']))
'''
