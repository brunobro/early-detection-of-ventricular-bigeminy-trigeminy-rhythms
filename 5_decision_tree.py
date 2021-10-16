'''
Plot some Decision tree of the forest and show feature importance
author: Bruno Rodrigues de Oliveira
contact: bruno@editorapantanal.com.br
ORCID: https://orcid.org/0000-0002-1037-6541
'''
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import OrdinalEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
import matplotlib.pyplot as plt
import params

'''
Machine Learning models training/test
'''

#Read dataset
data_X = np.array(pd.read_csv('data/data_X.csv', dtype=str))
data_y = np.array(pd.read_csv('data/data_y.csv', dtype=str))
data_y = data_y.ravel() #convert for avoid warning error

data_XX = np.copy(data_X)

#Encodes the rhythm data (input)
Enc_X       = OrdinalEncoder().fit(data_X[:,2].reshape(-1, 1))
data_X[:,2] = Enc_X.transform(data_X[:,2].reshape(-1, 1))[:,0]
data_X      = data_X.astype(np.float)

#Show codification
print('Codification rhythms: ')
used_code = []
for i in range(0, len(data_X)):
    if data_XX[i, 2] not in used_code:
        print(data_XX[i, 2], ' ', data_X[i, 2])
        used_code.append(data_XX[i, 2])

#Encodes the rhythm data (output)
data_y = params.change_codes(data_y)

#Divide dataset
X_train, X_test, y_train, y_test = train_test_split(data_X, data_y, test_size=0.3, random_state=0)

#Model fit and show a singular decision tree and their accuracy and tree structure
clf = DecisionTreeClassifier(random_state=0, max_depth=4).fit(X_train, y_train)
y_pred = clf.predict(X_test)

print('\nAccuracy from some DT: ', accuracy_score(y_test, y_pred))

fig = plt.figure(dpi=300, figsize=(12,12))
axes = fig.add_subplot(111)
tree.plot_tree(clf, 
	feature_names=['Normal', 'Abnormal', 'Rhythm'], 
	class_names=['Negative', 'Positive'], 
	filled=True, 
	precision=2, 
	fontsize=10, 
	ax=axes)
plt.tight_layout()
plt.savefig('tree.png')

