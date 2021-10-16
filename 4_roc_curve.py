'''
Machine Learning models training/test
Plot the ROC curve
author: Bruno Rodrigues de Oliveira
contact: bruno@editorapantanal.com.br
ORCID: https://orcid.org/0000-0002-1037-6541
'''

import numpy as np
import pandas as pd
from sklearn.model_selection import cross_validate
from sklearn.metrics import confusion_matrix, auc, plot_roc_curve
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import OrdinalEncoder
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import params

#Read dataset
data_X = np.array(pd.read_csv('data/data_X.csv', dtype=str))
data_y = np.array(pd.read_csv('data/data_y.csv', dtype=str))
data_y = data_y.ravel() #convert for avoid warning error

#Encodes the data
Enc_X  = OrdinalEncoder().fit(data_X)
data_X = Enc_X.transform(data_X)

data_y = params.change_codes(data_y)

#5-fold cross-validation
cv  = StratifiedKFold(n_splits = 5)

#Random Forest
clf = RandomForestClassifier(random_state=0)

tprs = []
aucs = []
mean_fpr = np.linspace(0, 1, 100)

fig = plt.figure(dpi=150)
ax = fig.add_subplot(111)
ax.grid()
for i, (train, test) in enumerate(cv.split(data_X, data_y)):
    clf.fit(data_X[train], data_y[train])

    viz           = plot_roc_curve(clf, data_X[test], data_y[test], name='ROC fold {}'.format(i + 1), alpha=0.5, lw=1, ax=ax)
    interp_tpr    = np.interp(mean_fpr, viz.fpr, viz.tpr)
    interp_tpr[0] = 0.0

    tprs.append(interp_tpr)
    aucs.append(viz.roc_auc)

ax.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r', label='Chance', alpha=.8)

mean_tpr     = np.mean(tprs, axis=0)
mean_tpr[-1] = 1.0
mean_auc     = auc(mean_fpr, mean_tpr)
std_auc      = np.std(aucs)
ax.plot(mean_fpr, mean_tpr, color='k', label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc), lw=2, alpha=.8)

std_tpr    = np.std(tprs, axis=0)
tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
ax.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2, label=r'$\pm \sigma$')

ax.set(xlim=[-0.05, 1.05], ylim=[-0.05, 1.05])
ax.legend(loc="lower right")
plt.show()
