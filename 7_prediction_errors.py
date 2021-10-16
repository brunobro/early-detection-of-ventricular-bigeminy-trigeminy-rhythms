'''
Checks which heart rhythms generate the most prediction errors
author: Bruno Rodrigues de Oliveira
contact: bruno@editorapantanal.com.br
ORCID: https://orcid.org/0000-0002-1037-6541
'''
import numpy as np
import pandas as pd
from sklearn.model_selection import cross_validate
from sklearn.preprocessing import OrdinalEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import params

#Read dataset
data_X = np.array(pd.read_csv('data/data_X.csv', dtype=str))
data_y = np.array(pd.read_csv('data/data_y.csv', dtype=str))
data_y = data_y.ravel() #convert for avoid warning error

#Encodes the data
Enc_X  = OrdinalEncoder().fit(data_X)
data_X = Enc_X.transform(data_X)

data_y = params.change_codes(data_y)

#Splits data into training and testing
X_train, X_test, y_train, y_test = train_test_split(data_X, data_y, test_size=0.3, random_state=0)

#Induced Random Forest
clf     = RandomForestClassifier(random_state=0).fit(X_train, y_train)
y_preds = clf.predict(X_test)

#Store erros in prediction
previous_seg  = []
predicted_seg = []
posterior_seg = []

for i, y_pred in enumerate(y_preds):
	if y_pred != y_test[i]:
		previous_seg.append(X_test[i])
		predicted_seg.append(y_pred)
		posterior_seg.append(y_test[i])

#Comvert to array
previous_seg  = np.array(previous_seg)
posterior_seg = np.array(posterior_seg)
predicted_seg = np.array(predicted_seg)

#Decode
previous_seg  = Enc_X.inverse_transform(previous_seg)
errors = np.column_stack((previous_seg, posterior_seg, predicted_seg))

'''
Create a DataFrame and count occurrence
'''
print('Ocurrences of prediction errors')
df = pd.DataFrame(errors, columns=['N', 'O', 'RHYTHM_PREV', 'RHYTHM_POST', 'RHYTHM_PRED'])
print(df.groupby(['RHYTHM_PREV', 'RHYTHM_POST', 'RHYTHM_PRED']).size())
