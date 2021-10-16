'''
Organizes the data to classify heart rates using past heartbeat information.
author: Bruno Rodrigues de Oliveira
contact: bruno@editorapantanal.com.br
ORCID: https://orcid.org/0000-0002-1037-6541
'''
import os
import numpy as np
import pandas as pd
import params

'''
Read all data and create Dataset for training and test ML models with 3 features (inputs) and 1 output: 
input: 1) and 2) number of Normal and Abnormal heartbeats and 3) previous rhythm
output: rhythm to be predicted, that can be: "B" bimeniny and "T" trigeminy
'''
data_X = [] #store inputs
data_y = [] #store outputs

for _, _, files in os.walk('data/all/'): #Read all data
	
	for file in files:
		
		#Create DataFrame to each ECG register
		df = pd.read_csv('data/all/'+ file, dtype=str)

		ndf = df.loc[df['TYPE_ANN'] == '+']

		for idx in ndf.index:

			data_output = df.loc[idx, 'RHYTHM'] #the is the current rhythm

			if data_output in params.rhythms_considered:

				start = idx - params.PREVIOUS_BEATS #Start of interval

				if start > 0:
					
					info = np.array(df.loc[start : idx - 1, ['TYPE_ANN', 'RHYTHM']]) #Data interval
					
					if info[0][1] in params.rhythms_available: #Only for valid rhythms

						if np.count_nonzero(info == '+') == 0: #Disregards segments with changes in pace
							cN = np.count_nonzero(info == 'N')
							cO = np.count_nonzero(info != 'N')
							T  = cN + cO

							#Normalize
							cN = cN/T
							cO = cO/T

							#Append the data
							data_X.append([cN, cO, info[0][1]]) #info[0][1] is the previous rhythm
							data_y.append(data_output)

'''
Create a DataFrame and save to csv
'''
df = pd.DataFrame(data_X, columns=['N', 'O', 'RHYTHM'])
df.to_csv('data/data_X.csv', index=False)

df = pd.DataFrame(data_y, columns=['RHYTHM'])
df.to_csv('data/data_y.csv', index=False)