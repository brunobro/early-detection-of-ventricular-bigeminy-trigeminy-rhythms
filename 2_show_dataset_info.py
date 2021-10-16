'''
Show information over datasets
author: Bruno Rodrigues de Oliveira
contact: bruno@editorapantanal.com.br
ORCID: https://orcid.org/0000-0002-1037-6541
'''
import os
import numpy as np
import pandas as pd

'''
Path of ECG annotations files
'''
dirs = ['mitdb', 'ltafdb']

'''
To store heartbeat type and the rhythm
'''
for dir in dirs:
	path = 'data/' + dir + '/'

	print('Database: ', dir)

	B_rhythm = 0
	T_rhythm = 0
	N_rhythm = 0

	for _, _, files in os.walk(path): #Read all data
		
		for file in files:
			
			#Create DataFrame to each ECG register
			df = pd.read_csv('data/all/'+ dir + file.replace('txt', 'csv'), dtype=str)

			ndf = df.loc[df['TYPE_ANN'] == '+']

			for idx in ndf.index:

				data_output = df.loc[idx, 'RHYTHM'] #the is the current rhythm

				if data_output == '(N':
					N_rhythm += 1
				elif data_output == '(B':
					B_rhythm += 1
				elif data_output == '(T':
					T_rhythm += 1

	print('B:%d, T:%d , N:%d ' % (B_rhythm, T_rhythm, N_rhythm))