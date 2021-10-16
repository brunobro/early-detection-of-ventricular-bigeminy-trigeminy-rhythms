
'''
Prepare data by removing unimportant information. 
Creates a csv file with only the data that will be used for the recognition of cardiac rhythms.
author: Bruno Rodrigues de Oliveira
contact: bruno@editorapantanal.com.br
ORCID: https://orcid.org/0000-0002-1037-6541
'''
import os
import pandas as pd

'''
For remove duplicate spaces
'''
def remove(line):
	for i in range(0, 3):
		line = line.replace('  ', ' ')
	return line

'''
Path of ECG annotations files
'''
dirs = ['mitdb/', 'ltafdb/']

'''
To store heartbeat type and the rhythm
'''
for dir in dirs:
	path = 'data/' + dir

	for _, _, files in os.walk(path):

		for file in files:

			data = [] #To store data

			'''
			Read ECG annotations
			'''
			f = open(path + file)
			lines = f.readlines()
			f.close()

			'''
			Store heartbeat type and the rhythm
			'''
			rhythm = ''
			for line in lines:
				sep = remove(line).split(' ')
				r   = sep[6].split('\t')
				if len(r) > 1: #If there is no change of pace, keep the last
					rhythm = r[1].replace('\n', '') #rhythm

				'''
				Identification of ECG
				'''
				id_reg = file.replace('.txt', '')
				id_reg = dir.replace('/', '') + id_reg

				data.append([id_reg, sep[3], rhythm])


			'''
			Create a DataFrame and save to csv
			'''
			df = pd.DataFrame(data, columns=['ID_ECG', 'TYPE_ANN', 'RHYTHM'])
			df.to_csv('data/all/' + id_reg + '.csv', index=False)