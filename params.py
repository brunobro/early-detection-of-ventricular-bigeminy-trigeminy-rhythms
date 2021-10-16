

'''
Useful for the other scripts
author: Bruno Rodrigues de Oliveira
contact: bruno@editorapantanal.com.br
ORCID: https://orcid.org/0000-0002-1037-6541
'''

import numpy as np

'''
Previous heartbeat number used to train ML models
'''
PREVIOUS_BEATS = 150

'''
List of rhytms according to https://archive.physionet.org/physiobank/annotations.shtml
'''
rhythms_available = ['(AB', '(AFIB', '(AFL', '(B', '(BII', '(IVR', '(N', '(NOD', '(P', '(PREX', '(SBR', '(SVTA', '(T', '(VFL', '(VT']

'''
Rhytms considered for recoginition using ML algorithms
'''
rhythms_considered = ['(B', '(T', '(N']
#rhythms_considered = ['(B', '(T']

def change_codes(data_y):
	'''
	Change codes:
		(B and (T to P = positive class
		(N to N = negative class
	'''
	codes = {
		'(B': 'P',
		'(T': 'P',
		'(N': 'N'
	}

	for key, value in codes.items():
		data_y = np.where(data_y == key, value, data_y)

	return data_y