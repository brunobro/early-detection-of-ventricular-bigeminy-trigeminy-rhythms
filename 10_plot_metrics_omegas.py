'''
Plot the performance for many values o Omega_Q
author: Bruno Rodrigues de Oliveira
contact: bruno@editorapantanal.com.br
ORCID: https://orcid.org/0000-0002-1037-6541
'''
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

'''
Machine Learning models training/test
'''

#Read dataset
results = np.array(pd.read_excel('results_other_omegas.xlsx'))

plt.figure(dpi=150)
plt.plot(results[:,0], results[:,1], label='Accuracy')
plt.plot(results[:,0], results[:,2], label='Sensitivity')
plt.plot(results[:,0], results[:,3], label='Specificity')
plt.ylabel('Performance')
plt.xlabel('$\Omega_Q$')
plt.xticks(np.arange(10, 160, 10))
plt.grid()
plt.legend()
plt.savefig('plot.png', dpi=150)
plt.show()