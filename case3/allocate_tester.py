import pandas as pd
import scipy
import numpy as np

from allocate import allocate_portfolio

pred1_df = pd.read_csv('Predicted Testing Data Analyst 1.csv')
pred2_df = pd.read_csv('Predicted Testing Data Analyst 2.csv')
pred3_df = pd.read_csv('Predicted Testing Data Analyst 3.csv')

df = pd.read_csv('actual_testing_data.csv')

data = np.asarray([df[df.columns[i]] for i in range(1, len(df.columns))])
pred1 = np.asarray([pred1_df[df.columns[i]] for i in range(1, len(df.columns))])
pred2 = np.asarray([pred2_df[df.columns[i]] for i in range(1, len(df.columns))])
pred3 = np.asarray([pred3_df[df.columns[i]] for i in range(1, len(df.columns))])

rp = np.zeros((data.shape[1]-1))

for i in range(0, data.shape[1]-1):
# for i in range(0, 100):
    w = allocate_portfolio(data[:,i].tolist(), pred1[:,i].tolist(), pred2[:,i].tolist(), pred3[:,i].tolist())

    if i > 0:
    # rp[i] = w.T @ ((data[:,i]-data[:,i-1]) / data[:,i-1])
        rp[i-1] = np.sum(w * ((data[:,i+1]-data[:,i]) / data[:,i]))

print(rp.shape)
print('Sharpe ratio: ' + str((252**0.5)*np.mean(rp)/np.std(rp)))