import numpy as np
import os
data_path = 'data/SFC_data_csv'
data_list = os.listdir(data_path)

for file_name in data_list:
    data = os.path.join(data_path,file_name)
    data = np.loadtxt(data,delimiter=',')[:,2:]
    data = data*10
    data[:,0] = (data[:,0]-0.25)/4*5*2
    data[:,1] = data[:,1]/2*5
    print(data[:,0].max(),data[:,0].min())
    print(data[:,1].max(),data[:,1].min())
    