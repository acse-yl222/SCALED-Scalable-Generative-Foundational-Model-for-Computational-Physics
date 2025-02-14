import os
from torch.utils.data import Dataset
import numpy as np
import torch

class SFCDataset(Dataset):
    def __init__(
            self,
            data_dir='',
            data_list=range(1000),
            skip_timestep=5):
        self.data_dir = data_dir
        self.skip_timestep = skip_timestep
        self.data_list = [f"data_{i}.csv" for i in data_list]

    def __len__(self):
        return len(self.data_list)-self.skip_timestep

    def get_data(self,timestep):
        result = np.zeros((20560,2))
        data = np.loadtxt(os.path.join(self.data_dir, self.data_list[timestep]),delimiter=',')[:,2:]
        data = data
        result[:data.shape[0],:] = data
        result = result.transpose(1, 0)
        return result*10

    def __getitem__(self,idx):
        time_step = idx
        ori_data = self.get_data(time_step)
        future_data = self.get_data(time_step+self.skip_timestep)
        return torch.from_numpy(ori_data).float(),torch.from_numpy(future_data).float()