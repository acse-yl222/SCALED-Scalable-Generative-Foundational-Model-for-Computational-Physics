from scaled.model.transformers.dit_transformers import DiTTransformer1DModel
from scaled.dataset.sfc_dataset import SFCDataset
import numpy as np
import torch
from tqdm import tqdm

n = 300
save_dir = "output/csv"

model = DiTTransformer1DModel(in_channels=2,out_channels=2,sample_size=20560,patch_size=16,num_layers=16,attention_head_dim=36).to("cuda")

model.load_state_dict(torch.load("weight_save/transformer/denoising_unet-426000.pth"))


train_dataset = SFCDataset(
        data_dir="data/SFC/SFC_data_csv",
        data_list=[i for i in range(3500,3590)])

data0,data1 = train_dataset[20]
data0 = data0.unsqueeze(0).to("cuda")
data1 = data1.unsqueeze(0).to("cuda")
first_two_colums = np.loadtxt("data/SFC/SFC_data_csv/data_0.csv", delimiter=",")[:, :2]

with torch.no_grad():
        for i in tqdm(range(n)):
                data0 = model(data0).sample
                # import pdb; pdb.set_trace()
                result = data0.cpu().detach().numpy().squeeze()/10
                result = result.transpose(1, 0)[0:20550]
                result = np.concatenate([first_two_colums, result], axis=1)
                np.savetxt(save_dir+f"/data_{i}.csv", result, delimiter=",")