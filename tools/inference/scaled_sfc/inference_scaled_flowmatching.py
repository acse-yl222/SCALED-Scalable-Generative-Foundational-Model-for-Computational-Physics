from scaled.model.unets.unet_1ds import UNet1DsModel
from scaled.dataset.sfc_dataset import SFCDiffusionDatasetv1
import numpy as np
import torch
from flow_matching.solver import Solver, ODESolver
from tqdm import tqdm
from flow_matching.utils import ModelWrapper
n = 300
save_dir = "output/csv"

model = UNet1DsModel(
        in_channels=2,
        out_channels=2,
        down_block_types=("DownBlock1D", "DownBlock1D", "DownBlock1D", "DownBlock1D"),
        up_block_types=("UpBlock1D", "UpBlock1D", "UpBlock1D", "UpBlock1D"),
        block_out_channels=(128, 256, 384, 512),
        add_attention=False,
        ).to("cuda")

model.load_state_dict(torch.load("weight_save/model_25000.pth"))

train_dataset = SFCDiffusionDatasetv1(
        data_dir="data/SFC/SFC_data_csv",
        data_list=[i for i in range(3500,3590)])

data0,data1 = train_dataset[20]
data0 = data0.unsqueeze(0).to("cuda")
data1 = data1.unsqueeze(0).to("cuda")

first_two_colums = np.loadtxt("data/SFC/SFC_data_csv/data_0.csv", delimiter=",")[:, :2]


T = torch.linspace(0,1,10)
T = T.to(device='cuda')
class WrappedModel(ModelWrapper):
        def forward(self, x: torch.Tensor, t: torch.Tensor, **extras):
            return self.model(x, t,return_dict=False)[0]
        
wrapped_vf = WrappedModel(model)

solver = ODESolver(velocity_model=wrapped_vf)  # create an ODESolver class

with torch.no_grad():
        for i in tqdm(range(n)):
                data0 = solver.sample(time_grid=T, x_init=data0, method='midpoint', step_size=0.05, return_intermediates=False)  # sample from the model
                result = data0.cpu().numpy()[0]
                result = result.transpose(1, 0)[0:20550]
                result[:,0] = (result[:,0]+0.95)/25
                result[:,1] = result[:,1]/25
                result = np.concatenate([first_two_colums, result], axis=1)
                np.savetxt(save_dir+f"/data_{i}.csv", result, delimiter=",")