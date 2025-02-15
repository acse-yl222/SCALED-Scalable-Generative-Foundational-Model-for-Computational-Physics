from scaled.model.unets.unet_1ds import UNet1DsModel
from scaled.dataset.sfc_dataset import SFCDataset
import numpy as np
import torch
from tqdm import tqdm


save_dir = "output/csv"

model = UNet1DsModel(
        in_channels=2,
        out_channels=2,
        down_block_types=("DownBlock1D", "DownBlock1D", "DownBlock1D", "DownBlock1D"),
        up_block_types=("UpBlock1D", "UpBlock1D", "UpBlock1D", "UpBlock1D"),
        block_out_channels=(128, 256, 384, 512),
        add_attention=False,
        ).to("cuda")

model.load_state_dict(torch.load("weight_save/denoising_unet-139000.pth"))


train_dataset = SFCDataset(
        data_dir="data/SFC_data_csv",
        data_list=[i for i in range(3500,3590)])

data0,data1 = train_dataset[20]
data0 = data0.unsqueeze(0).to("cuda")
data1 = data1.unsqueeze(0).to("cuda")

first_two_colums = np.loadtxt("/lcrc/project/SFC_Transformers/SFC-CAE/csv_data/data_0.csv", delimiter=",")[:, :2]


with torch.no_grad():
        for i in tqdm(range(100)):
                data0 = model(data0).sample
                # import pdb; pdb.set_trace()
                result = data0.cpu().detach().numpy().squeeze()/10
                result = result.transpose(1, 0)[0:20550]
                result = np.concatenate([first_two_colums, result], axis=1)
                np.savetxt(save_dir+f"/data_{i}.csv", result, delimiter=",")