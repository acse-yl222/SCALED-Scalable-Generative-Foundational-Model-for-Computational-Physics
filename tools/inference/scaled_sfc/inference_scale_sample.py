from scaled.model.unets.unet_1ds import UNet1DsModel
from scaled.dataset.sfc_dataset import SFCDataset

import numpy as np
from scaled.pipelines.pipline_ddim_scaled_sfc import SCALEDSFCPipeline
from diffusers import DDIMScheduler
import torch
from tqdm import tqdm


save_dir = "output/csv"

model = UNet1DsModel(
        in_channels=4,
        out_channels=2,
        down_block_types=("DownBlock1D", "DownBlock1D", "DownBlock1D", "DownBlock1D"),
        up_block_types=("UpBlock1D", "UpBlock1D", "UpBlock1D", "UpBlock1D"),
        block_out_channels=(128, 256, 384, 512),
        add_attention=False,
        ).to("cuda")

model.load_state_dict(torch.load("weight_save/scale_sample_sfc/denoising_unet-93000.pth",map_location='cpu'))

scheduler = DDIMScheduler(
        num_train_timesteps=1000,
        beta_start=0.00085,
        beta_end=0.012,
        beta_schedule="scaled_linear",
        steps_offset=1,
        clip_sample=False,
        rescale_betas_zero_snr=True,
        timestep_spacing="trailing",
        prediction_type="sample",
)

pipe = SCALEDSFCPipeline(
        model,
        scheduler=scheduler,
    )
pipe = pipe.to('cuda')


train_dataset = SFCDataset(
        data_dir="data/SFC_data_csv",
        data_list=[i for i in range(3500,3590)])

data0,data1 = train_dataset[20]
data0 = data0.unsqueeze(0).to("cuda")
data1 = data1.unsqueeze(0).to("cuda")

first_two_colums = np.loadtxt("/lcrc/project/SFC_Transformers/SFC-CAE/csv_data/data_0.csv", delimiter=",")[:, :2]


with torch.no_grad():
        for i in tqdm(range(100)):
                data0 = pipe(data0,
                             num_inference_steps=1000,
                                guidance_scale=0,
                                length=20560,
                                generator=None,
                                return_dict=False,)
                # import pdb; pdb.set_trace()
                result = data0.cpu().detach().numpy().squeeze()/10
                result = result.transpose(1, 0)[0:20550]
                result = np.concatenate([first_two_colums, result], axis=1)
                np.savetxt(save_dir+f"/data_{i}.csv", result, delimiter=",")