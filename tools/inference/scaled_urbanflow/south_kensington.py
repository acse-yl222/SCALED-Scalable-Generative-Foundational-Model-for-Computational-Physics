from scaled.model.unets.unet_3ds import UNet3DsModel
from scaled.pipelines.pipline_ddim_scaled_urbanflow import SCALEDUrbanFlowPipeline
from diffusers import DDIMScheduler
from scaled.dataset.urban_flow_dataset import UrbanFlowDataset
import torch
from scaled.validation.domain_decomposition import CCSNN_FlowPastBuilding_multiGPUs
from scaled.validation.domain_decomposition import predict_model

## model preparation
def main():
    subdomain_size = (64,136,136)
    wholedomain_size = (64,976,976)
    halo_size = 8
    weight_path = 'BS/backup/unet_flow/mc_128_correction/denoising_unet-2333000.pth'
    save_dir = 'BS/result/flow_past_building_mc_v8'
    
    model  = UNet3DsModel(
        in_channels=9,
        out_channels=3,
        down_block_types=("DownBlock3D", "DownBlock3D", "DownBlock3D", "DownBlock3D"),
        up_block_types=("UpBlock3D", "UpBlock3D", "UpBlock3D", "UpBlock3D"),
        block_out_channels=(64, 128, 192, 256),
        add_attention=False,
        )
    
    model.load_state_dict(
        torch.load(weight_path,map_location='cpu')
        )
    
    val_noise_scheduler = DDIMScheduler(
        num_train_timesteps=1000,
        beta_start=0.00085,
        beta_end=0.012,
        beta_schedule="scaled_linear",
        steps_offset=1,
        clip_sample=False,
        rescale_betas_zero_snr=True,
        timestep_spacing="trailing",
        prediction_type="v_prediction",
    )
    model = predict_model(model,
                          val_noise_scheduler,
                          subdomain_size,
                          num_inference_steps=10)
    
    flow_data = UrbanFlowDataset(
        data_dir="/lustre/scratch/mmm1460/data/flow_past_building_l",
        rotato_ratio=0,
        skip_timestep=50,
        time_steps_list=[i for i in range(5000, 10000)]
        )
    
    ccsnn = CCSNN_FlowPastBuilding_multiGPUs(
        model,
        wholedomain_size,
        flow_data,
        save_dir,
    )
    ccsnn.predict(
        AI4PDEs_timesteps=50,
        subdomain_iteration=3,
        subdomain_size=subdomain_size,
        halo_size=halo_size,
        divide_number=2,
        num_gpus=8,
    )

main()