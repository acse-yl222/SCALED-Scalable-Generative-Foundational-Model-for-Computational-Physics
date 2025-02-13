import random
from accelerate.logging import get_logger
logger = get_logger(__name__, log_level="INFO")
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.gridspec as gridspec
from scaled.pipelines.pipline_ddim_scaled_urbanflow import SCALEDUrbanFlowPipeline
import torch
from scaled.model.unets.unet_3ds import UNet3DsModel
import torch.nn as nn

class Net(nn.Module):
    def __init__(
            self,
            denoising_unet: UNet3DsModel,
    ):
        super().__init__()
        self.denoising_unet = denoising_unet
    def forward(
            self,
            noisy_latents,
            timesteps,
    ):
        model_pred = self.denoising_unet(
        noisy_latents,
        timesteps,
        ).sample
        return model_pred


def compute_snr(noise_scheduler, timesteps):
    """
    Computes SNR as per
    https://github.com/TiankaiHang/Min-SNR-Diffusion-Training/blob/521b624bd70c67cee4bdf49225915f5945a872e3/guided_diffusion/gaussian_diffusion.py#L847-L849
    """
    alphas_cumprod = noise_scheduler.alphas_cumprod
    sqrt_alphas_cumprod = alphas_cumprod ** 0.5
    sqrt_one_minus_alphas_cumprod = (1.0 - alphas_cumprod) ** 0.5
    sqrt_alphas_cumprod = sqrt_alphas_cumprod.to(device=timesteps.device)[
        timesteps
    ].float()
    while len(sqrt_alphas_cumprod.shape) < len(timesteps.shape):
        sqrt_alphas_cumprod = sqrt_alphas_cumprod[..., None]
    alpha = sqrt_alphas_cumprod.expand(timesteps.shape)
    sqrt_one_minus_alphas_cumprod = sqrt_one_minus_alphas_cumprod.to(
        device=timesteps.device
    )[timesteps].float()
    while len(sqrt_one_minus_alphas_cumprod.shape) < len(timesteps.shape):
        sqrt_one_minus_alphas_cumprod = sqrt_one_minus_alphas_cumprod[..., None]
    sigma = sqrt_one_minus_alphas_cumprod.expand(timesteps.shape)
    snr = (alpha / sigma) ** 2
    return snr

def log_validation(
        denoising_unet,
        scheduler,
        accelerator,
        generator=None,
        valid_dataset=None
):
    logger.info("Running validation... ")
    if generator is None:
        generator = torch.manual_seed(42)
    dataset_len = len(valid_dataset)
    sample_idx = [random.randint(0, dataset_len) for _ in range(1)]

    pipe = SCALEDUrbanFlowPipeline(
        denoising_unet,
        scheduler=scheduler,
    )
    pipe = pipe.to(accelerator.device)
    results = {}
    ori_data, gt_result = valid_dataset[sample_idx[0]] # condition （9x32x32x32） gt_result (9x32x32x32)
    previous_value = ori_data[:3].unsqueeze(0).to('cuda')
    control_value = ori_data[3:].unsqueeze(0).to('cuda')
    next_value = gt_result[:3].unsqueeze(0).to('cuda')
    background_value = control_value.clone().bool()
    back_data = next_value.clone()
    back_data[:, :, 1:-1] = 1
    back_data[:, 0:1][background_value] = 0
    back_data[:, 1:2][background_value] = 0
    back_data[:, 2:3][background_value] = 0
    pre = pipe(
        previous_value,
        back_data,
        num_inference_steps=25,
        guidance_scale=0,
        depth=64,
        height=128,
        width=128,
        generator=generator,
        return_dict=False,
    )
    results['WithoutBackground'] = {
        "prediction_flow": pre.detach().cpu().numpy()[0],
        "gt_flow": next_value.detach().cpu().numpy()[0],
        "original_flow": previous_value.detach().cpu().numpy()[0]
    }
    del pipe
    return results

def visualize_with_diff(data_pre, data_gt, data_ori, filename):
    # Create a figure with a larger size and higher resolution
    fig = plt.figure(figsize=(24, 24), dpi=300)  # Adjust figsize to accommodate diff images
    gs = gridspec.GridSpec(4, 4, width_ratios=[1, 1, 1, 0.1])  # Add extra row for diff and column for colorbars
    
    for i in range(3):
        ax1 = plt.subplot(gs[i, 0])
        im1 = ax1.imshow(data_pre[i, 4], vmin=-1, vmax=1)
        ax1.set_title('Prediction')
        ax1.axis('off')
        
        ax2 = plt.subplot(gs[i, 1])
        im2 = ax2.imshow(data_gt[i, 4], vmin=-1, vmax=1)
        ax2.set_title('Ground Truth')
        ax2.axis('off')
        
        ax3 = plt.subplot(gs[i, 2])
        im3 = ax3.imshow(data_ori[i, 4], vmin=-1, vmax=1)
        ax3.set_title('Original Data')
        ax3.axis('off')
        
        # Add colorbars in the 4th column
        cbar_ax1 = plt.subplot(gs[i, 3])
        fig.colorbar(im1, cax=cbar_ax1)
        
        cbar_ax2 = plt.subplot(gs[i, 3])
        fig.colorbar(im2, cax=cbar_ax2)
        
        cbar_ax3 = plt.subplot(gs[i, 3])
        fig.colorbar(im3, cax=cbar_ax3)
    
    # Now plot the differences in the 4th row
    diff_pre_gt = np.abs(data_pre - data_gt)  # Difference between Prediction and Ground Truth
    diff_pre_ori = np.abs(data_pre - data_ori)  # Difference between Prediction and Original
    diff_gt_ori = np.abs(data_gt - data_ori)  # Difference between Ground Truth and Original
    
    ax4 = plt.subplot(gs[3, 0])
    im4 = ax4.imshow(diff_pre_gt[0, 4], vmin=0, vmax=0.25)
    ax4.set_title('Diff Pre-GT')
    ax4.axis('off')
    
    ax5 = plt.subplot(gs[3, 1])
    im5 = ax5.imshow(diff_pre_ori[0, 4], vmin=0, vmax=0.25)
    ax5.set_title('Diff Pre-Ori')
    ax5.axis('off')
    
    ax6 = plt.subplot(gs[3, 2])
    im6 = ax6.imshow(diff_gt_ori[0, 4], vmin=0, vmax=0.25)
    ax6.set_title('Diff GT-Ori')
    ax6.axis('off')
    
    # Add colorbars for the differences
    cbar_ax4 = plt.subplot(gs[3, 3])
    fig.colorbar(im4, cax=cbar_ax4)
    
    cbar_ax5 = plt.subplot(gs[3, 3])
    fig.colorbar(im5, cax=cbar_ax5)
    
    cbar_ax6 = plt.subplot(gs[3, 3])
    fig.colorbar(im6, cax=cbar_ax6)
    
    # Adjust layout to avoid overlap
    plt.tight_layout()
    
    # Save the figure with higher resolution
    plt.savefig(filename, dpi=300)
    plt.close(fig)
