from dataclasses import dataclass
from scaled.model.unets.unet_1ds import UNet1DsModel
import torch
from PIL import Image
from diffusers import DDPMScheduler
import torch.nn.functional as F
from diffusers.optimization import get_cosine_schedule_with_warmup
from scaled.pipelines.pipline_ddpm_scaled_sfc import DDPMPipeline
import math
import os
from accelerate import Accelerator
from huggingface_hub import HfFolder, Repository, whoami
from tqdm.auto import tqdm
from pathlib import Path
import os
import matplotlib.gridspec as gridspec
import numpy as np

import random

def visualize_with_diff(data_pre, data_gt, data_ori, filename):
    print("data_gt shape: ", data_gt.shape)
    data_gt = data_gt.reshape((2, 257,80))
    data_pre = data_pre.reshape((2, 257,80))
    data_ori = data_ori.reshape((2, 257,80))
    # Create a figure with a larger size and higher resolution
    fig = plt.figure(figsize=(12, 12), dpi=300)  # Adjust figsize to accommodate diff images
    gs = gridspec.GridSpec(3, 3, width_ratios=[1, 1, 1])  # Add extra row for diff and column for colorbars
    
    for i in range(2):
        ax1 = plt.subplot(gs[i, 0])
        im1 = ax1.imshow(data_pre[i], vmin=-0.5, vmax=0.5)
        ax1.set_title('Prediction')
        ax1.axis('off')
        
        ax2 = plt.subplot(gs[i, 1])
        im2 = ax2.imshow(data_gt[i], vmin=-0.5, vmax=0.5)
        ax2.set_title('Ground Truth')
        ax2.axis('off')
        
        ax3 = plt.subplot(gs[i, 2])
        im3 = ax3.imshow(data_ori[i], vmin=-0.5, vmax=0.5)
        ax3.set_title('Original Data')
        ax3.axis('off')
    
    # Now plot the differences in the 4th row
    diff_pre_gt = np.abs(data_pre - data_gt)  # Difference between Prediction and Ground Truth
    diff_pre_ori = np.abs(data_pre - data_ori)  # Difference between Prediction and Original
    diff_gt_ori = np.abs(data_gt - data_ori)  # Difference between Ground Truth and Original
    
    ax4 = plt.subplot(gs[2, 0])
    im4 = ax4.imshow(diff_pre_gt[0], vmin=0, vmax=0.1)
    ax4.set_title('Diff Pre-GT')
    ax4.axis('off')
    
    ax5 = plt.subplot(gs[2, 1])
    im5 = ax5.imshow(diff_pre_ori[0], vmin=0, vmax=0.1)
    ax5.set_title('Diff Pre-Ori')
    ax5.axis('off')
    
    ax6 = plt.subplot(gs[2, 2])
    im6 = ax6.imshow(diff_gt_ori[0], vmin=0, vmax=0.1)
    ax6.set_title('Diff GT-Ori')
    ax6.axis('off')
    
    plt.tight_layout()
    
    # Save the figure with higher resolution
    plt.savefig(filename)
    plt.close(fig)


@dataclass
class TrainingConfig:
    image_size = 128  # the generated image resolution
    train_batch_size = 16
    eval_batch_size = 16  # how many images to sample during evaluation
    num_epochs = 1000
    gradient_accumulation_steps = 1
    learning_rate = 1e-5
    lr_warmup_steps = 500
    save_image_epochs = 1
    mixed_precision = "fp16"  # `no` for float32, `fp16` for automatic mixed precision
    output_dir = "ddpm-butterflies-128"  # the model name locally and on the HF Hub
    push_to_hub = True  # whether to upload the saved model to the HF Hub
    hub_private_repo = False
    overwrite_output_dir = True  # overwrite the old model when re-running the notebook
    seed = 0


config = TrainingConfig()

from scaled.dataset.sfc_dataset import SFCDiffusionDataset

train_dataset = SFCDiffusionDataset(
        data_dir="data/SFC/SFC_data_csv",
        data_list=[i for i in range(5, 3500)])
val_dataset  = SFCDiffusionDataset(
    data_dir="data/SFC/SFC_data_csv",
    data_list=[i for i in range(3500,3990)])

import matplotlib.pyplot as plt
import torch

train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=config.train_batch_size, shuffle=True)

model = UNet1DsModel(
        in_channels=4,
        out_channels=2,
        down_block_types=("DownBlock1D", "DownBlock1D", "DownBlock1D", "DownBlock1D"),
        up_block_types=("UpBlock1D", "UpBlock1D", "UpBlock1D", "UpBlock1D"),
        block_out_channels=(128, 256, 384, 512),
        add_attention=False,
        )


noise_schedular = DDPMScheduler(num_train_timesteps=1000)


optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
lr_scheduler = get_cosine_schedule_with_warmup(
    optimizer=optimizer,
    num_warmup_steps=config.lr_warmup_steps,
    num_training_steps=(len(train_dataloader) * config.num_epochs),
)

def evaluate(config, epoch, pipeline,dataset):
    # Sample some images from random noise (this is the backward diffusion process).
    # The default pipeline output type is `List[PIL.Image]`
    results = {}
    sample_idx = [random.randint(0, len(dataset)) for _ in range(1)]
    pre,nex = dataset[sample_idx[0]]
    pre = pre.to(pipeline.device).unsqueeze(0)
    nex = nex.to(pipeline.device).unsqueeze(0)
    images = pipeline(
        pre,
        batch_size=1,
        generator=torch.manual_seed(config.seed),
    ).images
    results['WithoutBackground'] = {
        "prediction_flow": images.detach().cpu().numpy()[0],
        "gt_flow": nex.detach().cpu().numpy()[0],
        "original_flow": pre.detach().cpu().numpy()[0]
    }
    os.makedirs('exp_out',exist_ok=True)
    visualize_with_diff(images.detach().cpu().numpy()[0],nex.detach().cpu().numpy()[0],pre.detach().cpu().numpy()[0],f'exp_out/scaled_ddpm_sfc_{epoch}.png')
    

def train_loop(config, model, noise_scheduler, optimizer, train_dataloader, lr_scheduler):

    # Initialize accelerator and tensorboard logging
    accelerator = Accelerator(
        mixed_precision=config.mixed_precision,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        log_with="tensorboard",
        project_dir=os.path.join(config.output_dir, "logs"),
    )
    if accelerator.is_main_process:
        if config.output_dir is not None:
            os.makedirs(config.output_dir, exist_ok=True)
        accelerator.init_trackers("train_example")

    model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, lr_scheduler
    )

    global_step = 0
    

    if accelerator.is_main_process:
        pipeline = DDPMPipeline(unet=accelerator.unwrap_model(model), scheduler=noise_scheduler)
        pipeline = pipeline.to(accelerator.device)
        evaluate(config, 0, pipeline,val_dataset)

    # Now you train the model
    for epoch in range(config.num_epochs):
        progress_bar = tqdm(total=len(train_dataloader), disable=not accelerator.is_local_main_process)
        progress_bar.set_description(f"Epoch {epoch}")

        for step, batch in enumerate(train_dataloader):
            pre,nex = batch
            # Sample noise to add to the images
            noise = torch.randn(pre.shape).to(pre.device)
            bs = pre.shape[0]

            # Sample a random timestep for each image
            timesteps = torch.randint(
                0, noise_scheduler.config.num_train_timesteps, (bs,), device=pre.device
            ).long()
            # Add noise to the clean images according to the noise magnitude at each timestep
            # (this is the forward diffusion process)
            noisy_latent = noise_scheduler.add_noise(nex, noise, timesteps)
            # import pdb; pdb.set_trace()
            input_latent = torch.cat([pre,noisy_latent],dim=1)

            with accelerator.accumulate(model):
                # Predict the noise residual
                noise_pred = model(input_latent, timesteps, return_dict=False)[0]
                loss = F.mse_loss(noise_pred, noise)
                accelerator.backward(loss)
                accelerator.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            progress_bar.update(1)
            logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0], "step": global_step}
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)
            global_step += 1

        # After each epoch you optionally sample some demo images with evaluate() and save the model
        if accelerator.is_main_process:
            pipeline = DDPMPipeline(unet=accelerator.unwrap_model(model), scheduler=noise_scheduler)

            if (epoch + 1) % config.save_image_epochs == 0 or epoch == config.num_epochs - 1:
                evaluate(config, epoch, pipeline,val_dataset)

train_loop(config, model,noise_schedular, optimizer, train_dataloader, lr_scheduler)