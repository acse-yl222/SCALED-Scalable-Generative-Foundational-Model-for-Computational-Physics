## import pakages
import argparse
import logging
import math
import os
import os.path as osp
import random
import time
import warnings
from collections import OrderedDict
from datetime import datetime
import diffusers
import torch
import torch.nn as nn
import torch.nn.functional as F
from scaled.dataset.urban_flow_dataset import UrbanFlowSplitDataset
import torch.utils.checkpoint
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import DistributedDataParallelKwargs
from diffusers import DDIMScheduler
from diffusers.utils import check_min_version
from diffusers.utils.import_utils import is_xformers_available
from omegaconf import OmegaConf
from scaled.model.unets.unet_3ds import UNet3DsModel
from tqdm.auto import tqdm
from scaled.utils.util import import_filename,seed_everything
import sys
import os
from trainning_validation import log_validation,visualize_with_diff,logger,compute_snr,Net

# initialize the enviroment
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../scaled')))
warnings.filterwarnings("ignore")


def main(cfg):

    # Initialize the accelerator. We will let the accelerator handle everything for us.
    kwargs = DistributedDataParallelKwargs(find_unused_parameters=False)
    accelerator = Accelerator(
        gradient_accumulation_steps=cfg.solver.gradient_accumulation_steps,
        mixed_precision=cfg.solver.mixed_precision,
        kwargs_handlers=[kwargs],
    )

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()
    if cfg.seed is not None:
        seed_everything(cfg.seed)

    # setup basic experiments
    exp_name = cfg.exp_name
    save_dir = f"{cfg.output_dir}/{exp_name}"
    if accelerator.is_main_process:
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
    sample_dir = os.path.join(save_dir, 'samples')
    if accelerator.is_main_process and not os.path.exists(sample_dir):
        os.makedirs(sample_dir)

    if cfg.weight_dtype == "fp16":
        weight_dtype = torch.float16
    elif cfg.weight_dtype == "fp32":
        weight_dtype = torch.float32
    else:
        raise ValueError(
            f"Do not support weight dtype: {cfg.weight_dtype} during training"
        )

    sched_kwargs = OmegaConf.to_container(cfg.noise_scheduler_kwargs)
    if cfg.enable_zero_snr:
        sched_kwargs.update(
            rescale_betas_zero_snr=True,
            timestep_spacing="trailing",
            prediction_type=cfg.prediction_type,
        )
    sched_kwargs.update({"beta_schedule": "scaled_linear"})
    train_noise_scheduler = DDIMScheduler(**sched_kwargs)

    # Initialize the model
    denoising_unet  = UNet3DsModel(in_channels=9,
                                  out_channels=3,
                                  down_block_types=("DownBlock3D", "DownBlock3D", "DownBlock3D", "DownBlock3D"),
                                  up_block_types=("UpBlock3D", "UpBlock3D", "UpBlock3D", "UpBlock3D"),
                                  block_out_channels=(64, 128, 192, 256),
                                  add_attention=False,
                                  )
    net = Net(denoising_unet)
    denoising_unet.requires_grad_(True)

    if cfg.solver.enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            denoising_unet.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError(
                "xformers is not available. Make sure it is installed correctly"
            )

    if cfg.solver.gradient_checkpointing:
        denoising_unet.enable_gradient_checkpointing()

    # Initialize the learning rate
    if cfg.solver.scale_lr:
        learning_rate = (
                cfg.solver.learning_rate
                * cfg.solver.gradient_accumulation_steps
                * cfg.train_bs
                * accelerator.num_processes
        )
    else:
        learning_rate = cfg.solver.learning_rate

    # Initialize the optimizer
    if cfg.solver.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError(
                "Please install bitsandbytes to use 8-bit Adam. You can do so by running `pip install bitsandbytes`"
            )

        optimizer_cls = bnb.optim.AdamW8bit
    else:
        optimizer_cls = torch.optim.AdamW
    trainable_params = list(filter(lambda p: p.requires_grad, net.parameters()))
    logger.info(f"Total trainable params {len(trainable_params)}")
    optimizer = optimizer_cls(
        trainable_params,
        lr=learning_rate,
        betas=(cfg.solver.adam_beta1, cfg.solver.adam_beta2),
        weight_decay=cfg.solver.adam_weight_decay,
        eps=cfg.solver.adam_epsilon,
    )

    # Load the dataset
    train_dataset = UrbanFlowSplitDataset(
        data_dir=cfg.dataset_path,
        rotato_ratio=0.8,
        stride=4,
        skip_timestep=cfg.skip_timestep,
        subdomain_size=32,
        time_steps_list=[i for i in range(0, 5000)])
    val_dataset  = UrbanFlowSplitDataset(
        data_dir=cfg.dataset_path,
        rotato_ratio=0,
        stride=16,
        subdomain_size=128,
        skip_timestep=cfg.skip_timestep,
        time_steps_list=[i for i in range(5000, 6000)])

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=cfg.train_bs,
        shuffle=True,
        num_workers=8,
        drop_last=True,
    )

    # Prepare everything with our `accelerator`.
    (
        net,
        optimizer,
        train_dataloader,
    ) = accelerator.prepare(
        net,
        optimizer,
        train_dataloader,
    )

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / cfg.solver.gradient_accumulation_steps
    )
    # Afterwards we recalculate our number of training epochs
    num_train_epochs = math.ceil(
        cfg.solver.max_train_steps / num_update_steps_per_epoch
    )

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        run_time = datetime.now().strftime("%Y%m%d-%H%M")
        accelerator.init_trackers(
            exp_name,
        )

    # Train!
    total_batch_size = (
            cfg.train_bs
            * accelerator.num_processes
            * cfg.solver.gradient_accumulation_steps
    )

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {cfg.train_bs}")
    logger.info(
        f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}"
    )
    logger.info(
        f"  Gradient Accumulation steps = {cfg.solver.gradient_accumulation_steps}"
    )
    logger.info(f"  Total optimization steps = {cfg.solver.max_train_steps}")
    global_step = 0
    first_epoch = 0

    # Potentially load in the weights and states from a previous save
    if cfg.resume_from_checkpoint:
        if cfg.resume_from_checkpoint != "latest":
            resume_dir = cfg.resume_from_checkpoint
        else:
            resume_dir = save_dir
        # Get the most recent checkpoint
        dirs_ = os.listdir(resume_dir)
        dirs = [d for d in dirs_ if d.startswith("denoising_unet")]
        dirs = sorted(dirs, key=lambda x: int(x.split("-")[1][:-4]))
        path = dirs[-1]
        weight = torch.load(os.path.join(resume_dir, path),map_location='cpu')
        denoising_unet.load_state_dict(weight, strict=False)
        accelerator.print(f"Resuming from checkpoint {path}")
        global_step = int(path.split("-")[1][:-4])

        first_epoch = global_step // num_update_steps_per_epoch
        resume_step = global_step % num_update_steps_per_epoch

    # Only show the progress bar once on each machine.
    progress_bar = tqdm(
        range(cfg.solver.max_train_steps),
        disable=not accelerator.is_local_main_process,
    )
    progress_bar.set_description("Steps")
    progress_bar.update(global_step)

    for epoch in range(first_epoch, num_train_epochs):
        train_loss = 0.0
        t_data_start = time.time()
        for step, batch in enumerate(train_dataloader):
            t_data = time.time() - t_data_start
            with accelerator.accumulate(net):
                data_0 = batch[0].to(weight_dtype)
                data_1 = batch[1].to(weight_dtype)
                control_value = data_0[:,3:].clone()
                data_0 = data_0[:,:3]
                data_1 = data_1[:,:3]

                # change the background data for trainning
                background_value = control_value.clone().bool()
                halo_cell  = random.choice([4,6,8,12])
                choice = random.random()
                if choice<0:
                    back_data = data_1.clone()
                    back_data[:, :, 1:-1, halo_cell:-halo_cell, halo_cell:-halo_cell] = 1
                    back_data[:,0:1][background_value] = 0
                    back_data[:,1:2][background_value] = 0
                    back_data[:,2:3][background_value] = 0
                else:
                    back_data = data_1.clone()
                    back_data[:, :, 1:-1] = 1
                    back_data[:, 0:1][background_value] = 0
                    back_data[:, 1:2][background_value] = 0
                    back_data[:, 2:3][background_value] = 0
                noise = torch.randn_like(data_1)

                if cfg.noise_offset > 0:
                    noise += cfg.noise_offset * torch.randn(
                        (data_0.shape[0], data_0.shape[1], 1, 1, 1),
                        device=data_0.device,
                    )
                bsz = data_0.shape[0]
                # Sample a random timestep for each video
                timesteps = torch.randint(
                    0,
                    train_noise_scheduler.num_train_timesteps,
                    (bsz,),
                    device=data_0.device,
                )
                timesteps = timesteps.long()
                noisy_latents = train_noise_scheduler.add_noise(data_1, noise, timesteps)
                input_latent = torch.cat([data_0, back_data,noisy_latents], dim=1)  # [B, 2+2, D//8,  H//8, W//8]
                if train_noise_scheduler.prediction_type == "epsilon":
                    target = noise
                elif train_noise_scheduler.prediction_type == "v_prediction":
                    target = train_noise_scheduler.get_velocity(
                        data_1, noise, timesteps
                    )
                elif train_noise_scheduler.prediction_type == "sample":
                    target = data_1
                else:
                    raise ValueError(
                        f"Unknown prediction type {train_noise_scheduler.prediction_type}"
                    )

                # ---- Forward!!! -----
                model_pred = net(
                    input_latent,
                    timesteps
                )
                
                weight = [1+max(0, 3* (25-i)/24) for i in range(1,64)]
                weight = [1]+weight
                weight = torch.tensor(weight).to(accelerator.device)
                weight = weight.view(1, 1, 64, 1, 1)

                if cfg.snr_gamma == 0:
                    loss = F.l1_loss(
                        model_pred.float()*weight, target.float()*weight, reduction="mean"
                    )
                else:
                    snr = compute_snr(train_noise_scheduler, timesteps)
                    if train_noise_scheduler.config.prediction_type == "v_prediction":
                        # Velocity objective requires that we add one to SNR values before we divide by them.
                        snr = snr + 1
                    mse_loss_weights = (
                            torch.stack(
                                [snr, cfg.snr_gamma * torch.ones_like(timesteps)], dim=1
                            ).min(dim=1)[0]
                            / snr
                    )
                    loss = F.l1_loss(
                        model_pred.float(), target.float(), reduction="none"
                    )
                    loss = (
                            loss.mean(dim=list(range(1, len(loss.shape))))
                            * mse_loss_weights
                    )
                    loss = loss.mean()

                # Gather the losses across all processes for logging (if we use distributed training).
                avg_loss = accelerator.gather(loss.repeat(cfg.train_bs)).mean()
                train_loss += avg_loss.item() / cfg.solver.gradient_accumulation_steps

                # Backpropagate
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(
                        trainable_params,
                        cfg.solver.max_grad_norm,
                    )
                optimizer.step()
                optimizer.zero_grad()

            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
                accelerator.log({"train_loss": train_loss}, step=global_step)
                train_loss = 0.0

                if (global_step % cfg.val.validation_steps == 0) or (global_step in cfg.val.validation_steps_tuple):
                    if accelerator.is_main_process:
                        generator = torch.Generator(device=accelerator.device)
                        generator.manual_seed(cfg.seed)
                        unwrap_net = accelerator.unwrap_model(net)
                        save_checkpoint(
                            unwrap_net.denoising_unet,
                            save_dir,
                            "denoising_unet",
                            global_step,
                            total_limit=4,
                        )
                        ori_net = accelerator.unwrap_model(net)
                        results = log_validation(ori_net.denoising_unet,train_noise_scheduler, accelerator, generator,val_dataset)
                        sample_indexs = list(results.keys())
                        for index in range(len(sample_indexs)):
                            index_sample = sample_indexs[index]
                            path = os.path.join(sample_dir, f"{global_step}_{index_sample}_pre.png")
                            visualize_with_diff(results[index_sample]['prediction_flow'],results[index_sample]['gt_flow'],results[index_sample]['original_flow'],path)

            logs = {
                "step_loss": loss.detach().item(),
                "lr": learning_rate,
                "td": f"{t_data:.2f}s",
            }
            t_data_start = time.time()
            progress_bar.set_postfix(**logs)
            if global_step >= cfg.solver.max_train_steps:
                break
        # save model after each epoch
        # Create the pipeline using the trained modules and save it.
        accelerator.wait_for_everyone()


def save_checkpoint(model, save_dir, prefix, ckpt_num, total_limit=None):
    save_path = osp.join(save_dir, f"{prefix}-{ckpt_num}.pth")

    if total_limit is not None:
        checkpoints = os.listdir(save_dir)
        checkpoints = [d for d in checkpoints if d.startswith(prefix)]
        checkpoints = sorted(
            checkpoints, key=lambda x: int(x.split("-")[1].split(".")[0])
        )

        if len(checkpoints) >= total_limit:
            num_to_remove = len(checkpoints) - total_limit + 1
            removing_checkpoints = checkpoints[0:num_to_remove]
            logger.info(
                f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
            )
            logger.info(f"removing checkpoints: {', '.join(removing_checkpoints)}")
            for removing_checkpoint in removing_checkpoints:
                removing_checkpoint = os.path.join(save_dir, removing_checkpoint)
                os.remove(removing_checkpoint)

    mm_state_dict = OrderedDict()
    state_dict = model.state_dict()
    for key in state_dict:
        mm_state_dict[key] = state_dict[key]
    torch.save(mm_state_dict, save_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="./config/trainning_stage1.yaml")
    args = parser.parse_args()

    if args.config[-5:] == ".yaml":
        config = OmegaConf.load(args.config)
    elif args.config[-3:] == ".py":
        config = import_filename(args.config).cfg
    else:
        raise ValueError("Do not support this format config file")
    main(config)