import inspect
from dataclasses import dataclass
from typing import Callable, List, Optional, Union
import numpy as np
import torch
from diffusers import DiffusionPipeline
from diffusers.schedulers import (
    DDIMScheduler,
    DPMSolverMultistepScheduler,
    EulerAncestralDiscreteScheduler,
    EulerDiscreteScheduler,
    LMSDiscreteScheduler,
    PNDMScheduler,
)
from diffusers.utils import BaseOutput, is_accelerate_available
from diffusers.utils.torch_utils import randn_tensor
from einops import rearrange


@dataclass
class Pose2ImagePipelineOutput(BaseOutput):
    images: Union[torch.Tensor, np.ndarray]


class SCALEDUrbanFlowPipeline(DiffusionPipeline):
    _optional_components = []
    def __init__(
            self,
            denoising_unet,
            scheduler: Union[
                DDIMScheduler,
                PNDMScheduler,
                LMSDiscreteScheduler,
                EulerDiscreteScheduler,
                EulerAncestralDiscreteScheduler,
                DPMSolverMultistepScheduler,
            ],
    ):
        super().__init__()
        self.register_modules(
            denoising_unet=denoising_unet,
            scheduler=scheduler,
        )


    def enable_sequential_cpu_offload(self, gpu_id=0):
        if is_accelerate_available():
            from accelerate import cpu_offload
        else:
            raise ImportError("Please install accelerate via `pip install accelerate`")

        device = torch.device(f"cuda:{gpu_id}")

        for cpu_offloaded_model in [self.unet, self.text_encoder, self.vae]:
            if cpu_offloaded_model is not None:
                cpu_offload(cpu_offloaded_model, device)

    @property
    def _execution_device(self):
        if self.device != torch.device("meta") or not hasattr(self.unet, "_hf_hook"):
            return self.device
        for module in self.unet.modules():
            if (
                    hasattr(module, "_hf_hook")
                    and hasattr(module._hf_hook, "execution_device")
                    and module._hf_hook.execution_device is not None
            ):
                return torch.device(module._hf_hook.execution_device)
        return self.device

    def prepare_extra_step_kwargs(self, generator, eta):
        # prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
        # eta (η) is only used with the DDIMScheduler, it will be ignored for other schedulers.
        # eta corresponds to η in DDIM paper: https://arxiv.org/abs/2010.02502
        # and should be between [0, 1]

        accepts_eta = "eta" in set(
            inspect.signature(self.scheduler.step).parameters.keys()
        )
        extra_step_kwargs = {}
        if accepts_eta:
            extra_step_kwargs["eta"] = eta

        # check if the scheduler accepts generator
        accepts_generator = "generator" in set(
            inspect.signature(self.scheduler.step).parameters.keys()
        )
        if accepts_generator:
            extra_step_kwargs["generator"] = generator
        return extra_step_kwargs

    def prepare_latents(
            self,
            batch_size,
            num_channels_latents,
            depth,
            width,
            height,
            dtype,
            device,
            generator,
            latents=None,
    ):
        shape = (
            batch_size,
            num_channels_latents,
            depth,
            height,
            width,
        )
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        if latents is None:
            latents = randn_tensor(
                shape, generator=generator, device=device, dtype=dtype
            )
        else:
            latents = latents.to(device)

        # scale the initial noise by the standard deviation required by the scheduler
        latents = latents * self.scheduler.init_noise_sigma
        return latents

    @torch.no_grad()
    def __call__(
            self,
            previous_flow_value,
            background_value,
            depth=8,
            width=8,
            height=8,
            num_inference_steps=50,
            guidance_scale=3.5,
            num_images_per_prompt=1,
            eta: float = 0.0,
            generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
            output_type: Optional[str] = "tensor",
            return_dict: bool = True,
            callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
            callback_steps: Optional[int] = 1,
            **kwargs,
    ):

        device = self._execution_device

        do_classifier_free_guidance = guidance_scale > 1.0

        # Prepare timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps
        batch_size = len(previous_flow_value)
        
        
        previous_flow_value = previous_flow_value.to(
            dtype=self.denoising_unet.dtype, device=self.denoising_unet.device
        )
        
        background_value = background_value.to(
            dtype=self.denoising_unet.dtype, device=self.denoising_unet.device
        )
        
        if do_classifier_free_guidance:
            negtive_previous_flow_value = torch.zeros_like(previous_flow_value)
            negtive_background_value = torch.zeros_like(background_value)
        num_channels_latents = self.denoising_unet.out_channels
        latents = self.prepare_latents(
            batch_size,
            num_channels_latents,
            depth,
            width,
            height,
            torch.float32,
            device,
            generator,
        )  # (bs, c, 1, h', w')

        # Prepare extra step kwargs.
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        # denoising loop
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                # 3.1 expand the latents if we are doing classifier free guidance
                sheduled_latent = self.scheduler.scale_model_input(
                    latents, t
                )
                latent_model_input = torch.cat([previous_flow_value,background_value,sheduled_latent],dim=1) # 1x9x32x32x32
                if do_classifier_free_guidance:
                    negtive_latent_model_input = torch.cat([negtive_previous_flow_value,negtive_background_value,sheduled_latent],dim=1) # 1x24x32x32x32
                    latent_model_input = torch.cat([negtive_latent_model_input,latent_model_input],dim=0) # 2x9x32x32x32
                # input_latent = torch.cat([latent_model_input, latent_particle_condition_input], dim=1)
                # import pdb; pdb.set_trace()
                noise_pred = self.denoising_unet(
                    latent_model_input,
                    t,
                )[0]

                # perform guidance
                if do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + guidance_scale * (
                            noise_pred_text - noise_pred_uncond
                    )

                # compute the previous noisy sample x_t -> x_t-1
                latents = self.scheduler.step(
                    noise_pred, t, latents, **extra_step_kwargs, return_dict=False
                )[0]

                # call the callback, if provided
                if i == len(timesteps) - 1 or (
                        (i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0
                ):
                    progress_bar.update()
                    if callback is not None and i % callback_steps == 0:
                        step_idx = i // getattr(self.scheduler, "order", 1)
                        callback(step_idx, t, latents)
        # Post-processing
        # next_timestep_velocity = self.decode_latents(latents)  # (b, c, d, h, w)
        return latents