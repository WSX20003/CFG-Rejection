import argparse
import json
import os

import torch
import numpy as np
from PIL import Image
from tqdm import tqdm, trange
from einops import rearrange
from torchvision.utils import make_grid
from torchvision.transforms import ToTensor
from pytorch_lightning import seed_everything
from diffusers import DiffusionPipeline, StableDiffusionPipeline

import inspect

torch.set_grad_enabled(False)

def retrieve_timesteps(
    scheduler,
    num_inference_steps = None,
    device = None,
    timesteps = None,
    **kwargs,
):

    if timesteps is not None:
        accepts_timesteps = "timesteps" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accepts_timesteps:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" timestep schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(timesteps=timesteps, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    else:
        scheduler.set_timesteps(num_inference_steps, device=device, **kwargs)
        timesteps = scheduler.timesteps
    return timesteps, num_inference_steps

def get_embds(pipe,prompt,device,num_images_per_prompt):
    prompt_embeds, negative_prompt_embeds = pipe.encode_prompt(
        prompt,
        device,
        num_images_per_prompt,
        do_classifier_free_guidance=True,
        negative_prompt=None,
        prompt_embeds=None,
        negative_prompt_embeds=None,
        lora_scale=None,
        clip_skip=None,
    )
    torch.cuda.empty_cache()    
    return prompt_embeds, negative_prompt_embeds

model_name =  "/120090444/hugging_face/SD1.5"
model = StableDiffusionPipeline.from_pretrained(model_name, torch_dtype=torch.float16)
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model = model.to(device)
model.enable_attention_slicing()

seed = 42
outdir = "120090444/100_DPG_SDv1-5_guidance5.0"
num_images_per_prompt = 20
height = model.unet.config.sample_size * model.vae_scale_factor
width = model.unet.config.sample_size * model.vae_scale_factor
guidance_scale = 5.0
num_inference_steps = 50
device = model._execution_device
total_images = 100

prompt_path = "/120090444/dpg_bench/prompts"
for filename in tqdm(os.listdir(prompt_path)):
    if filename.endswith('.txt'):
        file_path = os.path.join(prompt_path, filename)
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()
    name_without_extension = os.path.splitext(filename)[0] 
    prompt = content
    print(prompt)

    seed_everything(seed)
    outpath = os.path.join(outdir, name_without_extension)
    os.makedirs(outpath, exist_ok=True)
    sample_path = os.path.join(outpath, "samples")
    os.makedirs(sample_path, exist_ok=True)

    seed_everything(seed)
    prompt_embeds, negative_prompt_embeds = get_embds(model,prompt,device,num_images_per_prompt)
    prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])        
    nv_store = []

    for num in range(int(total_images/num_images_per_prompt)):
        nv_gap = []
        timesteps, num_inference_steps = retrieve_timesteps(model.scheduler, num_inference_steps, device, timesteps=None)
        num_channels_latents = model.unet.config.in_channels
        latents = model.prepare_latents(
            num_images_per_prompt,
            num_channels_latents,
            height,
            width,
            prompt_embeds.dtype,
            device,
            generator=None,
            latents=None,
        )

        for i, t in enumerate(timesteps):
            latent_model_input = torch.cat([latents] * 2) 
            latent_model_input = model.scheduler.scale_model_input(latent_model_input, t)
            noise_pred = model.unet(
                latent_model_input,
                t,
                encoder_hidden_states=prompt_embeds,
                timestep_cond=None,
                cross_attention_kwargs=None,
                added_cond_kwargs=None,
                return_dict=False,
            )[0]

            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
            latents = model.scheduler.step(noise_pred, t, latents, return_dict=False)[0]

            cur_gap = noise_pred_text - noise_pred_uncond
            cur_gap = cur_gap.cpu()
            nv_gap_flatten = torch.flatten(cur_gap,1,-1)
            nv_gap_norm = torch.norm(nv_gap_flatten,dim=-1)
            nv_gap.append(nv_gap_norm)


        image = model.vae.decode(latents / model.vae.config.scaling_factor, return_dict=False, generator=None)[0]
        do_denormalize = [True] * image.shape[0]
        image = model.image_processor.postprocess(image, output_type="pil", do_denormalize=do_denormalize)
        nv_gap = torch.stack(nv_gap)
        nv_gap = nv_gap.permute(-1,0)
        nv_store.append(nv_gap)   

        for iter_idx in range(num_images_per_prompt):
            cur_idx = num*num_images_per_prompt + iter_idx
            image[iter_idx].save(os.path.join(sample_path,f"{cur_idx:05}.png"))
        del latents,latent_model_input,nv_gap,timesteps,image
        torch.cuda.empty_cache()   

    nv_store = torch.stack(nv_store)
    nv_store = nv_store.reshape(total_images,num_inference_steps+1)
    nv_path = os.path.join(outpath,"guidance{}.pth".format(guidance_scale))
    torch.save(nv_store,nv_path)
    del nv_store,prompt_embeds,negative_prompt_embeds
    torch.cuda.empty_cache()