# Environment
To run the image generation and evaluation, you can run the following codes:
```bash
conda env create -f flux.yaml
conda activate flux_dev
```
You can directly download the Flux model from the following link:
* FLUX.1-dev: https://huggingface.co/black-forest-labs/FLUX.1-dev

# Method
The procedure for tracking Gt(c) follows that of SDXL: we compute the difference between the conditional and unconditional model outputs at each step, flatten the resulting tensor into a vector, and compute its ℓ2-norm. We further scale this value by σt at each step t, consistent with the noise estimation mechanism in the sampling scheduler.

As FLUX is trained with guidance distillation, practical deployment requires an additional model evaluation with ω = 1 to obtain the hidden outputs of the conditional and unconditional branches. The score difference is then tracked using these outputs.
```bash
with torch.no_grad():
    noise_pred1 = opt_model(
        hidden_states=latents,
        timestep=timestep / 1000,
        guidance=guidance1,
        pooled_projections=pooled_prompt_embeds,
        encoder_hidden_states=prompt_embeds,
        txt_ids=text_ids,
        img_ids=latent_image_ids,
        joint_attention_kwargs={},
        return_dict=False,
    )[0]

    noise_pred2 = opt_model(
        hidden_states=latents,
        timestep=timestep / 1000,
        guidance=guidance2,
        pooled_projections=pooled_prompt_embeds,
        encoder_hidden_states=prompt_embeds,
        txt_ids=text_ids,
        img_ids=latent_image_ids,
        joint_attention_kwargs={},
        return_dict=False,
    )[0]

noise_gap_origion = (noise_pred1 - noise_pred2)/guidance1[0]
```