# Environment
To run the image generation and evaluation, you can build the environment through the instructions from the following links:

* Geneval: https://github.com/djghosh13/geneval
* DPG-Bench: https://github.com/TencentQQGYLab/ELLA

You can download the diffusion models by codes or directly from the following links:

* SDv1.5: https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5
* SDXL: https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0

# Method
## ASD Tracking
CFG-Rejection integrates seamlessly into existing generation pipelines with minimal computational overhead. We only need to track the score difference for each denoisng steps.

For SDv1.5, we employ the default PNDM scheduler at a resolution of 512×512. The latent noise dimension is 4×64×64. To track the score difference, we compute the direct difference between the outputs of the conditional and unconditional models, flatten the resulting tensors into a vector, and measure its ℓ2-norm:
```bash
cur_gap = noise_pred_text - noise_pred_uncond
nv_gap_flatten = torch.flatten(cur_gap,1,-1)
nv_gap_norm = torch.norm(nv_gap_flatten,dim=-1)
```
You don't need to strictly follow our function organization (e.g., in `dpg_sd1.5.py` or `generate_sd1.5.py`). As long as the score difference is computed in the same way, the filtering will consistently yield high-quality samples.

For SDXL, we utilize the default Flow Match Euler Discrete scheduler with a resolution of 1024×1024. The latent noise dimension is 4×128×128. The procedure for tracking Gt(c) follows that of SDv1.5.

## Selection
For SDv1.5, we directly calculate the mean of score differences along the defined inference steps:
```bash
noise_mean = torch.mean(nv_classifier,dim=0)
noise_mean = noise_mean[begin_idx:begin_idx+total_num]
sorted_mean,idx_mean = torch.sort(noise_mean,descending=True)
```

As for SDXL, the key distinction is a scaling by σt at each step t due to the way noise estimation is incorporated in different sampling schedulers.
```bash
noise_gap = sigma_SDXL*nv_gap_origin.permute(1,0)
nv_classifier = noise_gap[:select_step]
```
You can run `sort_100_gen_sd1.5_guidance5.0.ipynb` or `sort_100_gen_sdxl_guidance5.0` to conduct the selection. One should replace the paths of images and score differences with their own.

# Evaluation
To get the ratings from Geneval, first evluate each images from each prompts. For example:
```bash
python evaluation/evaluate_images_final.py "D:\100_images_SDXL_guidance9.0" --outfile "D:\AAA_linux_scores\100_images_SDXL_guidance9.0\4_from_20_trial1_step40.jsonl" --model-path "object_detect" --sample_path "step40_4_from_20_trial1" --begin_idx 0
```
Then run `summary_scores.py` to get the overall scores. One should replace the input and output path with their own.
```bash
python evaluation/summary_scores.py  "D:\AAA_linux_scores\100_images_SDXL_guidance9.0\4_from_20_trial1_step40.jsonl"  
```

For DPG-bench evaluation, one can directly run "compute_dpg_bench.py".
```bash
 python compute_dpg_bench.py --csv "dpg_bench.csv" --resolution 1024 --image-root-path "D:\100_DPG_SDXL_guidance5.0\random_begin28" --res-path "D:\AAA_linux_scores\100_DPG_SDXL_guidance5.0\random_begin28.txt"
 ```