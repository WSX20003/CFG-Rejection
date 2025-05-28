# Diffusion Sampling Path Tells More: An EfficientPlug-and-Play Strategy for Sample Filtering
This is the official implementation for the paper "Diffusion Sampling Path Tells More: An Efficient Plug-and-Play Strategy for Sample Filtering". 
We propose the CFG-Rejection method to filter low-quality samples at an early stage of the denoising process, utilizing the the cumulative divergence between conditional and unconditional scores, which we define as Accumulated Score Differences (ASD).
Below is a comparision which provides a visual demonstration to give an early glimpse into the remarkable effectiveness of the ASD metrics in distinguishing the distribution of good and bad samples.
![qualitative comparison](images/prompt1_bottom.png)
![qualitative comparison](images/prompt1_top.png)
