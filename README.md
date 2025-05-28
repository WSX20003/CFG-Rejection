# Diffusion Sampling Path Tells More: An EfficientPlug-and-Play Strategy for Sample Filtering
This is the official implementation for the paper "Diffusion Sampling Path Tells More: An Efficient Plug-and-Play Strategy for Sample Filtering". 
We propose the CFG-Rejection method to filter low-quality samples at an early stage of the denoising process, utilizing the the cumulative divergence between conditional and unconditional scores, which we define as Accumulated Score Differences (ASD).

![qualitative comparison](images/prompt1_bottom.png)
![qualitative comparison](images/prompt1_top.png)

Above is a direct comparision which provides a visual demonstration to give an early glimpse into the remarkable effectiveness of the ASD metrics in distinguishing the distribution of good and bad samples. This is a visual text rendering for the prompt "A night sky with constellations forming the words ’Among the stars, we find our dreams and destiny’". Low-ASD images (top row) exhibit completely missing strokes, while high-ASD samples (bottom row) ensure textual requirements.