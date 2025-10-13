# VQVAE Project Instructions

This project implements and compares Variational Autoencoder (VAE), Vector Quantized VAE (VQVAE), Custom VQVAE, and Finite Scalar Quantization (FSQ) models on the STL10 dataset. Please note that this lab is designed to teach about VAEs, VQ-VAEs, and other quantization techniques. It may be slightly different or use different techniques than the original VQ-VAE paper due to stability and simplicity considerations.

## Prerequisites
- This lab comes with just the necessary code structure and some parts of the implementation. You will need to fill in the missing parts as indicated in the code files.
- To get started navigate to the project directory and run `uv run src/main.py --data` on the login node initially just to download the dataset and modules.
- You are welcome to complete this lab wherever you like (e.g., your local machine, Google Colab, or the ORC cluster).
- To run on the ORC cluster, you can submit jobs by running: `sbatch submit.sh`. Make sure to adjust the script for the different experiments if necessary.

## Step-by-Step Instructions

### 1. VAE
1. Implement the forward pass in `src/vae.py` as per the instructions in the file which is just the ONE TODO section.
2. Execute the VAE model to train and evaluate it.
```bash
uv run src/main.py --exp vae
```
This will train the VAE for 12 epochs and generate loss curves and reconstruction images. The output should be blurry but recognizable.

### 2. Run VQVAE Experiment
1. Implement the forward pass in `src/vqvae.py` as per the instructions in the file which is just the ONE TODO section.
2. Next, execute the standard VQVAE model.
```bash
uv run src/main.py --exp vqvae
```
This trains the VQVAE with a codebook size of 512 and generates the outputs which should be sharper than the VAE reconstructions.

### 3. Run Custom VQVAE Experiment
1. Implement the forward pass in `src/my_vector_quantizer.py` by following the instructions in the file. This will involve completing FOUR TODO sections: `calculate_distances`, `find_quantized_latents`, `normalize_with_ema`, and `apply_ste`.
2. Run the custom VQVAE implementation (using `MyVectorQuantizer`).
```bash
uv run src/main.py --exp myvqvae
```
This uses your custom quantization layer for training and evaluation. The outputs should be similar to the standard VQVAE.

```
This trains FSQ with 16 levels and generates the outputs which should be comparable to the VQVAE results.

## Submission
Once all experiments are complete, submit the following:
1. All individual reconstruction images (e.g., `images/vae.png`, `images/vqvae.png`, `images/myvqvae.png`).
  - Ensure the images demonstrate the model's performance on test data.
  - You can run all experiments in sequence using:
  ```bash
  uv run src/main.py
  ```
2. Please also submit your code by downloading and submitting your three .py vae files (vae.py, vqvae.py, and my_vector_quantizer.py).
