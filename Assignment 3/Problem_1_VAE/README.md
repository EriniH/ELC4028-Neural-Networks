# Problem 1: VAE Synthetic Data with Low-Data Stabilization

## Overview
This directory contains the solution for Problem 1, which explores using a Conditional Variational Autoencoder (VAE) to generate synthetic data for low-data stabilization. 
The objective is to train a VAE using only 350 real examples per digit, generate 50,000 synthetic samples, and then use a "Judge" network to select the most useful synthetic samples for data augmentation.

## Methodology
1. **Baseline Augmentation:** The pure 350-real dataset was augmented using shifts, rotations, scaling, and noise to prepare it for the VAE training.
2. **VAE Training:** A conditional VAE was trained on the augmented dataset to learn the latent space representation of the digits.
3. **Data Generation:** 50,000 new synthetic samples (5,000 per digit) were generated using the trained VAE by sampling from the normal distribution `z ~ N(0, I)`.
4. **Filtering via LeNet-5 Judge:** A LeNet-5 model trained exclusively on the 350-real images acted as a judge. It predicted labels for the generated images, and its maximum softmax output was used as the "Confidence Level" (CL).
5. **Dataset Segregation:**
   - **Set A:** All 50,000 generated samples.
   - **Set B:** High-confidence samples (CL >= 0.9).
   - **Set C:** Mid-confidence samples (0.6 <= CL <= 0.9).

## Results
We evaluated the effectiveness of the generated datasets by appending them to the 350-real baseline and training new LeNet-5 models. 

| Model | Selection Strategy | Set Size | Test Accuracy |
| :--- | :--- | :--- | :--- |
| **LeNet-5 Baseline** | 350-Real Baseline | 3,500 | 96.85% |
| **LeNet-5 Baseline** | 1000-Real Baseline | 10,000 | 97.50% |
| **VAE + LeNet-5** | Set A (All Data) | 50,000 | 97.60% |
| **VAE + LeNet-5** | Set B (CL >= 0.9) | 31,289 | 97.75% |
| **VAE + LeNet-5** | Set C (0.6 <= CL <= 0.9)| 12,298 | 96.90% |

**Conclusion:** 
- **Which selection strategy is best?** Set B (High Confidence) gave the best accuracy (97.75%). High confidence data (>=0.9) acts as clean, high-quality reinforcement, preventing the model from learning blurry or confused VAE artifacts.
- **Does it reduce the need for real data?** YES. The VAE-augmented dataset (97.75%) successfully matched or outperformed the pure 1000-Real dataset (97.50%). This proves generative models can effectively synthesize missing data.

## Directory Structure
- `main_prob1.py`: Main execution script that orchestrates loading data, training the judge, filtering datasets, and benchmarking.
- `models.py`: Defines the LeNet-5 classification architecture.
- `vae_generator.py`: Defines the VAE model architecture and training loops.
- `data_loader.py`: Handles loading and parsing of the MNIST datasets.
- `Outputs/`: Contains the pre-trained weights, filtered arrays, and generated plots.
