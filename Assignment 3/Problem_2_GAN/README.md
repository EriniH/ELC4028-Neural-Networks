# Problem 2: GAN Synthetic Data with Low-Data Stabilization

## Overview
This directory contains the solution for Problem 2, which involves training a Conditional Generative Adversarial Network (cDCGAN) on a limited dataset of 350 real examples per digit. The goal is to evaluate different confidence-based selection strategies for the GAN-generated data and compare its performance against the baseline.

## Methodology
1. **Baseline Augmentation:** Similar to the VAE approach, the 350 real images were heavily augmented before training.
2. **GAN Training:** A cDCGAN (Generator + Discriminator) was trained on the augmented data.
3. **Data Generation:** 50,000 synthetic images were generated across all 10 digits.
4. **Confidence-Based Filtering:** The LeNet-5 "Judge" (trained on the 350 real images) evaluated the 50k synthetic images to categorize them based on Confidence Level (CL).

## Datasets and Filtering Results
- **Set A (All Generated):** 50,000 images
- **Set B (High Confidence: CL >= 0.9):** 27,252 images
- **Set C (Mid Confidence: 0.6 <= CL <= 0.9):** 14,803 images

## Performance Benchmark

| Model | Selection Strategy | Set Size | Test Accuracy |
| :--- | :--- | :--- | :--- |
| **LeNet-5 Baseline** | 350-Real Baseline | 3,500 | 95.85% |
| **LeNet-5 Baseline** | 1000-Real Baseline | 10,000 | 97.50% |
| **GAN + LeNet-5** | Set A (All Data) | 50,000 | 92.20% |
| **GAN + LeNet-5** | Set B (CL >= 0.9) | 27,252 | 92.65% |
| **GAN + LeNet-5** | Set C (0.6 <= CL <= 0.9)| 14,803 | 93.45% |

## Conclusion
- **Which selection strategy is best?** 
  Set C (Mid Confidence) provided the best improvement (93.45%). These edge-case images act as effective regularizers for the model, pushing it to learn better decision boundaries rather than overfitting on "easy" Set B images.
- **Does it reduce the need for real data?** 
  Partially, but not completely. While GAN-generated data improved the model compared to pure fake data or zero augmentation, it could not beat the pure 1000-Real data baseline (97.50%). Real data remains superior.

## Directory Structure
- `main_prob2.py`: The evaluation, filtering, and benchmarking pipeline.
- `models_gan.py`: Definition of LeNet-5 architecture.
- `gan_generator.py`: Definition of the Conditional GAN architecture.
- `data_loader_gan.py`: Loading and parsing logic.
- `Outputs/`: Filtered datasets, evaluation reports, and model weights.
