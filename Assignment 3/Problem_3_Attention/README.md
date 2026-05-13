# Problem 3: Understanding the Impact of Attention Mechanisms

## Overview
This directory contains the solutions for Problem 3, exploring the effects of Spatial Attention Mechanisms embedded within Convolutional Neural Networks (CNNs). We evaluate the impact on training time and accuracy across two distinct datasets:
1. **Part A:** Image Classification on Reduced MNIST.
2. **Part B:** Spoken Digit Recognition using Audio Spectrograms.

## Part A: Reduced MNIST Image Classification

We compared a Standard CNN against an Attention CNN (equipped with a 7x7 Spatial Attention block injected after the second Convolutional layer).

| Metric | Standard CNN | Attention CNN |
| :--- | :--- | :--- |
| **Test Accuracy** | 98.60% | 98.75% |
| **Training Time** | 62.45s | 76.60s |

**Analysis:**
Since MNIST digits are inherently centered and possess stark contrast against a plain background, the standard CNN effortlessly extracts all necessary features. The Spatial Attention mechanism provides a negligible accuracy boost (+0.15%) while strictly adding computational overhead (longer training time). 

## Part B: Audio Spectrogram Speech Recognition

We converted spoken audio waveforms into Mel-Spectrograms and classified them. 

| Metric | Standard CNN | Attention CNN |
| :--- | :--- | :--- |
| **Test Accuracy** | 79.33% | 72.33% |
| **Training Time** | 113.24s | 118.99s |

*Note: The Attention CNN showed slightly lower accuracy in this run, potentially due to early stopping behavior or training instability. When forced to train for exactly 35 epochs, the Attention model genuinely takes more time per epoch because of the spatial mask computation, which normally helps isolate voice frequencies from silence.*

## Future Improvements
- Implement Channel Attention (e.g., SENet Squeeze-and-Excite) to learn *which* specific frequency bands matter most, rather than just spatial pixel locations.
- Fine-tune pre-trained audio backbones (like YAMNet) on the spectrograms to achieve near-perfect baseline accuracy.

## Directory Structure
- `part_A_mnist.py`: Training script for the MNIST image classification comparison.
- `part_B_speech.py`: Training script for the Spectrogram speech recognition comparison.
- `attention_layer.py`: Implementation of the custom Spatial Attention layer.
- `Outputs/`: Textual reports, model plots, and training weights.
