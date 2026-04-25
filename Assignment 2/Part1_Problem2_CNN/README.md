# Problem 2 — Convolutional Neural Network (CNN) on ReducedMNIST

## 🎯 Objective

Train a **Convolutional Neural Network (CNN)** to classify handwritten digits (0–9) using the ReducedMNIST dataset **directly from raw images** — no manual feature extraction needed.

## 📋 Requirements

### Part (a) — Base CNN Architecture

Use a **LeNet-5-style** architecture as the starting point. You must adjust the network to work with **28×28 grayscale images** (the original LeNet-5 was designed for 32×32).

**Key hints from the assignment:**
- Adjust parameters to fit 28×28 images instead of 32×32
- Use **ReLU** activation function

**Typical LeNet-5 structure (to adapt):**
```
Input (28×28×1)
→ Conv Layer 1 (filters, kernel size, activation=ReLU)
→ Pooling Layer 1
→ Conv Layer 2 (filters, kernel size, activation=ReLU)
→ Pooling Layer 2
→ Flatten
→ Fully Connected Layer 1
→ Fully Connected Layer 2
→ Output Layer (10 classes, softmax)
```

### Part (b) — Hyperparameter Variations

Make **at least two variations** in the hyperparameters and analyze the effect on performance. Possible variations include:
- Number of filters in convolutional layers
- Activation function (e.g., ReLU vs. Tanh vs. LeakyReLU)
- Adding or removing layers
- Changing kernel sizes
- Adding dropout or batch normalization
- Changing the optimizer or learning rate

### Part (c) — Comparison with Assignment 1

Fill in the comprehensive comparison table that includes results from:
- **Assignment 1**: K-means clustering, SVM (linear & nonlinear) with DCT/PCA/AutoEncoder features
- **This assignment**: MLP (Problem 1) and CNN (this problem)

## 📊 Results to Collect

### CNN Variations Table

| Variation | Description | Accuracy (%) | Training Time (ms) | Testing Time (ms) |
|-----------|-------------|-------------|--------------------|--------------------|
| Variation 1 | (describe changes) | | | |
| Variation 2 | (describe changes) | | | |
| Variation 3 | (describe changes — optional) | | | |
| Variation 4 | (describe changes — optional) | | | |

> **Note**: The CNN does NOT require a separate feature extraction step — it learns features directly from images.

## 🔧 Steps to Complete

1. [ ] Load ReducedMNIST (raw 28×28 images, no feature extraction)
2. [ ] Build the base CNN model (adapted LeNet-5 for 28×28, ReLU activation)
3. [ ] Train the base model and record accuracy + timing
4. [ ] Create at least 2 variations by changing hyperparameters
5. [ ] Train each variation and record results
6. [ ] Create the full comparison table including Assignment 1 results
7. [ ] Write comments analyzing the differences in performance

## 📝 Notes

- CNN works on raw pixel data — no DCT/PCA features needed
- Compare training time, testing time, and accuracy across all methods
- The comparison table is a key deliverable — make sure to include ALL methods from both assignments

## ✅ Status

- ⬜ Not started
