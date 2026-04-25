# Problem 1 — Multi-Layer Perceptron (MLP) on ReducedMNIST

## 🎯 Objective

Build and train a **Multi-Layer Perceptron (MLP)** to classify handwritten digits (0–9) using the ReducedMNIST dataset. You must experiment with **three different network depths**: 1, 3, and 4 hidden layers.

## 📋 Requirements

1. **Dataset**: ReducedMNIST
   - Training: 1,000 randomly selected examples per digit from the original MNIST
   - Testing: 200 examples per digit
2. **Features**: Use **DCT (Discrete Cosine Transform)** features — the same that were used in Assignment 1
3. **Architectures to implement**:
   - MLP with **1 hidden layer**
   - MLP with **3 hidden layers**
   - MLP with **4 hidden layers**
4. **Hyperparameters**: Free choice (learning rate, batch size, number of neurons per layer, activation functions, optimizer, etc.)

## 📊 Results to Collect

For each MLP variation (1, 3, 4 hidden layers), record:

| Feature | Variation | Accuracy (%) | Processing Time (ms) |
|---------|-----------|-------------|---------------------|
| DCT | 1-Hidden | | |
| DCT | 3-Hidden | | |
| DCT | 4-Hidden | | |
| PCA | 1-Hidden | | |
| PCA | 3-Hidden | | |
| PCA | 4-Hidden | | |
| AutoEncoder | 1-Hidden | | |
| AutoEncoder | 3-Hidden | | |
| AutoEncoder | 4-Hidden | | |

> **Note**: Accuracy should be reported as `XX.X%` (one decimal place). Processing time in milliseconds as `XX.X ms`.

## 🔧 Steps to Complete

1. [ ] Load the MNIST dataset and create the ReducedMNIST subset (1,000 train + 200 test per digit)
2. [ ] Extract DCT features from the images (reuse code from Assignment 1 if available)
3. [ ] Optionally extract PCA and AutoEncoder features for the comparison table
4. [ ] Build MLP models with 1, 3, and 4 hidden layers
5. [ ] Train each model and record training time
6. [ ] Evaluate on test set and record accuracy and testing time
7. [ ] Fill in the results table above

## 📝 Notes

- This problem connects directly to Assignment 1 results. The comparison table in Problem 2 will include both Assignment 1 classifiers (K-means, SVM) and the MLP results from this problem.
- Consider trying different numbers of neurons per layer and different activation functions.

## ✅ Status

- ⬜ Not started
