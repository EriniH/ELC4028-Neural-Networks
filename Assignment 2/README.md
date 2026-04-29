# Assignment 2 — Artificial Neural Networks (ANN)

This repository contains the solutions for **Assignment 2** of the Neural Networks course (EECE4).

## 📂 Repository Structure

```
Assignment2/
├── README.md                  ← You are here
├── meterials/                 ← Provided materials (PDF + audio dataset)
│   ├── Assignment 2 ANN_V4.pdf
│   └── audio-dataset/
│       ├── Train/             ← 1200 .wav files (60 speakers × 10 digits × 2 versions)
│       └── Test/              ← 300 .wav files (15 speakers × 10 digits × 2 versions)
│
├── Part1_Problem1_MLP/        ← MLP on ReducedMNIST (DCT features)
│   └── README.md
│
├── Part1_Problem2_CNN/        ← CNN on ReducedMNIST (raw images)
│   └── README.md
│
├── Part1_Problem3_Speech/     ← Speech recognition from spectrograms
│   └── README.md
│
├── Part2_Problem4_AE_Speech/  ← Autoencoder for speech utterance representation
│   └── README.md
│
├── Part2_Problem5_Augmentation/ ← Data augmentation study on ReducedMNIST
│   └── README.md
│
└── Part2_Problem6_GAN/        ← GAN-based synthetic data generation
    └── README.md
```

## 🧩 Assignment Overview

| Part | Problem | Topic | Dataset |
|------|---------|-------|---------|
| I | 1 | Multi-Layer Perceptron (MLP) | ReducedMNIST |
| I | 2 | Convolutional Neural Network (CNN) | ReducedMNIST |
| I | 3 | Speech Recognition from Spectrograms | Audio dataset |
| II | 4 | Autoencoder for Speech Representation | Audio dataset |
| II | 5 | Data Augmentation Study | ReducedMNIST |
| II | 6 | GAN-based Synthetic Data Generation | ReducedMNIST |

## 📦 Dataset Details

### ReducedMNIST (Problems 1, 2, 5, 6)
A reduced version of the original MNIST handwritten digit dataset:
- **Training set**: 1,000 randomly selected examples per digit (10,000 total)
- **Test set**: 200 examples per digit (2,000 total)
- Images are 28×28 grayscale

### Audio Dataset (Problems 3, 4)
Speech recordings of digits 0–9 spoken by multiple speakers:
- **Training set**: ~1,200 `.wav` files (60 speakers × 10 digits, with clean + noisy versions)
- **Test set**: ~300 `.wav` files (15 speakers × 10 digits, with clean + noisy versions)
- File naming convention: `{SpeakerID}_{Digit}.wav` / `{SpeakerID}n_{Digit}.wav` (noisy)
- Speaker prefixes: `M` = Male, `F` = Female, `C` = Child, `U` = Unknown/Other

## 🚀 Getting Started

Navigate to each problem's folder and follow the instructions in its `README.md`.

## 📌 Status

| Problem | Status |
|---------|--------|
| Problem 1 — MLP | ⬜ inprogress |
| Problem 2 — CNN | ⬜ inprogress |
| Problem 3 — Speech Recognition | ⬜ Not started |
| Problem 4 — Autoencoder Speech | ⬜ Not started |
| Problem 5 — Data Augmentation | ⬜ Not started |
| Problem 6 — GAN | ⬜ Not started |
