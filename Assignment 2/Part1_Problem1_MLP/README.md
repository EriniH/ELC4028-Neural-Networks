# Problem 1 - MLP Classification on ReducedMNIST

This folder contains the Assignment 2, Problem 1 implementation for classifying ReducedMNIST digits using Multi-Layer Perceptrons (MLPs) with different hidden-layer depths.

## Objective

Train and evaluate MLP classifiers on the ReducedMNIST dataset using the required network depths:

- 1 hidden layer
- 3 hidden layers
- 4 hidden layers

The experiment records classification accuracy, training time, testing time, and training curves for each configuration.

## Dataset

The local dataset is stored in [`Reduced MNIST Data/`](Reduced%20MNIST%20Data/).

| Split | Samples per digit | Total samples |
|---|---:|---:|
| Train | 1,000 | 10,000 |
| Test | 200 | 2,000 |

Each image is loaded as a grayscale `28 x 28` image and normalized before training.

## Implementation

| File | Purpose |
|---|---|
| [`data_loader.py`](data_loader.py) | Loads train/test image folders, converts images to arrays, and normalizes pixel values. |
| [`models.py`](models.py) | Defines a configurable Keras MLP builder for 1, 3, or 4 hidden layers. |
| [`main_prob1.py`](main_prob1.py) | Runs the main MLP experiments and saves plots plus a comparative report. |
| [`run_autoencoder_features.py`](run_autoencoder_features.py) | Optional experiment script for AutoEncoder bottleneck features and classical classifiers. |
| [`requirements.txt`](requirements.txt) | Python package requirements. |

The main MLP uses ReLU activations, Adam optimization, optional Batch Normalization, and Dropout regularization.

## Results

Results are saved in [`results/prob1_comparative_report.txt`](results/prob1_comparative_report.txt).

| Configuration | Hidden layers | Regularization | Learning rate | Accuracy | Training time | Testing time |
|---|---:|---|---:|---:|---:|---:|
| Exp1_1_Hidden_Layer | 1 | BatchNorm + Dropout | 0.001 | 97.30% | 7111.6 ms | 113.9 ms |
| Exp2_3_Hidden_Layers | 3 | BatchNorm + Dropout | 0.001 | 97.25% | 9993.0 ms | 121.9 ms |
| Exp3_4_Hidden_Layers | 4 | BatchNorm + Dropout | 0.001 | 97.25% | 11824.0 ms | 133.3 ms |
| Exp4_No_Regularization | 3 | None | 0.001 | 96.70% | 5333.0 ms | 119.5 ms |
| Exp5_High_Learning_Rate | 3 | BatchNorm + Dropout | 0.010 | 96.60% | 8338.0 ms | 124.3 ms |

The required 1-, 3-, and 4-hidden-layer models all reached approximately `97.3%` accuracy. The additional ablation runs show that removing regularization or increasing the learning rate reduced performance slightly.

## How to Run

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the experiment pipeline:

```bash
python main_prob1.py
```

Generated artifacts are written to [`results/`](results/), including:

- `prob1_comparative_report.txt`
- one training/validation curve image per experiment

## Completeness Check

Implemented:

- ReducedMNIST train/test split is present with the required sample counts.
- MLPs with 1, 3, and 4 hidden layers are implemented and evaluated.
- Accuracy, training time, testing time, and learning curves are saved.

Needs attention if strict assignment-table reproduction is required:

- The assignment brief mentions using DCT features from Assignment 1. The current runnable script feeds flattened normalized `28 x 28` image vectors into the MLP. The report text labels the results as DCT-based, so the code and report wording should be aligned before final submission if DCT evidence is required.
- The optional AutoEncoder feature script exists, but the current Problem 1 results folder does not include a final DCT/PCA/AutoEncoder comparison table generated from that script.
