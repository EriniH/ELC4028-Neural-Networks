# Problem 2 - CNN Classification on ReducedMNIST

This folder contains the Assignment 2, Problem 2 implementation for training Convolutional Neural Networks (CNNs) directly on ReducedMNIST images without a separate feature-extraction stage.

## Objective

Train a baseline CNN on `28 x 28` grayscale ReducedMNIST images, then evaluate at least two hyperparameter or architecture variations and compare their performance.

## Dataset

The default dataset path is:

```text
../../Materials/ReducedMNIST_generated
```

Expected structure:

```text
ReducedMNIST_generated/
├── train/
│   ├── 0/
│   ├── 1/
│   └── ...
└── test/
    ├── 0/
    ├── 1/
    └── ...
```

| Split | Samples per digit | Total samples |
|---|---:|---:|
| Train | 1,000 | 10,000 |
| Test | 200 | 2,000 |

## Implementation

| File | Purpose |
|---|---|
| [`LoadData.py`](LoadData.py) | Builds PyTorch `DataLoader` objects from the train/test image folders. |
| [`model.py`](model.py) | Defines the baseline CNN and three model variations. |
| [`train_cnn_variations.py`](train_cnn_variations.py) | Trains all CNN variants, evaluates them, and writes CSV/JSON summaries. |
| [`requirements.txt`](requirements.txt) | Python package requirements. |

The baseline model is a LeNet-style network adapted for `28 x 28` grayscale images with ReLU activations.

## CNN Variations

| Variation | Description |
|---|---|
| `BaseCNN` | Baseline LeNet-style CNN with ReLU. |
| `Variation1_Wider` | Increased convolution filters and fully connected layer widths. |
| `Variation2_LeakyReLU` | Replaced ReLU with LeakyReLU. |
| `Variation3_Dropout` | Added Dropout regularization before the later fully connected layers. |

## Results

Results are saved in [`part2_results/cnn_variations_results.csv`](part2_results/cnn_variations_results.csv) and [`part2_results/cnn_variations_results.json`](part2_results/cnn_variations_results.json).

| Variation | Accuracy | Training time | Testing time |
|---|---:|---:|---:|
| BaseCNN | 94.9% | 90467.0 ms | 14418.7 ms |
| Variation1_Wider | 97.0% | 19062.4 ms | 503.3 ms |
| Variation2_LeakyReLU | 96.3% | 15780.3 ms | 433.4 ms |
| Variation3_Dropout | 97.0% | 16025.5 ms | 426.3 ms |

The wider CNN and dropout CNN achieved the best recorded CNN accuracy at `97.0%`.

## How to Run

Install dependencies:

```bash
pip install -r requirements.txt
```

Run a quick smoke test:

```bash
python train_cnn_variations.py --epochs 1 --max-train-batches 2 --max-test-batches 2
```

Run the full experiment set:

```bash
python train_cnn_variations.py --epochs 6 --batch-size 64
```

Use a custom dataset path if needed:

```bash
python train_cnn_variations.py --data-dir "../../Materials/ReducedMNIST_generated"
```

## Completeness Check

Implemented:

- CNNs train directly on images with no DCT/PCA/AutoEncoder feature extraction.
- Baseline CNN uses ReLU and is adjusted for `28 x 28` inputs.
- Three variations are implemented, exceeding the requirement of at least two.
- Accuracy, training time, and testing time are exported in GitHub-readable CSV/JSON files.

Remaining report-level item:

- The broader comparison with Assignment 1 classifiers and Problem 1 MLP results is documented in the Assignment 2 report; this folder only stores the CNN-specific experiment outputs.
