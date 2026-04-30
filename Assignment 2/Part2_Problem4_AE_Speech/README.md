# Problem 4 - AutoEncoder Representation for Speech Utterances

This folder contains the Assignment 2, Problem 4 implementation for representing each spoken-digit utterance as a single fixed-length vector, then classifying the utterance from that representation.

## Objective

Evaluate two single-vector representations for the speech utterances from Problem 3:

- Baseline average-frame vector
- AutoEncoder bottleneck vector from concatenated spectrogram frames

The final classifier predicts the spoken digit `0` through `9`.

## Dataset

This folder includes a local copy of the audio dataset:

```text
audio-dataset/
├── Train/
└── Test/
```

| Split | Files |
|---|---:|
| Train | 1,200 |
| Test | 300 |

## Implementation

| File | Purpose |
|---|---|
| [`baseline.py`](baseline.py) | Extracts the baseline average-frame features and saves `.npy` arrays. |
| [`baseline_classifier_run.py`](baseline_classifier_run.py) | Trains a dense classifier on the baseline features. |
| [`input_preparation.py`](input_preparation.py) | Frames each utterance, pads/truncates to a fixed length, and prepares AutoEncoder inputs. |
| [`auto_encoder.py`](auto_encoder.py) | Trains the AutoEncoder and saves bottleneck features. |
| [`classifier_main.py`](classifier_main.py) | Trains MLP classifiers on the AutoEncoder bottleneck features. |
| [`requirements.txt`](requirements.txt) | Python package requirements. |

## Feature Representations

### Baseline Average Frame

Each utterance is split into 15 ms spectrogram frames. The 257 frequency bins are averaged across time, producing one `257`-dimensional vector per utterance.

### AutoEncoder Bottleneck

Each utterance is converted into 100 spectrogram frames with 257 bins per frame:

```text
100 frames x 257 bins = 25,700 input features
```

Shorter utterances are padded with zeros and longer utterances are truncated. The AutoEncoder compresses the flattened vector into a `256`-dimensional bottleneck representation:

```text
25700 -> 1024 -> 256 -> 1024 -> 25700
```

The bottleneck vector is then used as input to the final digit classifier.

## Results

AutoEncoder classifier results are saved in [`results_after_classification/prob4_comparative_report.txt`](results_after_classification/prob4_comparative_report.txt). The baseline result is documented in the Assignment 2 report.

| Representation / classifier | Vector length | Accuracy |
|---|---:|---:|
| Baseline average frame + dense classifier | 257 | 28.3% |
| AE bottleneck + 1-hidden-layer MLP | 256 | 78.0% |
| AE bottleneck + 3-hidden-layer MLP | 256 | 82.0% |
| AE bottleneck + 4-hidden-layer MLP | 256 | 82.0% |
| AE bottleneck + 3-hidden-layer MLP, no regularization | 256 | 79.3% |
| AE bottleneck + 3-hidden-layer MLP, high learning rate | 256 | 79.3% |

The AutoEncoder bottleneck representation significantly outperformed the simple average-frame baseline.

## How to Run

Install dependencies:

```bash
pip install -r requirements.txt
```

Generate baseline average-frame features:

```bash
python baseline.py
```

Train the baseline classifier:

```bash
python baseline_classifier_run.py
```

Train the AutoEncoder and export bottleneck features:

```bash
python auto_encoder.py
```

Train and evaluate classifiers on the AutoEncoder features:

```bash
python classifier_main.py
```

Generated artifacts are written to:

- [`results/`](results/) for AutoEncoder plots and feature arrays
- [`results_after_classification/`](results_after_classification/) for classifier curves and reports

## Completeness Check

Implemented:

- Average-frame baseline feature extraction is implemented.
- Concatenated-frame AutoEncoder representation is implemented with padding/truncation.
- Bottleneck features are classified using multiple MLP configurations.
- AutoEncoder curves, classifier curves, and classifier report files are present.

Needs attention for clean GitHub reproducibility:

- The generated `.npy` feature arrays are not present in the current checkout. Run `baseline.py` and `auto_encoder.py` before running the classifier scripts.
- `baseline_classifier_run.py` prints the baseline accuracy to the terminal but does not save a baseline result text file. The baseline accuracy is currently preserved in the Assignment 2 report.
