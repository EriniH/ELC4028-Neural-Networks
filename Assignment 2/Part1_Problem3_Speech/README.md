# Problem 3 - Speech Recognition from Spectrograms

This folder contains the Assignment 2, Problem 3 implementation for recognizing spoken digits from speech spectrograms and evaluating the effect of data augmentation.

## Objective

Convert each `.wav` utterance into a spectrogram image, train a CNN classifier, and repeat the experiment under four required settings:

- Part A: baseline model with no augmentation
- Part B: speech-domain augmentation
- Part C: spectrogram-image augmentation
- Part D: combined speech and image augmentation

## Dataset

The shared audio dataset is stored at:

```text
../../Materials/audio-dataset
```

Expected structure:

```text
audio-dataset/
├── Train/
└── Test/
```

| Split | Files | Description |
|---|---:|---|
| Train | 1,200 | Clean and noisy spoken-digit recordings |
| Test | 300 | Held-out clean and noisy spoken-digit recordings |

Filename format:

- `{SpeakerID}_{Digit}.wav` for clean recordings
- `{SpeakerID}n_{Digit}.wav` for noisy recordings

## Implementation

| File | Purpose |
|---|---|
| [`dataset.py`](dataset.py) | Loads audio files, creates Mel spectrograms, applies optional augmentations, and returns tensors. |
| [`model.py`](model.py) | Defines the compact spectrogram CNN. |
| [`main.py`](main.py) | Runs the four required experiments and saves curves plus a final summary. |
| [`requirements.txt`](requirements.txt) | Python package requirements. |

Each waveform is loaded at 16 kHz, converted to a 64-bin Mel spectrogram, resized to `64 x 64`, normalized, and classified by a CNN.

## Augmentation Strategy

| Experiment | Audio augmentation | Image augmentation |
|---|---|---|
| Part A | None | None |
| Part B | Speed factor `0.97` or `1.03`, plus small waveform noise | None |
| Part C | None | Horizontal squeeze/expand by about 3%, plus image noise |
| Part D | Speed/noise augmentation | Squeeze/expand/noise augmentation |

The test set is not augmented.

## Results

Results are saved in [`results/final_results.txt`](results/final_results.txt).

| Experiment | Accuracy | Training time | Testing time |
|---|---:|---:|---:|
| Part A - Baseline | 94.0% | 28.01 s | 0.40 s |
| Part B - Audio augmentation | 90.0% | 38.61 s | 0.35 s |
| Part C - Image augmentation | 93.0% | 37.49 s | 0.42 s |
| Part D - Combined augmentation | 89.7% | 47.68 s | 0.41 s |

The baseline spectrogram CNN produced the strongest recorded result. The augmentation settings increased training time and did not improve accuracy on this split.

## How to Run

Install dependencies:

```bash
pip install -r requirements.txt
```

Before running, update the `AUDIO_DATA_DIR` constant in [`main.py`](main.py) so it points to the audio dataset on your machine. For this repository layout, the intended dataset folder is:

```text
../../Materials/audio-dataset
```

Run all four experiments:

```bash
python main.py
```

Generated artifacts are written to [`results/`](results/), including:

- `final_results.txt`
- one training curve image per experiment

## Completeness Check

Implemented:

- Speech files are converted to spectrogram images.
- A CNN classifier is trained and evaluated.
- All four required experiment settings, Parts A through D, are implemented.
- Final accuracies, training times, testing times, and learning curves are saved.

Needs attention for clean GitHub reproducibility:

- [`main.py`](main.py) currently contains an absolute dataset path from the original machine. Update it to the repository-relative dataset path before rerunning.
