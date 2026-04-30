# Problem 3 — Speech Recognition from Speech Spectrograms

## 🎯 Objective

Develop a neural network to **recognize spoken digits (0–9)** by converting speech audio into **spectrogram images** and treating the problem as an image classification task. Additionally, explore the impact of **data augmentation** on recognition performance.

## 📋 Requirements

### Part (a) — Base Speech Recognition Model

1. Convert each speech `.wav` file into a **spectrogram image**
2. Train a CNN to classify the spectrogram images into one of 10 digit classes
3. **Hint**: You may start from the CNN in Problem 2 and enhance it, or use architectures from ImageNet-style networks

### Part (b) — Speech-level Data Augmentation

Repeat part (a) after augmenting the **raw audio data**:
- Speed up by ~3%
- Slow down by ~3%
- Optionally add speech noise

### Part (c) — Image-level Data Augmentation

Repeat part (a) after augmenting the **spectrogram images**:
- Squeeze (horizontally) by ~3%
- Expand (horizontally) by ~3%
- Optionally add noise to the images

### Part (d) — Combined Augmentation

Repeat part (a) using **both** speech-level and image-level augmentation from parts (b) and (c) combined.

## 📦 Dataset

Located in `../meterials/audio-dataset/`:

| Split | Files | Description |
|-------|-------|-------------|
| Train | ~1,200 `.wav` | 60 speakers × 10 digits × 2 (clean + noisy) |
| Test | ~300 `.wav` | 15 speakers × 10 digits × 2 (clean + noisy) |

**File naming convention:**
- `{SpeakerID}_{Digit}.wav` — clean version
- `{SpeakerID}n_{Digit}.wav` — noisy version
- Speaker prefixes: `M` = Male, `F` = Female, `C` = Child, `U` = Unknown/Other

## 📊 Results to Collect

| Experiment | Accuracy (%) | Training Time | Testing Time | Comments |
|------------|-------------|---------------|--------------|----------|
| (a) Base model (no augmentation) | | | | |
| (b) Speech augmentation only | | | | |
| (c) Image augmentation only | | | | |
| (d) Speech + Image augmentation | | | | |

## 🔧 Steps to Complete

1. [ ] Load and explore the audio dataset (understand file naming, sampling rate, duration)
2. [ ] Convert each `.wav` file to a spectrogram image
3. [ ] Build and train a CNN on the spectrogram images (part a)
4. [ ] Implement speech-level augmentation (speed up/down, noise) and retrain (part b)
5. [ ] Implement image-level augmentation (squeeze/expand, noise) and retrain (part c)
6. [ ] Combine both augmentations and retrain (part d)
7. [ ] Compare results across all four experiments
8. [ ] Write comments on the effect of each augmentation strategy

## 📝 Notes

- The key idea is to treat **speech as images** by using spectrograms
- Spectrograms can be generated using `librosa`, `scipy.signal.spectrogram`, or similar libraries
- Data augmentation is expected to improve robustness — document how much improvement you observe
- This dataset is also used in Problem 4 (Part II)

## ✅ Status

- ✅ Completed

## 🚀 How to Run

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the Experiment**:
   ```bash
   python main.py
   ```
   This script handles loading the speech data, extracting and augmenting spectrograms, training the models, and evaluating the final output.
