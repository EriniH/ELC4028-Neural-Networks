# Problem 4 — Autoencoder for Speech Utterance Representation

## 🎯 Objective

Develop an **Autoencoder (AE)** to compress each speech utterance into a **single fixed-length vector** representation, then use that representation to classify the spoken digit. This problem explores how autoencoders can be used for **dimensionality reduction** of variable-length sequential data.

## 📋 Requirements

### Step 1 — Baseline: Average Frame

- Divide each utterance into frames (~15 ms each)
- Calculate the **average frame** across all frames to get a single fixed-length vector per utterance
- Use this vector as input to a classifier
- This serves as the **baseline** for comparison

### Step 2 — Autoencoder Approach

Use an autoencoder to generate a single vector per utterance:

#### Method: Concatenate All Frames

1. Divide each utterance into frames (~15 ms)
2. **Concatenate all frames** into one long vector
3. Since utterances have different lengths, **pad shorter utterances with zero frames** to match the maximum length
4. Feed the concatenated vector into an autoencoder
5. The **bottleneck layer** output becomes the fixed-length representation

**Conceptual diagram:**
```
Frame1 | Frame2 | Frame3 | ... | FrameN | ZeroPad | ZeroPad
                          ↓
                    Autoencoder
                          ↓
              Single Fixed-Length Vector
                          ↓
                      Classifier
```

The AE progressively reduces dimensionality:
```
All Frames Concatenated → AE Layer 1 → ... → Bottleneck (single vector) → ... → Reconstruction
```

## 📦 Dataset

Same audio dataset as Problem 3, located in `../meterials/audio-dataset/`

## 📊 Results to Collect

| Method | Vector Length | Accuracy (%) | Comments |
|--------|-------------|-------------|----------|
| Baseline (average frame) | | | |
| AE (concatenated frames) | | | |

## 🔧 Steps to Complete

1. [ ] Load the audio dataset
2. [ ] Frame each utterance into ~15 ms frames
3. [ ] Implement the baseline: compute average frame per utterance, classify
4. [ ] Determine the maximum utterance length (number of frames)
5. [ ] Concatenate all frames per utterance, zero-pad to max length
6. [ ] Build and train an autoencoder on the concatenated frame vectors
7. [ ] Extract bottleneck representations for all utterances
8. [ ] Train a classifier on the bottleneck features
9. [ ] Compare with baseline results

## 📝 Notes

- The bottleneck vector length is **your choice** — experiment with different sizes
- Consider using a symmetric autoencoder architecture
- The classifier on top can be any standard classifier (MLP, SVM, etc.)
- Compare how the AE representation performs vs. the simple average frame baseline
- This problem demonstrates how AEs can handle variable-length inputs via padding + compression

## ✅ Status

- ⬜ Not started
