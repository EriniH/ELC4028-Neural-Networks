# Problem 6 — GAN-based Synthetic Data Generation

## 🎯 Objective

Use a **Generative Adversarial Network (GAN)** to generate synthetic handwritten digit images and study how GAN-generated data can supplement limited real training data for improving recognition accuracy.

## 📋 Requirements

### Step 1 — Train a GAN and Generate Examples

1. Train a GAN network using only **350 training examples** per digit
2. Generate **3 different synthetic examples** from each digit
3. **Comment on**:
   - How close the generated images are to real ones (visual quality)
   - Computer time for training and generation
   - Mention the computer hardware setup used

**Reference papers/resources:**
- DCGAN paper: [https://arxiv.org/abs/1511.06434](https://arxiv.org/abs/1511.06434)
- Conditional GAN (CGAN) tutorial: [https://towardsdatascience.com/cgan-conditional-generative-adversarial-network-how-to-gain-control-over-gan-outputs-b30620bd0cc8/](https://towardsdatascience.com/cgan-conditional-generative-adversarial-network-how-to-gain-control-over-gan-outputs-b30620bd0cc8/)

> **Important**: Add timestamps in your code to capture training and generation time.

### Step 2 — Generate Data at Scale (Same Table as Problem 5)

Use the GAN to generate synthetic data in the same configuration as Problem 5's table:

| | 350 real/digit | 750 real/digit | 1,000 real/digit |
|---|---|---|---|
| **0 generated** | 350 real | 750 real | 1,000 real |
| **1,000 generated** | 350 real + 1,000 GAN | 750 real + 1,000 GAN | 1,000 real + 1,000 GAN |
| **1,500 generated** | 350 real + 1,500 GAN | 750 real + 1,500 GAN | 1,000 real + 1,500 GAN |
| **2,000 generated** | 350 real + 2,000 GAN | 750 real + 2,000 GAN | 1,000 real + 2,000 GAN |

Train a recognition model for each cell and compare with Problem 5 results (augmentation).

### Step 3 — Reducing Need for Real Data

Using the **350 real examples** case:
1. Select a **reasonable combination** of augmented (Problem 5) + synthetic (GAN) data
2. Compare the best result you can achieve with only 350 real examples + generated data **against** using all 1,000 real examples per digit
3. Answer: **To what extent can we reduce the need for real data?**

## 📊 Results to Collect

### GAN Sample Quality (Step 1)

| Digit | Generated Sample 1 | Generated Sample 2 | Generated Sample 3 | Visual Quality Comment |
|-------|----|----|----|----|
| 0 | (image) | (image) | (image) | |
| 1 | (image) | (image) | (image) | |
| ... | ... | ... | ... | |
| 9 | (image) | (image) | (image) | |

**Timing**: Training time = ___ | Generation time = ___  
**Hardware**: ___

### GAN Data Scale Results (Step 2)

| Real per digit | GAN-generated per digit | Accuracy (%) |
|---------------|------------------------:|-------------|
| 350 | 0 | |
| 350 | 1,000 | |
| 350 | 1,500 | |
| 350 | 2,000 | |
| 750 | 0 | |
| 750 | 1,000 | |
| 750 | 1,500 | |
| 750 | 2,000 | |
| 1,000 | 0 | |
| 1,000 | 1,000 | |
| 1,000 | 1,500 | |
| 1,000 | 2,000 | |

### Comparison: Augmentation vs. GAN vs. Real Data (Step 3)

| Method | Real/digit | Generated/digit | Accuracy (%) |
|--------|-----------|----------------|-------------|
| Real data only | 1,000 | 0 | |
| Augmentation only (best from P5) | 350 | ? | |
| GAN only (best from P6) | 350 | ? | |
| Augmentation + GAN combined | 350 | ? | |

## 🔧 Steps to Complete

1. [ ] Read the DCGAN paper and CGAN tutorial
2. [ ] Implement a GAN (DCGAN or CGAN recommended) for MNIST digits
3. [ ] Train the GAN on 350 examples per digit
4. [ ] Generate 3 sample images per digit and evaluate quality
5. [ ] Record training time and generation time (add timestamps in code)
6. [ ] Generate data at scale (1,000 / 1,500 / 2,000 per digit)
7. [ ] Train recognition models (LeNet-5) with GAN-generated data
8. [ ] Fill the results table
9. [ ] Combine augmentation + GAN data for the 350-example case
10. [ ] Compare with 1,000-example real data and write conclusions

## 📝 Notes

- A **Conditional GAN (CGAN)** is strongly recommended — it lets you control which digit to generate
- DCGAN (Deep Convolutional GAN) provides a stable starting architecture
- The goal of Step 3 is to show whether synthetic data can **replace** the need for collecting/labeling more real data
- Always mention the hardware setup (CPU/GPU, RAM, etc.) since GAN training time is hardware-dependent

## ✅ Status

- ⬜ Not started
