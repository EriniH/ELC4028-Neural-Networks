# Problem 5 — Data Augmentation Study on ReducedMNIST

## 🎯 Objective

Investigate how **data augmentation** affects recognition accuracy on the ReducedMNIST dataset. You will generate augmented examples from varying amounts of real data and measure the impact on a **LeNet-5** classifier.

## 📋 Requirements

### Step 1 — Generate Augmented Data

Apply augmentation to the training images. Augmentation techniques to use:
- **Rotation**: rotate right/left by small angles (e.g., ~5°)
- **Translation**: shift the digit by random Δx and Δy
- **White noise**: add a percentage of white noise to the images
- Other transformations you find appropriate

### Step 2 — Design a Recognition Pipeline

Use **LeNet-5** as the base recognition model for all experiments.

### Step 3 — Fill the Results Table

Train models using different combinations of real and augmented data according to this matrix:

| | 350 real/digit | 750 real/digit | 1,000 real/digit |
|---|---|---|---|
| **0 generated** | 350 real | 750 real | 1,000 real |
| **1,000 generated** | 350 real + 1,000 gen | 750 real + 1,000 gen | 1,000 real + 1,000 gen |
| **1,500 generated** | 350 real + 1,500 gen | 750 real + 1,500 gen | 1,000 real + 1,500 gen |
| **2,000 generated** | 350 real + 2,000 gen | 750 real + 2,000 gen | 1,000 real + 2,000 gen |

> Each cell = one training experiment → one accuracy result  
> **Total: 12 experiments**  
> Test set is always the same: 200 examples per digit

### Key Insight

Generated (augmented) examples are created **from** the available real examples in that row. For instance:
- "350 real + 1,000 gen" means you have 350 real images and you generate 1,000 augmented images from those 350 real images
- "1,000 real + 2,000 gen" means you have 1,000 real images and generate 2,000 from them

## 📊 Results to Collect

| Real per digit | Generated per digit | Total Training | Accuracy (%) |
|---------------|--------------------:|---------------:|-------------|
| 350 | 0 | 3,500 | |
| 350 | 1,000 | 13,500 | |
| 350 | 1,500 | 18,500 | |
| 350 | 2,000 | 23,500 | |
| 750 | 0 | 7,500 | |
| 750 | 1,000 | 17,500 | |
| 750 | 1,500 | 22,500 | |
| 750 | 2,000 | 27,500 | |
| 1,000 | 0 | 10,000 | |
| 1,000 | 1,000 | 20,000 | |
| 1,000 | 1,500 | 25,000 | |
| 1,000 | 2,000 | 30,000 | |

## 🔧 Steps to Complete

1. [ ] Load ReducedMNIST
2. [ ] Implement augmentation functions (rotation, translation, noise, etc.)
3. [ ] Create subsets of 350, 750, and 1,000 real examples per digit
4. [ ] For each subset, generate 1,000 / 1,500 / 2,000 augmented examples
5. [ ] Build a LeNet-5 model
6. [ ] Run all 12 training experiments (3 real sizes × 4 generation amounts)
7. [ ] Evaluate each on the same test set (200 per digit)
8. [ ] Fill the results table
9. [ ] Comment on trends (e.g., does more augmentation always help? diminishing returns?)

## 📝 Notes

- No manual labeling is needed for augmented data — labels carry over from the originals
- The hint in the assignment says: "we increase the data generated as no labeling is needed, just computer time to generate"
- Results from this table will also be compared with Problem 6 (GAN-generated data)
- This problem demonstrates that **data augmentation is a powerful technique when real data is limited**

## ✅ Status

- ⬜ Not started
