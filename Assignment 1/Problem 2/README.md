# ReducedMNIST Classification Pipeline

## Overview

This project runs an end-to-end digit classification pipeline for ReducedMNIST and compares:

- Features: DCT, PCA, HOG
- Classifiers: per-class K-means and SVM (linear/RBF)

The script can use an existing ReducedMNIST folder or generate one automatically from MNIST.

## What The Script Does

1. Resolves dataset location (existing folder or generated dataset).
2. Validates train/test structure and class counts.
3. Loads and normalizes grayscale images.
4. Extracts DCT, PCA, and HOG features.
5. Runs 18 experiments:
   - K-means: K=1, 4, 16, 32 across 3 feature sets (12 total)
   - SVM: linear and RBF across 3 feature sets (6 total)
6. Saves assignment-style summary table, confusion matrices, and conclusions.

## Requirements

Install dependencies:

```bash
pip install numpy pillow scipy scikit-learn scikit-image tqdm
```

Only needed when auto-generating the dataset from MNIST:

```bash
pip install torch torchvision
```

Optional (for confusion matrix PNG images):

```bash
pip install matplotlib
```

## Usage

Run with defaults:

```bash
python part2_reduced_mnist.py
```

Use a specific dataset folder:

```bash
python part2_reduced_mnist.py --data-root path/to/ReducedMNIST
```

Use a custom output directory:

```bash
python part2_reduced_mnist.py --output-dir part2_results
```

Image size is expected to be 28:

```bash
python part2_reduced_mnist.py --image-size 28
```

## Default Dataset Resolution

If `--data-root` is not provided, the script searches in this order:

- `ReducedMNIST`
- `Reduced MNIST Data(just an experiment)`
- `ReducedMNIST_generated`

If none exist, it generates `ReducedMNIST_generated` from MNIST.

## Output Files

By default, outputs are written to `part2_results/`:

- `assignment_style_table.csv`: Accuracy and processing time table in assignment format
- `conclusions.txt`: Best model summary and comparisons
- `*_confusion.csv`: Confusion matrix values for best K-means and best SVM
- `*_confusion.png`: Confusion matrix plots (only if matplotlib is installed)

## Results Table (Latest Run)

The following values are from `part2_results/assignment_style_table.csv`:

| Classifier | Setting | DCT Accuracy | DCT Time | PCA Accuracy | PCA Time | HOG Accuracy | HOG Time |
| ----- | ----- | ----- | ----- | ----- | ----- | ----- | ----- |
| K-means Clustering | 1 | 78.10% | 4.8000 s | 85.20% | 2.4646 s | 89.50% | 34.9320 s |
| K-means Clustering | 4 | 84.40% | 5.0773 s | 82.60% | 3.6085 s | 92.00% | 42.1938 s |
| K-means Clustering | 16 | 88.75% | 12.0725 s | 85.70% | 5.8008 s | 94.35% | 81.4935 s |
| K-means Clustering | 32 | 89.65% | 38.3444 s | 85.60% | 8.7011 s | 93.95% | 211.3056 s |
| SVM | Linear | 90.25% | 11.4875 s | 90.65% | 10.6268 s | 97.45% | 58.1034 s |
| SVM | nonlinear* | 95.55% | 20.5602 s | 95.15% | 21.0544 s | 97.15% | 120.9442 s |

Notes:

- `nonlinear*` corresponds to RBF kernel (`gamma=scale`)
- Processing time includes feature extraction + training + testing

## Dataset Structure

Expected directory format:

```text
<data-root>/
  train/
    0/ ... 9/
  test/
    0/ ... 9/
```

Expected image counts:

- Train: 1000 images per digit
- Test: 200 images per digit

## Implementation Notes

- Reproducibility seed: `42`
- K-means uses `n_init=10` for more stable clustering
- `StandardScaler` is used in both per-class K-means and SVM pipeline
- Per-class K-means predicts by nearest centroid over all class centroids
- K-means distance computation is batched for memory efficiency

## Performance Considerations

- First run may download MNIST if dataset is not found locally
- HOG extraction is typically the slowest feature pipeline in this setup
- SVM usually produces higher accuracy than K-means on this dataset

## Example Results Format

The generated `assignment_style_table.csv` follows the assignment-friendly structure:

- grouped by classifier and setting
- each feature (DCT, PCA, HOG) has accuracy and processing time columns
- includes final notes describing nonlinear kernel and timing definition

## Git Tracking Note

Large generated datasets and caches are ignored by `.gitignore` (`ReducedMNIST_generated/`, `mnist_cache/`, etc.).

`part2_results/` is intentionally **not ignored**, so result artifacts can be committed.
