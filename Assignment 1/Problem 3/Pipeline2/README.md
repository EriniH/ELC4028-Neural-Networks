# Pipeline 2: Manual Seed Labeling + Active Learning + Pseudo-Labeling

## Overview

This project implements a human-in-the-loop labeling pipeline for handwritten Indian digits.

The pipeline now starts by creating a random oversized seed pool from the full 10,000-image dataset, asks you to manually label those sampled images, then randomly truncates to a balanced 300-image seed set (30 per class). From there it trains and iteratively refines with uncertain-sample annotation plus pseudo-labeling.

## Repository Structure

- `complete_pipeline.py`: Main end-to-end pipeline.
- `check_accuracy.py`: Optional oracle evaluator.
- `requirements.txt`: Dependencies.
- `Indian_Digits_Train/`: Full image pool (`*.bmp`, 10,000 expected).
- `PracticalGroundTruth500/`: Practical evaluation labels (`0..9` folders).
- `TrainingSeed300/`: Seed work area and final balanced seed labels.
- `uncertain_for_annotation/`: Iteration outputs and metrics.

Extract the 10,000-image dataset into `Indian_Digits_Train/` at the repository root.

## Data Contract

- Image format: `.bmp`
- Filename format: numeric id, for example `123.bmp`
- Full pool path: `Indian_Digits_Train/*.bmp`
- Practical ground truth path: `PracticalGroundTruth500/*/*.bmp`

Numeric IDs are used for sample tracking and evaluation keys.

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Run

```powershell
python complete_pipeline.py
```

## End-to-End Workflow

1. Script loads full dataset paths.
2. Script randomly samples an oversized seed pool (`SEED_POOL_SIZE`) from the full dataset and copies those images into `TrainingSeed300` root.
3. Script creates class folders `TrainingSeed300/0` ... `TrainingSeed300/9`.
4. You manually label sampled images by copying/moving them from `TrainingSeed300` root into the correct class folders.
5. Once each class has at least 30 labeled images, script randomly keeps exactly 30 per class (balanced 300 total).
6. Script trains stage-1 SVM on seed + deterministic augmentations.
7. Script exports 20 most uncertain samples per iteration to `uncertain_for_annotation/iter_XXX/` for manual labeling.
8. Script pseudo-labels high-confidence samples using margin threshold at the 75th percentile.
9. Script retrains, evaluates, records history, and checks stopping criteria.

## Manual Labeling Instructions

### A) Initial seed labeling (in `TrainingSeed300`)

1. Open sampled images in `TrainingSeed300` root.
2. Copy/move each sampled image into one of:

   - `TrainingSeed300/0`
   - `TrainingSeed300/1`
   - ...
   - `TrainingSeed300/9`

3. Keep labeling until each class folder has at least 30 images.
4. Press Enter in the running script.

### B) Iteration boundary labeling (in `uncertain_for_annotation/iter_XXX`)

1. Review `uncertain_for_annotation/iter_XXX/annotation_list.txt`.
2. Label by moving exported images into `uncertain_for_annotation/iter_XXX/0..9`.
3. Press Enter in the running script.

## Augmentation and Model

- Deterministic augmentations per seed image:

  - rotation `-5` and `+5` degrees
  - Gaussian noise
  - shifts up/down/left/right

- Model: `sklearn.svm.SVC` with `kernel="rbf"`, `decision_function_shape="ovo"`, `probability=True`.
- Sample weights:

  - seed/manual labels: `100`
  - augmentation/pseudo labels: `1`

## Pseudo-Label Selection Semantics

- For each class, the script takes the top 50 margin-ranked predictions.
- From that top-50 window, it keeps only those with margin >= global 75th percentile threshold.
- Reported rejected count is out of this top-50/class window, not out of all candidates.

## Stopping Criteria

Iteration stops when either:

- accuracy >= `99%`, or
- improvement < `0.1%`

If oracle is available, stopping metric is oracle accuracy; otherwise practical accuracy.

The script also prints an explicit final stop reason in the final summary.

## Outputs

### Iteration artifacts

- `uncertain_for_annotation/iter_XXX/annotation_list.txt`
- `uncertain_for_annotation/iter_XXX/0..9/` (manual boundary labels)

### Metrics

- `uncertain_for_annotation/iteration_metrics.tsv`

Per iteration columns include:

- practical accuracy
- oracle accuracy
- manual labels added
- pseudo labels added
- pseudo labels rejected
- pseudo candidate count
- pseudo threshold
- pseudo top-k considered count

### Console report

Final report includes:

- block diagram
- compact iteration table
- final accuracy
- iterations performed
- total manual labels and estimated manual time
- pseudo rejected counts with top-k denominator
- explicit stop reason

## Key Configuration (in `complete_pipeline.py`)

- `SEED_POOL_SIZE`
- `SEED_TARGET_PER_CLASS`
- `N_UNCERTAIN_PER_ITER`
- `N_PSEUDO_PER_CLASS`
- `PSEUDO_MARGIN_PERCENTILE`
- `TARGET_ACCURACY`
- `MIN_IMPROVEMENT`

## Troubleshooting

- If seed step is waiting, verify `TrainingSeed300/0..9` each has >= 30 labeled images from sampled pool.
- If iteration step is waiting, verify images are moved into `uncertain_for_annotation/iter_XXX/0..9`.
- If oracle output is missing, ensure `check_accuracy.py` is present and importable.
- If loading fails, verify folder names and `.bmp` naming conventions.
