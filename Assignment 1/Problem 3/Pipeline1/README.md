# Pipeline 1: K-Means Bootstrapping + Human-in-the-Loop SVM

## Overview

This pipeline labels 10,000 handwritten digit images with reduced manual effort by combining:

1. K-means cluster bootstrapping (human labels clusters instead of individual samples)
2. Weighted SVM training
3. Iterative boundary-sample correction (human labels only the most uncertain predictions)

The implementation is in one script: `pipeline1_human_in_loop.py`.

## Main Files

- `pipeline1_human_in_loop.py`: End-to-end pipeline + practical subset helper commands
- `pipeline1_results/`: Run outputs (`run_YYYYMMDD_HHMMSS/` folders)
- `practical_500_from_accuracy_dataset.csv`: Example practical annotation CSV
- `accuracy_dataset/`: Optional local folder for practical labeling workflow

## Data Contract

- Expected image format: `.bmp`
- Expected file naming: numeric IDs (`1.bmp` to `10000.bmp`)
- Expected count for full run: exactly 10,000 images
- Expected image size: `28x28` grayscale

If the folder does not contain exactly 10,000 `.bmp` files, the script exits with an error.

## Environment Setup

From this folder (`Problem 3/Pipeline1`):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install numpy pillow scipy scikit-image scikit-learn tqdm
```

## Run Pipeline 1

Minimal run (recommended to always provide `--data-dir` explicitly):

```powershell
python pipeline1_human_in_loop.py --data-dir <path-to-Indian_Digits_Train>
```

### Interactive flow during a run

1. Load all images and extract features (default: HOG)
2. Run K-means (`K=50`)
3. For each cluster:
	- View sampled examples (default: 8)
	- Enter one of:
	  - `0..9` to label the full cluster
	  - `m` for mixed cluster (no propagation)
4. Train weighted SVM on current labels
5. Select lowest-margin boundary samples (batch size fixed to 30)
6. For each boundary sample, enter:
	- `0..9` to assign trusted human label
	- `s` to skip
7. Repeat until stopping criteria are met

## Default and Fixed Settings

- Feature: `hog` (other options: `raw`, `dct`, `pca`)
- K-means clusters: `50` (fixed by script choices)
- Boundary batch size: `30` (fixed by script choices)
- Max iterations: `8` (can be overridden)
- Target accuracy: `0.99`
- Minimum improvement threshold: `0.001` (0.1%)
- Cluster label weight: `1.0`
- Human boundary label weight: `100.0`

## Practical 5% Hold-Out Workflow (Moved Here)

Use this when you want a practical, human-labeled evaluation subset without oracle labels.

### 1) Prepare a fixed 500-image annotation sheet

```powershell
python pipeline1_human_in_loop.py practical-prepare --data-dir <path-to-Indian_Digits_Train> --output-csv <path-to-practical_500_to_label.csv> --sample-size 500 --seed 42
```

This creates CSV columns:

- `image_id`
- `predicted_label` (blank by design in fixed-subset mode)
- `human_label` (fill manually)
- `notes`

### 2) Manually fill `human_label`

Fill all 500 rows with values `0..9`.

### 3) Score practical accuracy

```powershell
python pipeline1_human_in_loop.py practical-score --annotated-csv <path-to-practical_500_to_label.csv> --output-dir pipeline1_results --strict-size 500 --save-json <path-to-practical_accuracy_summary.json>
```

If `--predictions-csv` is omitted, the script auto-selects the newest `run_*/predictions_final.csv` under `pipeline1_results`.

Printed metrics include:

- `correct/labeled`
- practical accuracy
- 95% Wilson confidence interval

### 4) Use practical labels during iterative reporting

```powershell
python pipeline1_human_in_loop.py --data-dir <path-to-Indian_Digits_Train> --practical-annotated-csv <path-to-practical_500_to_label.csv>
```

Accuracy source precedence in pipeline logs:

1. `--ground-truth-csv` (if provided)
2. `--practical-annotated-csv`

Optional mode:

```powershell
python pipeline1_human_in_loop.py --data-dir <path-to-Indian_Digits_Train> --practical-annotated-csv <path-to-practical_500_to_label.csv> --use-practical-for-training
```

Without `--use-practical-for-training`, practical labels are evaluation-only.

## Practical Quality Gate

When exactly 500 practical labels are used, the script enforces:

- required practical accuracy: strictly greater than `99%`

If this gate fails, the run exits with an error.

## Stopping Criteria

The run stops when any applicable criterion is reached:

- measured accuracy reaches/exceeds target (`--target-accuracy`)
- improvement drops below `--min-improvement`
- `--max-iters` reached (except in practical target mode, where it can continue)

If no accuracy reference is provided (`--ground-truth-csv` and `--practical-annotated-csv` both missing), the script falls back to prediction-change based stopping behavior.

## Outputs Per Run

Each run creates a folder under `pipeline1_results/run_YYYYMMDD_HHMMSS/` with:

- `cluster_bootstrap_labels.csv`
- `boundary_labels.csv`
- `iteration_log.csv`
- `predictions_final.csv`
- `summary.json`
- `cluster_previews/` (contact sheets for cluster review)
- `boundary_previews/` (per-sample review images)

## Command-Line Reference (Common)

```text
--data-dir PATH
--output-dir PATH
--feature {raw,dct,hog,pca}
--sample-per-cluster INT
--max-iters INT
--target-accuracy FLOAT
--min-improvement FLOAT
--cluster-weight FLOAT
--trusted-weight FLOAT
--ground-truth-csv PATH
--practical-annotated-csv PATH
--use-practical-for-training
```

## Manual Effort Accounting

The script reports estimated manual effort using:

- `20` seconds per cluster decision
- `10` seconds per boundary sample label

These totals are saved in `summary.json` and reflected in console output.

## Notes

- Keep practical and oracle evaluation results separated in your report.
- For reproducibility, keep a fixed practical subset seed (`--seed 42`).
- If you run from another folder, prefer absolute paths for all CSV and data arguments.
