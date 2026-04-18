# Pipeline 3 - CNN Local Hybrid Track

## Overview

This folder contains the local CNN-based track for Assignment 1 Part 3.

The workflow is split into stages:

1. Train a base CNN on `AHDBase_TrainingSet`
2. Fine-tune the base checkpoint on `PracticalGroundTruth500`
3. Run practical-set inference with optional support prototypes and shape prior
4. Run full inference on `Indian_Digits_Train`

Main scripts:

- `train_digits_staged.py`: stage-wise training and fine-tuning
- `predict_digits_cnn.py`: single-image, dataset, and unlabeled-folder inference

## Folder Layout

- `AHDBase_TrainingSet/`: base training dataset (`0..9` folders)
- `PracticalGroundTruth500/`: labeled practical dataset
- `PracticalGroundTruth500_Unlabeled/`: practical unlabeled images
- `Indian_Digits_Train/`: full dataset for final prediction
- `runs/`: checkpoints and prediction outputs
- `requirements.txt`: Python dependencies

## Requirements

From this folder (`Problem 3/Pipeline3/CNN`):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

`requirements.txt` currently includes:

- `torch`
- `pillow`

## Data Contract

Expected dataset format for training/evaluation folders:

```text
<dataset_root>/
  0/
  1/
  ...
  9/
```

Accepted image extensions include `.bmp`, `.png`, `.jpg`, `.jpeg`, `.tif`, `.tiff`.

## Training Workflow

### 1) Base model training

```powershell
python train_digits_staged.py --data-dir .\AHDBase_TrainingSet --output-dir .\runs\ahd_base
```

### 2) Practical fine-tuning from base checkpoint

```powershell
python train_digits_staged.py --data-dir .\PracticalGroundTruth500 --resume-from .\runs\ahd_base\best_model.pt --stage-sizes 5,10,20,30,40 --epochs-per-stage 20 --patience 5 --target-acc 1.0 --output-dir .\runs\ahd_plus_practical_ft
```

## Inference Workflow

### 1) Practical-set tuning run (scored via labels CSV)

```powershell
python predict_digits_cnn.py --checkpoint .\runs\ahd_plus_practical_ft\best_model.pt --input-dir .\PracticalGroundTruth500_Unlabeled --labels-csv .\PracticalGroundTruth500\labels.csv --support-dir .\PracticalGroundTruth500 --support-per-class 20 --autocontrast --use-shape-prior --shape-prior-weight 0.2 --output-csv .\runs\practical_tuning.csv
```

### 2) Final full inference on 10,000 images

```powershell
python predict_digits_cnn.py --checkpoint .\runs\ahd_plus_practical_ft\best_model.pt --input-dir .\Indian_Digits_Train --support-dir .\PracticalGroundTruth500 --support-per-class 20 --autocontrast --use-shape-prior --shape-prior-weight 0.2 --output-csv .\runs\indian_final_predictions.csv
```

## Main Outputs

Training outputs in each run folder (for example `runs/ahd_base/`):

- `best_model.pt`
- `history.csv`
- `summary.json`
- stage checkpoints (`stage_01.pt`, `stage_02.pt`, ...)

Prediction outputs include:

- `runs/practical_tuning.csv`
- `runs/practical_tuning_support.csv`
- `runs/indian_final_predictions.csv`
- `runs/indian_final_predictions_support.csv`

`indian_final_predictions.csv` includes columns:

- `path`
- `predicted_label`
- `predicted_arabic_digit`
- `predicted_shape_description`
- `confidence`

## Current Recorded Metrics (From Existing Run Artifacts)

From `runs/ahd_base/summary.json`:

- `best_val_acc`: `0.98125`

From `runs/ahd_plus_practical_ft/summary.json`:

- `best_val_acc`: `1.0`

## Useful Notes

- Keep `--num-workers 0` on Windows if you see DataLoader worker issues.
- Use `--invert` during training or inference if foreground/background polarity is reversed.
- You can disable/enable adaptation features at inference time:
  - support prototypes via `--support-dir` and `--support-per-class`
  - shape prior via `--use-shape-prior` and `--shape-prior-weight`
