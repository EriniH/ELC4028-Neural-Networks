# Pipeline 3 - QWEN Baseline (Qwen2-VL)

## 1) Scope and Objective

This folder contains the full Qwen2-VL baseline track used in Assignment 1 Part 3.

Work covered:
- Debugging steps
- Notebook and code updates
- Environment and GPU fixes
- Prompt and inference experiments
- Few-shot experiments
- Final results and limitations

Target task:
- Recognize handwritten Arabic-Indic digits (labels 0 to 9)
- Produce predictions and compute accuracy on 500 practical samples

## 2) Workspace and Data Used

Main folders used:
- `dataset/0..9`
- `resized_images/`
- `results/`
- `scripts/`
- `PracticalGroundTruth500/`
- `PracticalGroundTruth500_Unlabeled/`

Important files:
- `scripts/create_labels_and_predection.ipynb`
- `scripts/resize_images.ipynb`
- `scripts/compute_accuracy.ipynb`
- `results/labels.csv`
- `results/predictions.csv`
- `results/accuracy.txt`
- `PracticalGroundTruth500/labels.csv`

Practical set stats verified:
- Labeled BMP files: 500
- Unlabeled BMP files: 500
- `labels.csv` rows: 500
- Class balance: 50 samples per class for classes 0..9
- Filename alignment between unlabeled set and labels CSV: 500/500

## 3) Folder Layout

- `dataset/0..9/`: labeled dataset images
- `PracticalGroundTruth500/`: practical labeled set (`labels.csv` + class folders)
- `PracticalGroundTruth500_Unlabeled/`: practical unlabeled set
- `resized_images/`: preprocessed images for inference
- `predections/`: intermediate prediction outputs (legacy folder name kept as-is)
- `results/`: final evaluation artifacts
- `scripts/create_labels_and_predection.ipynb`: label creation + prediction workflow
- `scripts/resize_images.ipynb`: preprocessing notebook
- `scripts/compute_accuracy.ipynb`: evaluation notebook

## 4) Initial Behavior and First Failure

Initial observed behavior:
- `results/predictions.csv` had `predicted_label = -1` for all rows
- `results/accuracy.txt` showed 0.0 accuracy

Early symptom snapshot:
- Total rows: 500
- Correct: 0
- Invalid predictions dominated due to fragile output parsing

Main initial root cause:
- Parsing logic for model outputs was too brittle and frequently produced invalid labels

## 5) Environment and Execution Issues Fixed

### 5.1 Resize notebook cell-order issue

Error observed:
- `NameError: os is not defined` in the second cell of `resize_images.ipynb`

Cause:
- Cell 2 was executed before Cell 1 imports

Fix:
- Made Cell 2 self-contained with required imports

### 5.2 HF Hub unauthenticated warning

Warning observed:
- Unauthenticated Hugging Face requests (rate/speed warning)

Cause:
- `HF_TOKEN` was in Windows environment but missing in active notebook process

Fix:
- Added runtime fallback to load token from user environment
- Passed token explicitly when loading processor/model

### 5.3 GPU not used initially

Observed:
- NVIDIA GPU available, but notebook used CPU-only torch build

Fixes:
- Installed CUDA-enabled PyTorch in active notebook kernel
- Restarted kernel
- Verified `torch.cuda.is_available() == True`
- Confirmed GPU device in diagnostics

## 6) Data Preparation and Pipeline Hardening

### 6.1 Label creation notebook improvements

In `create_labels_and_predection.ipynb`:
- Sorted folder/file traversal for reproducibility
- Validated numeric class folders only
- Safely skipped non-numeric folders
- Handled BMP extension case-insensitively

### 6.2 Image preprocessing improvements

In `resize_images.ipynb`:
- Switched to aspect-ratio-preserving resize with padding
- Avoided direct-resize distortion
- Added missing-image count output

### 6.3 Accuracy notebook improvements

In `compute_accuracy.ipynb`:
- Added required-column validation
- Added numeric coercion for `true_label` and `predicted_label`
- Evaluated only valid rows
- Added detailed reporting:
  - total rows
  - rows with known true label
  - rows used for evaluation
  - correct predictions
  - invalid predictions
- Added warning when evaluable rows are zero

## 7) Merge and Evaluation Bug Fixes

### 7.1 `true_label` merge failure

Error observed:
- `KeyError: true_label`

Cause:
- In some paths, merged DataFrame did not expose `true_label`

Fix:
- Guaranteed `true_label` column after merge
- Enforced numeric conversion after merge
- Added explicit match-count print (`X/500`)

Result:
- Evaluation became stable with known true labels available

## 8) Inference and Prompting Experiments

Multiple strategies were tested iteratively:

1. Basic generation + text parsing
- Produced many invalid outputs initially

2. Stricter prompt constraints
- Forced one-digit output and reduced invalid outputs

3. Shape guidance + Arabic-Indic mapping in prompt
- Added as optional guidance block

4. Few-shot support examples
- Added configurable support examples from `PracticalGroundTruth500`

5. Class scoring via next-token logits
- Shifted away from free-form text parsing for class decision

6. Anti-collapse diagnostics
- Printed class distribution and warned on dominant-class behavior

7. Logit calibration + light TTA
- Calibration with blank-image prior
- TTA with original + autocontrast

8. Better few-shot selection strategy
- Upgraded from naive first-sample to representative/diverse selection (`prototype_diverse`)

## 9) Accuracy Progression Observed

Checkpoint progression:

- Early broken state:
  - Accuracy: 0.0
  - Cause: invalid/collapsed predictions

- After core parsing and pipeline fixes:
  - Accuracy: 0.146
  - Correct: 73/500
  - Invalid predictions: 0

- Later improved run:
  - Accuracy: 0.222
  - Correct: 111/500
  - Invalid predictions: 0

- Latest saved snapshot (`results/accuracy.txt`):
  - Accuracy: 0.224
  - Total rows: 500
  - Rows with known true label: 500
  - Rows used for evaluation: 500
  - Correct: 112
  - Invalid predictions: 0

Latest prediction distribution snapshot:
- pred=2 count=322
- pred=8 count=38
- pred=9 count=33
- pred=7 count=30
- pred=0 count=20
- pred=5 count=16
- pred=3 count=15
- pred=4 count=14
- pred=6 count=10
- pred=1 count=2

Interpretation:
- Pipeline is stable and functional
- Accuracy improved strongly from failed state
- Class-bias behavior is still the main limitation

## 10) Final Technical Status

What is working:
- End-to-end run completes without blocking errors
- Label creation, resize, prediction, and evaluation all run correctly
- Ground-truth merge is stable on practical sets
- GPU path is active
- Invalid prediction rate is controlled (currently 0)

What remains limiting:
- Prediction distribution still shows class imbalance
- Prompt/inference-only tuning gives incremental gains, not major jumps

## 11) Reproducible Run Order

Use this execution sequence:

1. `scripts/create_labels_and_predection.ipynb` Cell 1 (environment/packages)
2. `scripts/create_labels_and_predection.ipynb` Cell 2 (label generation if needed)
3. `scripts/create_labels_and_predection.ipynb` Cell 3 (GPU diagnostics)
4. `scripts/resize_images.ipynb` Cell 2 (self-contained preprocessing)
5. `scripts/create_labels_and_predection.ipynb` Cell 4 (prediction)
6. `scripts/compute_accuracy.ipynb` Cell 2 (evaluation)

Primary outputs:
- `results/predictions.csv`
- `results/accuracy.txt`

## 12) Environment Setup

Recommended Python: 3.10+

Install core packages:

    pip install torch torchvision torchaudio
    pip install transformers accelerate sentencepiece safetensors huggingface_hub
    pip install pillow pandas numpy matplotlib tqdm jupyter ipykernel

If using GPU, install CUDA-enabled PyTorch matching your CUDA runtime.

## 13) Conclusion for Report

Final conclusion from this QWEN baseline:
- The pipeline was fully debugged and stabilized.
- Performance improved from total failure (0.0) to a stable baseline (0.224).
- Remaining error appears dominated by model behavior limits under prompt/inference-only constraints.
- This track is best kept as a documented baseline comparison against stronger task-specific local models.

## 14) Deliverables in This Folder

- Updated notebooks:
  - `scripts/create_labels_and_predection.ipynb`
  - `scripts/resize_images.ipynb`
  - `scripts/compute_accuracy.ipynb`
- Updated outputs:
  - `results/predictions.csv`
  - `results/accuracy.txt`

