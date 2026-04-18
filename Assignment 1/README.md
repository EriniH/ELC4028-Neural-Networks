# ELC4028 Neural Networks - Assignment 1

This repository contains the full work for Assignment 1, organized by problem and pipeline.

## Repository Structure

```text
Assignment 1/
	Problem 1/
		regression.ipynb
	Problem 2/
		part2_reduced_mnist.py
		part2_results/
	Problem 3/
		Pipeline1/
			pipeline1_human_in_loop.py
			pipeline1_results/
		Pipeline2/
			complete_pipeline.py
			uncertain_for_annotation/
		Pipeline3/
			train_digits_staged.py
			predict_digits_cnn.py
			runs/
	report/
		main.tex
		main.pdf
```

## Problem Index

- Problem 1 (Regression notebook): [Problem 1/README.md](Problem%201/README.md)
- Problem 2 (ReducedMNIST pipeline): [Problem 2/README.md](Problem%202/README.md)
- Problem 3 - Pipeline 1 (K-means + human-in-the-loop SVM): [Problem 3/Pipeline1/README.md](Problem%203/Pipeline1/README.md)
- Problem 3 - Pipeline 2 (seed labeling + active learning + pseudo-labeling): [Problem 3/Pipeline2/README.md](Problem%203/Pipeline2/README.md)
- Problem 3 - Pipeline 3 (staged CNN + practical adaptation): scripts in [Problem 3/Pipeline3](Problem%203/Pipeline3)

## Environment Setup

Use Python 3.10+.

Common packages across Problems 2 and 3:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install numpy pillow scipy scikit-learn scikit-image tqdm
```

Additional packages by pipeline:

- Problem 2 auto-dataset generation: `pip install torch torchvision`
- Problem 2 optional confusion matrix PNG export: `pip install matplotlib`
- Problem 3 Pipeline 3: `pip install torch pillow`

## Quick Start Commands

Run from the repository root (`Assignment 1`) unless noted.

### Problem 1

Open and run the notebook:

- `Problem 1/regression.ipynb`

### Problem 2

```powershell
python "Problem 2/part2_reduced_mnist.py"
```

Outputs are written to `Problem 2/part2_results/`.

### Problem 3 - Pipeline 1

```powershell
python "Problem 3/Pipeline1/pipeline1_human_in_loop.py" --data-dir <path-to-Indian_Digits_Train>
```

Run outputs are created under `Problem 3/Pipeline1/pipeline1_results/run_YYYYMMDD_HHMMSS/`.

For practical 5% workflow details, see [Problem 3/Pipeline1/README.md](Problem%203/Pipeline1/README.md).

### Problem 3 - Pipeline 2

```powershell
python "Problem 3/Pipeline2/complete_pipeline.py"
```

Main artifacts include:

- `Problem 3/Pipeline2/uncertain_for_annotation/iteration_metrics.tsv`
- per-iteration manual annotation folders in `Problem 3/Pipeline2/uncertain_for_annotation/iter_XXX/`

### Problem 3 - Pipeline 3

Stage-wise training (base model):

```powershell
python "Problem 3/Pipeline3/train_digits_staged.py" --data-dir "Problem 3/Pipeline3/AHDBase_TrainingSet" --output-dir "Problem 3/Pipeline3/runs/ahd_base"
```

Domain adaptation fine-tune example:

```powershell
python "Problem 3/Pipeline3/train_digits_staged.py" --data-dir "Problem 3/Pipeline3/PracticalGroundTruth500" --resume-from "Problem 3/Pipeline3/runs/ahd_base/best_model.pt" --stage-sizes 5,10,20,30,40 --epochs-per-stage 20 --patience 5 --target-acc 1.0 --output-dir "Problem 3/Pipeline3/runs/ahd_plus_practical_ft"
```

Final inference example:

```powershell
python "Problem 3/Pipeline3/predict_digits_cnn.py" --checkpoint "Problem 3/Pipeline3/runs/ahd_plus_practical_ft/best_model.pt" --input-dir "Problem 3/Pipeline3/Indian_Digits_Train" --support-dir "Problem 3/Pipeline3/PracticalGroundTruth500" --support-per-class 20 --autocontrast --use-shape-prior --shape-prior-weight 0.2 --output-csv "Problem 3/Pipeline3/runs/indian_final_predictions.csv"
```

## Report

The report source is in [report](report) and includes:

- `report/main.tex` (entry point)
- `report/content.tex` (section includes)
- `report/main.pdf` (latest compiled PDF)

Build with `pdflatex` (run twice for references):

```powershell
cd report
pdflatex main.tex
pdflatex main.tex
```

## Reproducibility Notes

- Most scripts use seed `42` by default.
- Problem 3 pipelines are interactive/human-in-the-loop in key stages, so final accuracy can vary with manual labels.
- Keep practical evaluation and oracle evaluation separated when reporting results.
