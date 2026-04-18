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
			CNN/
				train_digits_staged.py
				predict_digits_cnn.py
				runs/
			QWEN/
				scripts/
				results/
				README.md
	report/
		main.tex
		main.pdf
```

## Problem Index

- Problem 1 (Regression notebook): [Problem 1/README.md](Problem%201/README.md)
- Problem 2 (ReducedMNIST pipeline): [Problem 2/README.md](Problem%202/README.md)
- Problem 3 - Pipeline 1 (K-means + human-in-the-loop SVM): [Problem 3/Pipeline1/README.md](Problem%203/Pipeline1/README.md)
- Problem 3 - Pipeline 2 (seed labeling + active learning + pseudo-labeling): [Problem 3/Pipeline2/README.md](Problem%203/Pipeline2/README.md)
- Problem 3 - Pipeline 3 (CNN local hybrid): [Problem 3/Pipeline3/CNN/README.md](Problem%203/Pipeline3/CNN/README.md)
- Problem 3 - Pipeline 3 add-on baseline (Qwen2-VL): [Problem 3/Pipeline3/QWEN/README.md](Problem%203/Pipeline3/QWEN/README.md)

## Quick Start Commands

Run from the repository root (`Assignment 1`) unless noted.

### Problem 2

```powershell
python "Problem 2/part2_reduced_mnist.py"
```

### Problem 3 - Pipeline 1

```powershell
python "Problem 3/Pipeline1/pipeline1_human_in_loop.py" --data-dir <path-to-Indian_Digits_Train>
```

### Problem 3 - Pipeline 2

```powershell
python "Problem 3/Pipeline2/complete_pipeline.py"
```

### Problem 3 - Pipeline 3 (CNN)

```powershell
python "Problem 3/Pipeline3/CNN/train_digits_staged.py" --data-dir "Problem 3/Pipeline3/CNN/AHDBase_TrainingSet" --output-dir "Problem 3/Pipeline3/CNN/runs/ahd_base"
```

```powershell
python "Problem 3/Pipeline3/CNN/predict_digits_cnn.py" --checkpoint "Problem 3/Pipeline3/CNN/runs/ahd_plus_practical_ft/best_model.pt" --input-dir "Problem 3/Pipeline3/CNN/Indian_Digits_Train" --support-dir "Problem 3/Pipeline3/CNN/PracticalGroundTruth500" --support-per-class 20 --autocontrast --use-shape-prior --shape-prior-weight 0.2 --output-csv "Problem 3/Pipeline3/CNN/runs/indian_final_predictions.csv"
```

## Report

Report source and outputs:

- `report/main.tex`
- `report/content.tex`
- `report/main.pdf`

Build with `pdflatex` (run twice):

```powershell
cd report
pdflatex main.tex
pdflatex main.tex
```
