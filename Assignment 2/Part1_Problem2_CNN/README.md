# Problem 2 - CNN on ReducedMNIST

This folder contains a baseline LeNet-style CNN and multiple hyperparameter variations to satisfy Assignment 2, Problem 2.

## What is implemented

1. Baseline CNN for 28x28 grayscale images with ReLU activation.
2. At least two CNN variations (plus one optional extra):
	- `Variation1_Wider`: more convolution filters and wider fully connected layers.
	- `Variation2_LeakyReLU`: activation changed from ReLU to LeakyReLU.
	- `Variation3_Dropout`: dropout regularization added before the last FC layers.
3. End-to-end training script that records:
	- Accuracy (%)
	- Training time (ms)
	- Testing time (ms)
4. CSV and JSON output files ready to copy into your assignment table.

## Files

- `LoadData.py`: ReducedMNIST image loader (`train/` and `test/` folders).
- `model.py`: baseline CNN and variation model definitions.
- `train_cnn_variations.py`: experiment runner for all models.
- `requirements.txt`: minimal dependencies.

## Dataset path

Default dataset location expected by the script:

`../../Materials/ReducedMNIST_generated`

The directory must contain:

- `train/0..9`
- `test/0..9`

## Quick start

Install dependencies:

```bash
pip install -r requirements.txt
```

Run a fast smoke test:

```bash
python train_cnn_variations.py --epochs 1 --max-train-batches 2 --max-test-batches 2
```

Run full experiment set:

```bash
python train_cnn_variations.py --epochs 6 --batch-size 64
```

Optional custom dataset path:

```bash
python train_cnn_variations.py --data-dir "d:/EECE4/NeuralNetworks/repo-clone/Materials/ReducedMNIST_generated"
```

## Output files

Generated under `part2_results/`:

- `cnn_variations_results.csv`
- `cnn_variations_results.json`

The CSV is formatted for your Problem 2 table (variation description + accuracy + training/testing timing).

## Assignment mapping

- Problem 2(a): baseline CNN from raw images with ReLU and 28x28 input.
- Problem 2(b): hyperparameter variations and performance comparison.
- Problem 2(c): use the generated CNN results with your Assignment 1 results in the final report table.
