# ALLaM-7B-Instruct-preview — Arabic Emotion Detection

This directory contains the notebook to evaluate Saudi Arabia's `ALLaM-7B-Instruct-preview` model on the Arabic Emotion Detection task using **Zero-Shot** and **Few-Shot** prompting strategies.

> **Note:** This is an open-weights model (7B parameters). It requires a GPU to run and is designed to be executed on **Kaggle** with a T4 GPU.

## How to Run on Kaggle

### Step 1: Upload the Dataset to Kaggle

1. Go to [kaggle.com/datasets](https://www.kaggle.com/datasets) → **New Dataset**.
2. Upload the file `Dataset/emotion_dataset_v2.jsonl` from the `problem 5` directory.
3. Name it something like `emotion-dataset` (the notebook auto-detects the file).
4. Set visibility to **Private** and click **Create**.

### Step 2: Set Up Hugging Face Token

ALLaM requires Hugging Face authentication:

1. Go to [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens) and create a token with **Read** access.
2. Accept the model license at [huggingface.co/ALLaM-AI/ALLaM-7B-Instruct-preview](https://huggingface.co/ALLaM-AI/ALLaM-7B-Instruct-preview).
3. On Kaggle, go to **Settings** → **Secrets** → **Add Secret**.
4. Name: `HF_TOKEN`, Value: your Hugging Face token.

### Step 3: Create a New Kaggle Notebook

1. Go to [kaggle.com/code](https://www.kaggle.com/code) → **New Notebook**.
2. Click **File** → **Import Notebook** → upload `ALLaM-7B-Instruct-preview.ipynb`.
3. On the right panel:
   - **Accelerator** → Select **GPU T4 x2**
   - **Internet** → Enable **Internet**
   - **Add Data** → Search for your `emotion-dataset` and add it
   - **Secrets** → Toggle ON the `HF_TOKEN` secret
4. Click **Run All** (or run cells one by one).

### Step 4: Download Results

After execution completes (~15-20 min), go to the **Output** tab and download:

| File | Description |
|------|-------------|
| `ALLaM-7B-Instruct-preview_zero_shot_results.jsonl` | Raw zero-shot predictions |
| `ALLaM-7B-Instruct-preview_few_shot_results.jsonl` | Raw few-shot predictions |
| `1_ALLaM-7B-Instruct-preview_Overview_Summary.csv` | Overall accuracy & F1 summary |
| `2_ALLaM-7B-Instruct-preview_F1_Metrics_Details.csv` | Per-emotion metrics |
| `3_ALLaM-7B-Instruct-preview_Error_Analysis_Log.csv` | Misclassified sentences |

## Results

| Metric | Value |
|--------|-------|
| Zero-Shot Accuracy | 57.0% |
| Few-Shot Accuracy | 64.0% |
| Zero-Shot Macro F1 | 28.3% |
| Few-Shot Macro F1 | 30.5% |

> ALLaM-7B benefits the most from few-shot prompting among all 5 models (+7.0% accuracy), suggesting smaller Arabic LLMs can leverage in-context learning effectively.

## Notes

- **GPU Required:** T4 x2 recommended. The model is ~14 GB in FP16.
- **Runtime:** ~15-20 minutes for 200 total inferences (100 zero-shot + 100 few-shot).
- **Hugging Face Login:** Required to download model weights.
