# Jais-2-8B-Chat — Arabic Emotion Detection

This directory contains the notebook to evaluate UAE's `Jais-2-8B-Chat` model on the Arabic Emotion Detection task using **Zero-Shot** and **Few-Shot** prompting strategies.

> **Note:** This is an open-weights model (8B parameters). It requires a GPU to run and is designed to be executed on **Kaggle** with a T4 GPU.

## How to Run on Kaggle

### Step 1: Upload the Dataset to Kaggle

1. Go to [kaggle.com/datasets](https://www.kaggle.com/datasets) → **New Dataset**.
2. Upload the file `Dataset/emotion_dataset_v2.jsonl` from the `problem 5` directory.
3. Name it something like `emotion-dataset` (the notebook auto-detects the file).
4. Set visibility to **Private** and click **Create**.

> If you already uploaded the dataset for ALLaM, you can skip this step — both notebooks use the same dataset.

### Step 2: Set Up Hugging Face Token

Jais requires Hugging Face authentication:

1. Go to [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens) and create a token with **Read** access.
2. On Kaggle, go to **Settings** → **Secrets** → **Add Secret**.
3. Name: `HF_TOKEN`, Value: your Hugging Face token.

### Step 3: Create a New Kaggle Notebook

1. Go to [kaggle.com/code](https://www.kaggle.com/code) → **New Notebook**.
2. Click **File** → **Import Notebook** → upload `Jais-2-8B-Chat.ipynb`.
3. On the right panel:
   - **Accelerator** → Select **GPU T4 x1** (8B model fits in single T4)
   - **Internet** → Enable **Internet**
   - **Add Data** → Search for your `emotion-dataset` and add it
   - **Secrets** → Toggle ON the `HF_TOKEN` secret
4. Click **Run All** (or run cells one by one).

### Step 4: Download Results

After execution completes (~10-15 min), go to the **Output** tab and download:

| File | Description |
|------|-------------|
| `Jais-2-8B-Chat_zero_shot_results.jsonl` | Raw zero-shot predictions |
| `Jais-2-8B-Chat_few_shot_results.jsonl` | Raw few-shot predictions |
| `1_Jais-2-8B-Chat_Overview_Summary.csv` | Overall accuracy & F1 summary |
| `2_Jais-2-8B-Chat_F1_Metrics_Details.csv` | Per-emotion metrics |
| `3_Jais-2-8B-Chat_Error_Analysis_Log.csv` | Misclassified sentences |

## Results

| Metric | Value |
|--------|-------|
| Zero-Shot Accuracy | 66.0% |
| Few-Shot Accuracy | 66.0% |
| Zero-Shot Macro F1 | 42.0% |
| Few-Shot Macro F1 | 37.9% |

> Jais-2-8B-Chat is the strongest open-weights model in this benchmark, outperforming the 27B Fanar model in zero-shot F1 despite having only 8B parameters.

## Notes

- **GPU Required:** T4 x1 is sufficient for the 8B model.
- **Runtime:** ~10-15 minutes for 200 total inferences.
- **`trust_remote_code=True`:** Required because Jais uses a custom model architecture not yet natively supported by `transformers`.
- **Model fallback:** The notebook tries `core42/jais-2-8b-chat` first, then falls back to `inceptionai/jais-2-8b-chat` if the first fails.
