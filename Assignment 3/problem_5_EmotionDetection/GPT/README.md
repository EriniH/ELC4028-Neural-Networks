# GPT-4o-mini — Arabic Emotion Detection

This directory contains the notebook to evaluate OpenAI's `gpt-4o-mini` model on the Arabic Emotion Detection task using **Zero-Shot** and **Few-Shot** prompting strategies.

## How to Run

### Step 1: Set Up API Key

Create a `.env` file in the **parent directory** (`problem 5/`) with your OpenAI API key:

```
OPENAI_API_KEY=sk-proj-your_key_here
```

> You can get an API key from [OpenAI Platform](https://platform.openai.com/api-keys).

### Step 2: Install Dependencies

Open a terminal and run:

```bash
pip install openai pandas tqdm python-dotenv scikit-learn
```

### Step 3: Run the Notebook

**Option A — VSCode / Jupyter (Interactive):**
1. Open `gpt-4o-mini.ipynb` in VSCode or Jupyter Notebook.
2. Select the Python 3 kernel.
3. Run **Cell 1** first (`%pip install ...`) to ensure all packages are installed.
4. Run all remaining cells sequentially (Shift+Enter).

**Option B — Command Line (Headless):**
```bash
cd "Assignment 3/problem 5"
python -m jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=1800 "GPT/gpt-4o-mini.ipynb"
```

### Step 4: Check Results

After execution, the following files will appear in this directory:

| File | Description |
|------|-------------|
| `gpt-4o-mini_zero_shot_results.jsonl` | Raw zero-shot predictions |
| `gpt-4o-mini_few_shot_results.jsonl` | Raw few-shot predictions |
| `1_gpt-4o-mini_Overview_Summary.csv` | High-level accuracy & F1 summary |
| `2_gpt-4o-mini_F1_Metrics_Details.csv` | Per-emotion precision, recall, F1 |
| `3_gpt-4o-mini_Error_Analysis_Log.csv` | All misclassified sentences |

## Notes

- **Runtime:** ~5-7 minutes (OpenAI's API is fast).
- **Rate Limiting:** The notebook includes automatic exponential backoff for 429 errors.
- **Model Name:** You can change `MODEL_NAME` in the setup cell to use a different GPT model (e.g., `gpt-4o`, `gpt-4-turbo`).
