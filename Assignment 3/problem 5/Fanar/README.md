# Fanar-C-2-27B — Arabic Emotion Detection

This directory contains the notebook to evaluate Qatar's `Fanar-C-2-27B` model on the Arabic Emotion Detection task using **Zero-Shot** and **Few-Shot** prompting strategies.

## How to Run

### Step 1: Set Up API Key

Create a `.env` file in the **parent directory** (`problem 5/`) with your Fanar API key:

```
FANAR_API_KEY=your_key_here
```

> You can get an API key from [Fanar Platform](https://fanar.qa/).

### Step 2: Install Dependencies

Open a terminal and run:

```bash
pip install openai pandas tqdm python-dotenv scikit-learn
```

> **Note:** We use the `openai` Python library because Fanar's API is fully OpenAI-compatible. The client is configured to point to `https://api.fanar.qa/v1`.

### Step 3: Run the Notebook

**Option A — VSCode / Jupyter (Interactive):**
1. Open `Fanar-C-2-27B.ipynb` in VSCode or Jupyter Notebook.
2. Select the Python 3 kernel.
3. Run **Cell 1** first (`%pip install ...`) to ensure all packages are installed.
4. Run all remaining cells sequentially (Shift+Enter).

**Option B — Command Line (Headless):**
```bash
cd "Assignment 3/problem 5"
python -m jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=1800 "Fanar/Fanar-C-2-27B.ipynb"
```

### Step 4: Check Results

After execution, the following files will appear in this directory:

| File | Description |
|------|-------------|
| `Fanar-C-2-27B_zero_shot_results.jsonl` | Raw zero-shot predictions |
| `Fanar-C-2-27B_few_shot_results.jsonl` | Raw few-shot predictions |
| `1_Fanar-C-2-27B_Overview_Summary.csv` | High-level accuracy & F1 summary |
| `2_Fanar-C-2-27B_F1_Metrics_Details.csv` | Per-emotion precision, recall, F1 |
| `3_Fanar-C-2-27B_Error_Analysis_Log.csv` | All misclassified sentences |

## Notes

- **Runtime:** ~4-5 minutes.
- **Content Filter:** Fanar has an aggressive safety filter that blocks some Arabic sentences (particularly those about violence or strong emotions). These are automatically detected and marked as `"FILTERED"` in the output — the notebook does **not** retry on permanent `content_filter` errors.
- **Rate Limiting:** Transient 429 errors are handled with automatic retry (max 3 attempts).
