# Gemini 2.5 Flash — Arabic Emotion Detection

This directory contains the notebook to evaluate Google's `gemini-2.5-flash` model on the Arabic Emotion Detection task using **Zero-Shot** and **Few-Shot** prompting strategies.

## How to Run

### Step 1: Set Up API Key

Create a `.env` file in the **parent directory** (`problem 5/`) with your Gemini API key:

```
GEMINI_API_KEY=your_key_here
```

> You can get a free API key from [Google AI Studio](https://aistudio.google.com/apikey).

### Step 2: Install Dependencies

Open a terminal and run:

```bash
pip install google-genai pandas tqdm python-dotenv scikit-learn
```

### Step 3: Run the Notebook

**Option A — VSCode / Jupyter (Interactive):**
1. Open `gemini-2.5-flash.ipynb` in VSCode or Jupyter Notebook.
2. Select the Python 3 kernel.
3. Run **Cell 1** first (`%pip install ...`) to ensure all packages are installed.
4. Run all remaining cells sequentially (Shift+Enter).

**Option B — Command Line (Headless):**
```bash
cd "Assignment 3/problem 5"
python -m jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=1800 "gemini/gemini-2.5-flash.ipynb"
```

### Step 4: Check Results

After execution, the following files will appear in this directory:

| File | Description |
|------|-------------|
| `gemini-2.5-flash_zero_shot_results.jsonl` | Raw zero-shot predictions |
| `gemini-2.5-flash_few_shot_results.jsonl` | Raw few-shot predictions |
| `1_gemini-2.5-flash_Overview_Summary.csv` | High-level accuracy & F1 summary |
| `2_gemini-2.5-flash_F1_Metrics_Details.csv` | Per-emotion precision, recall, F1 |
| `3_gemini-2.5-flash_Error_Analysis_Log.csv` | All misclassified sentences |

## Notes

- **Runtime:** ~10-15 minutes (depends on API rate limits).
- **Rate Limiting:** The notebook includes automatic exponential backoff for 429/503 errors.
- **Safety Filters:** Some content may be blocked by Google's safety filters; these are handled gracefully.
