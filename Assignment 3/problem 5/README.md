# Problem 5: Benchmarking Arabic NLP Tasks Across LLMs

## Assignment Requirements

The goal is to benchmark the performance of **5 different Large Language Models** on an Arabic NLP task. Specifically:

1. Evaluate each model using **Zero-Shot** and **Few-Shot** prompting strategies.
2. Use a custom Arabic Emotion Detection dataset covering **MSA, Classical Arabic, and Dialects**.
3. Report **F1-Score** (macro) and **Accuracy** for each model/strategy pair.
4. Perform **Error Analysis** on failure cases.

---

## How We Meet the Requirements

| Requirement | Implementation |
|-------------|----------------|
| 5 LLMs | Gemini 2.5 Flash, GPT-4o-mini, Fanar-C-2-27B, ALLaM-7B, Jais-2-8B |
| Zero-Shot & Few-Shot | Identical Arabic prompts used across all 5 models |
| Arabic dataset | 100 sentences: ~33% MSA, ~33% Classical, ~33% Dialects |
| Metrics | Accuracy, Macro F1, per-emotion Precision/Recall/F1 |
| Error analysis | Misclassified sentences exported with ground truth vs prediction |

---

## Dataset

**File:** `Dataset/emotion_dataset_v2.jsonl`

Each sample contains Arabic text annotated with one of 6 emotions:

| Arabic | English |
|--------|---------|
| فرح | Joy |
| غضب | Anger |
| حزن | Sadness |
| خوف | Fear |
| اندهاش / مفاجأة | Surprise |
| اشمئزاز | Disgust |

The dataset includes Modern Standard Arabic (MSA), Classical Arabic poetry, and dialectal Arabic (Egyptian, Gulf, Levantine, Maghrebi).

---

## Models Evaluated

### 1. Gemini 2.5 Flash (Google)
- **Type:** API-based (cloud)
- **Parameters:** Not publicly disclosed (large-scale model)
- **API:** Google GenAI SDK (`google-genai`)
- **Execution:** Local machine via API key
- **Description:** Google's latest multimodal LLM. Accessed through the `genai.Client.models.generate_content` API with safety filters configured to BLOCK_NONE for evaluation integrity.

### 2. GPT-4o-mini (OpenAI)
- **Type:** API-based (cloud)
- **Parameters:** Not publicly disclosed
- **API:** OpenAI SDK (`openai`)
- **Execution:** Local machine via API key
- **Description:** OpenAI's cost-efficient model optimized for fast inference. Uses the `chat.completions.create` endpoint.

### 3. Fanar-C-2-27B (Qatar)
- **Type:** API-based (OpenAI-compatible endpoint)
- **Parameters:** 27 billion
- **API:** OpenAI SDK pointed to `https://api.fanar.qa/v1`
- **Execution:** Local machine via API key
- **Description:** Qatar's large Arabic-focused LLM. Uses an OpenAI-compatible API. Has an aggressive content safety filter that blocks some Arabic texts.

### 4. ALLaM-7B-Instruct-preview (Saudi Arabia)
- **Type:** Open-weights (Hugging Face)
- **Parameters:** 7 billion
- **Library:** `transformers` (AutoModelForCausalLM)
- **Execution:** Kaggle (T4 GPU)
- **Description:** Saudi Arabia's Arabic LLM developed by SDAIA. Requires Hugging Face authentication. Loaded in FP16 for memory efficiency.

### 5. Jais-2-8B-Chat (UAE)
- **Type:** Open-weights (Hugging Face)
- **Parameters:** 8 billion
- **Library:** `transformers` (AutoModelForCausalLM, `trust_remote_code=True`)
- **Execution:** Kaggle (T4 GPU)
- **Description:** UAE's bilingual Arabic-English LLM developed by Core42/Inception. Requires `trust_remote_code=True` due to custom architecture.

---

## Prompting Strategy

All 5 models use **identical prompts** in Arabic:

**Zero-Shot:** The model is given a task description and asked to classify the emotion in one word.

**Few-Shot:** The model receives 4 labeled examples (Joy, Sadness, Anger, Fear) before being asked to classify the target text.

Both prompts instruct the model to respond with **exactly one Arabic word** from the set: (فرح، غضب، حزن، خوف، مفاجأة، اشمئزاز).

---

## Results Summary

| Model | ZS Accuracy | FS Accuracy | ZS Macro F1 | FS Macro F1 |
|-------|-------------|-------------|-------------|-------------|
| Gemini 2.5 Flash | 73.0% | 73.0% | 58.3% | 58.6% |
| GPT-4o-mini | 74.0% | 69.0% | 46.8% | 48.5% |
| Fanar-C-2-27B | 69.0% | 62.0% | 39.4% | 37.8% |
| Jais-2-8B-Chat | 66.0% | 66.0% | 42.0% | 37.9% |
| ALLaM-7B | 61.0% | 64.0% | 28.3% | 30.5% |

---

## Manual Adjudication

An automated exact-match evaluation can sometimes be overly strict. To ensure fairness, we introduced a manual adjudication step (`manual_adjudication.py`) to review all mismatched predictions. This was necessary because:
1. **Label Synonymy:** Models may predict a valid synonym (e.g., `مفاجأة` instead of `اندهاش`).
2. **Emotion Overlap:** Many sentences can validly evoke multiple emotions (e.g., sadness and anger) depending on the situation.
3. **Golden Label Inaccuracies:** Subjectivity or human errors during dataset annotation.

After a human reviewer evaluated all mismatches, accepted predictions were reclassified as correct. The adjusted results are below:

### Adjusted Accuracy Results

| Model | ZS Original | ZS Adjusted | ZS Δ | FS Original | FS Adjusted | FS Δ |
|-------|-------------|-------------|------|-------------|-------------|------|
| Gemini 2.5 Flash | 73.0% | 87.0% | +14.0% | 73.0% | 86.0% | +13.0% |
| GPT-4o-mini | 74.0% | 91.0% | +17.0% | 69.0% | 84.0% | +15.0% |
| Fanar-C-2-27B | 69.0% | 86.0% | +17.0% | 62.0% | 75.0% | +13.0% |
| Jais-2-8B-Chat | 66.0% | 72.0% | +6.0% | 66.0% | 75.0% | +9.0% |
| ALLaM-7B | 61.0% | 69.0% | +8.0% | 64.0% | 78.0% | +14.0% |

---

## Project Structure

```
problem 5/
├── README.md                 # This file
├── .env                      # API keys (git-ignored)
├── Dataset/
│   └── emotion_dataset_v2.jsonl
├── gemini/
│   ├── README.md
│   ├── gemini-2.5-flash.ipynb
│   └── [output CSVs + JSONLs]
├── GPT/
│   ├── README.md
│   ├── gpt-4o-mini.ipynb
│   └── [output CSVs + JSONLs]
├── Fanar/
│   ├── README.md
│   ├── Fanar-C-2-27B.ipynb
│   └── [output CSVs + JSONLs]
├── ALLaM/
│   ├── README.md
│   ├── ALLaM-7B-Instruct-preview.ipynb
│   └── kernel-metadata.json
└── Jais/
    ├── README.md
    ├── Jais-2-8B-Chat.ipynb
    └── kernel-metadata.json
```

---

## How to Reproduce

### Local Models (Gemini, GPT, Fanar)

1. Create `.env` in the `problem 5/` directory:
   ```
   GEMINI_API_KEY=your_key
   OPENAI_API_KEY=your_key
   FANAR_API_KEY=your_key
   ```

2. Install dependencies:
   ```bash
   pip install openai google-genai pandas tqdm python-dotenv scikit-learn
   ```

3. Run each notebook:
   ```bash
   python -m jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=1800 "gemini/gemini-2.5-flash.ipynb"
   python -m jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=1800 "GPT/gpt-4o-mini.ipynb"
   python -m jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=1800 "Fanar/Fanar-C-2-27B.ipynb"
   ```

### Kaggle Models (ALLaM, Jais)

See the individual README files in the `ALLaM/` and `Jais/` directories for step-by-step Kaggle instructions.

---

## Output Files (Per Model)

Each model generates:
| File | Content |
|------|---------|
| `<MODEL>_zero_shot_results.jsonl` | Raw predictions (zero-shot) |
| `<MODEL>_few_shot_results.jsonl` | Raw predictions (few-shot) |
| `1_<MODEL>_Overview_Summary.csv` | Overall accuracy & F1 summary |
| `2_<MODEL>_F1_Metrics_Details.csv` | Per-emotion precision/recall/F1 |
| `3_<MODEL>_Error_Analysis_Log.csv` | All misclassified sentences |
