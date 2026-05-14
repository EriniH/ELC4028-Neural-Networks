# Arabic RAG System for *Falsafat Al-Ghazali*

A Retrieval-Augmented Generation (RAG) project for Arabic question answering over the book **فلسفة الغزالي** by **عباس محمود العقاد**.

The system combines:
- classical retrieval with **TF-IDF** and **BM25**
- semantic retrieval with **Sentence Transformers + FAISS**
- Arabic answer generation with a small open-source **LLM**
- evaluation scripts and a desktop GUI

## Features

- Arabic text preprocessing and chunking
- Multilingual sentence embeddings for semantic search
- FAISS index for fast vector retrieval
- Side-by-side comparison of **RAG** vs **LLM-only** answers
- Batch evaluation over 10 Arabic queries
- PySimpleGUI desktop interface

## Project Structure

- `main.py` - command-line entry point for preparing, evaluating, or querying the system
- `book_preparation.py` - loads the book, chunks text, generates embeddings, and builds the FAISS index
- `retrieval.py` - TF-IDF, BM25, and semantic retrievers
- `rag.py` - LLM wrapper and RAG pipeline
- `evaluation.py` - evaluation queries, result saving, and report generation
- `interface_gui.py` - desktop GUI built with PySimpleGUI
- `data/ghazali_philosophy.txt` - source book text
- `outputs/` and `outputs_Silma/` - generated chunks, embeddings, FAISS indexes, and reports

## Requirements

- Python 3.10 or later is recommended
- A working internet connection is needed the first time models are downloaded from Hugging Face
- GPU is optional, but helps with the LLM

Install dependencies:

```bash
pip install -r requirements.txt
```

## Quick Start

### 1. Prepare the book index

Build the chunked corpus, embeddings, and FAISS index:

```bash
python main.py --mode prepare
```

### 2. Run the full evaluation

Execute the retrieval and RAG evaluation pipeline and generate a report:

```bash
python main.py --mode evaluate
```

This saves timestamped results and an `evaluation_report.txt` file in the configured output directory.

### 3. Run a single query

You can test a one-off Arabic query and save the retrieval results to JSON:

```bash
python main.py --query "ما موقف الغزالي من السببية؟"
```

## Desktop GUI

Launch the PySimpleGUI interface with:

```bash
python interface_gui.py
```

The GUI includes:
- a retrieval tab for TF-IDF, BM25, and semantic search
- a RAG tab to compare RAG and LLM-only answers
- a system info tab describing preprocessing and model choices

## Output Files

The pipeline produces files such as:
- `chunks.json`
- `embeddings.npy`
- `faiss.index`
- `retrieval_results_*.json`
- `rag_results_*.json`
- `evaluation_report.txt`

The current main entry point uses `outputs_Silma/` as its output directory. The GUI script uses `outputs/` by default.

## Model Choices

The current configuration uses:
- embedding model: `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`
- LLM: `silma-ai/SILMA-Kashif-2B-Instruct-v1.0`

These can be changed in `main.py` and `interface_gui.py` if needed.

## Notes

- The first run may take time because it prepares the index and downloads model weights.
- If you want to force a rebuild of the book index, run the prepare mode again after deleting the saved output files or by using the preparation function directly.
- The project is focused on Arabic information retrieval and question answering over a single source text.

## Example Questions

- من هو الغزالي وما لقبه؟
- ما تعريف الغزالي للتصوف؟
- ما رأي الغزالي في قدم العالم؟
- ما موقف الغزالي من السببية؟

## License

No explicit license file is included in this workspace.
