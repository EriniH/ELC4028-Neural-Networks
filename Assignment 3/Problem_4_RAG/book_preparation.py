"""
Task 1: Book Preparation for Search
- Load and preprocess Arabic text
- Split into 2-4 sentence paragraphs
- Generate embeddings using multilingual sentence transformer
- Index with FAISS
"""

import re
import os
import json
import numpy as np
from pathlib import Path
from typing import List, Tuple

# ── Arabic text utilities ──────────────────────────────────────────────────────

def clean_arabic_text(text: str) -> str:
    """Normalize and clean Arabic text."""
    # Remove diacritics (tashkeel)
    diacritics = re.compile(r'[\u0610-\u061A\u064B-\u065F\u0670\u06D6-\u06DC\u06DF-\u06E4\u06E7\u06E8\u06EA-\u06ED]')
    text = diacritics.sub('', text)

    # Normalize Arabic letters
    text = re.sub(r'[أإآ]', 'ا', text)   # normalize alef
    text = re.sub(r'ة', 'ه', text)        # normalize ta marbuta
    text = re.sub(r'ى', 'ي', text)        # normalize alef maqsura

    # Remove extra whitespace and non-Arabic/punctuation chars
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    return text


def split_into_sentences(text: str) -> List[str]:
    """Split Arabic text into sentences."""
    # Arabic sentence endings: period, question mark, exclamation + Arabic-specific
    sentence_endings = re.compile(r'(?<=[.؟!،])\s+(?=[^\s])|(?<=\n)\s*(?=\S)')
    # Also split on Arabic full stop and other markers
    text = re.sub(r'[•]{3,}', '\n', text)   # replace ••• separators
    sentences = re.split(r'(?<=[.؟!\n])\s+', text)
    sentences = [s.strip() for s in sentences if s.strip() and len(s.strip()) > 15]
    return sentences


def chunk_text(text: str, min_sentences: int = 2, max_sentences: int = 4) -> List[dict]:
    """
    Split text into overlapping chunks of 2-4 sentences each.
    Returns list of dicts with chunk text, sentence indices, and metadata.
    """
    sentences = split_into_sentences(text)
    chunks = []
    i = 0
    chunk_id = 0

    while i < len(sentences):
        # Take 2-4 sentences per chunk
        end = min(i + max_sentences, len(sentences))
        chunk_sentences = sentences[i:end]

        # Ensure minimum length
        chunk_text_raw = ' '.join(chunk_sentences)
        if len(chunk_text_raw) < 30:
            i += 1
            continue

        chunk = {
            'id': chunk_id,
            'text': chunk_text_raw,
            'text_clean': clean_arabic_text(chunk_text_raw),
            'sentence_start': i,
            'sentence_end': end - 1,
            'num_sentences': len(chunk_sentences),
            'char_count': len(chunk_text_raw),
        }
        chunks.append(chunk)
        chunk_id += 1

        # Slide by min_sentences for slight overlap
        i += min_sentences

    return chunks


# ── Embedding & Indexing ───────────────────────────────────────────────────────

def load_embedding_model(model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"):
    """Load a multilingual sentence embedding model."""
    from sentence_transformers import SentenceTransformer
    print(f"[INFO] Loading embedding model: {model_name}")
    model = SentenceTransformer(model_name)
    return model


def generate_embeddings(chunks: List[dict], model, batch_size: int = 32) -> np.ndarray:
    """Generate embeddings for all chunks."""
    texts = [c['text'] for c in chunks]
    print(f"[INFO] Generating embeddings for {len(texts)} chunks...")
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,   # L2-normalize for cosine similarity via dot product
    )
    return embeddings.astype(np.float32)


def build_faiss_index(embeddings: np.ndarray):
    """Build a FAISS flat inner-product index (cosine sim with normalized vecs)."""
    import faiss
    dim = embeddings.shape[1]
    print(f"[INFO] Building FAISS index (dim={dim}, n={len(embeddings)})...")
    index = faiss.IndexFlatIP(dim)   # Inner Product = cosine sim for normalized vecs
    index.add(embeddings)
    print(f"[INFO] Index contains {index.ntotal} vectors.")
    return index


# ── Persistence ────────────────────────────────────────────────────────────────

def save_index(index, chunks: List[dict], embeddings: np.ndarray, output_dir: str = "outputs"):
    """Save FAISS index, chunks, and embeddings to disk."""
    import faiss
    os.makedirs(output_dir, exist_ok=True)
    faiss.write_index(index, os.path.join(output_dir, "faiss.index"))
    with open(os.path.join(output_dir, "chunks.json"), "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)
    np.save(os.path.join(output_dir, "embeddings.npy"), embeddings)
    print(f"[INFO] Saved index and chunks to {output_dir}/")


def load_index(output_dir: str = "outputs"):
    """Load saved FAISS index and chunks."""
    import faiss
    index = faiss.read_index(os.path.join(output_dir, "faiss.index"))
    with open(os.path.join(output_dir, "chunks.json"), encoding="utf-8") as f:
        chunks = json.load(f)
    embeddings = np.load(os.path.join(output_dir, "embeddings.npy"))
    print(f"[INFO] Loaded index with {index.ntotal} vectors and {len(chunks)} chunks.")
    return index, chunks, embeddings


# ── Main pipeline ──────────────────────────────────────────────────────────────

def prepare_book(
    book_path: str = "data/ghazali_philosophy.txt",
    output_dir: str = "outputs",
    embedding_model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    force_rebuild: bool = False,
) -> Tuple:
    """Full pipeline: load → chunk → embed → index."""

    index_path = os.path.join(output_dir, "faiss.index")

    if not force_rebuild and os.path.exists(index_path):
        print("[INFO] Found existing index. Loading...")
        index, chunks, embeddings = load_index(output_dir)
        model = load_embedding_model(embedding_model_name)
        return index, chunks, embeddings, model

    # Load book
    print(f"[INFO] Loading book from {book_path}")
    with open(book_path, encoding="utf-8") as f:
        raw_text = f.read()

    # Remove header lines (title, author)
    lines = raw_text.strip().split('\n')
    # Skip the first few header lines (title, author, lecture header)
    content_start = 0
    for i, line in enumerate(lines):
        if len(line.strip()) > 40:   # first substantive paragraph
            content_start = i
            break
    text = '\n'.join(lines[content_start:])

    print(f"[INFO] Book loaded: {len(text)} chars")

    # Chunk
    chunks = chunk_text(text, min_sentences=2, max_sentences=4)
    print(f"[INFO] Created {len(chunks)} chunks")
    for c in chunks[:3]:
        print(f"  Chunk {c['id']}: {c['num_sentences']} sentences, {c['char_count']} chars")
        print(f"    Preview: {c['text'][:80]}...")

    # Embed
    model = load_embedding_model(embedding_model_name)
    embeddings = generate_embeddings(chunks, model)

    # Index
    index = build_faiss_index(embeddings)

    # Save
    save_index(index, chunks, embeddings, output_dir)

    return index, chunks, embeddings, model


if __name__ == "__main__":
    import sys
    book_path = sys.argv[1] if len(sys.argv) > 1 else "data/ghazali_philosophy.txt"
    prepare_book(book_path, force_rebuild=True)
