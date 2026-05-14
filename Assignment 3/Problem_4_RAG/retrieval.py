"""
Task 2: Retrieval System
- Classical search: TF-IDF and BM25
- Semantic search: embedding-based with FAISS
- Returns top-5 results from both methods
"""

import re
import math
import numpy as np
from typing import List, Dict, Tuple, Optional
from collections import Counter


# ── Arabic tokenizer ───────────────────────────────────────────────────────────

ARABIC_STOPWORDS = {
    'في', 'من', 'إلى', 'على', 'عن', 'مع', 'هذا', 'هذه', 'ذلك', 'تلك',
    'التي', 'الذي', 'الذين', 'اللاتي', 'وهو', 'وهي', 'وهم', 'وهن',
    'كان', 'كانت', 'كانوا', 'يكون', 'تكون', 'هو', 'هي', 'هم', 'هن',
    'أن', 'إن', 'لا', 'لم', 'لن', 'ما', 'لقد', 'قد', 'قال', 'قالت',
    'أو', 'و', 'ف', 'ب', 'ل', 'ك', 'ن', 'ثم', 'حتى', 'بل',
    'هل', 'إذا', 'إذ', 'حين', 'عند', 'بعد', 'قبل', 'بين', 'فوق', 'تحت',
    'كل', 'بعض', 'غير', 'بغير', 'مما', 'منه', 'منها', 'فيه', 'فيها',
    'به', 'بها', 'له', 'لها', 'عليه', 'عليها', 'إليه', 'إليها',
    'ولا', 'ولو', 'وإن', 'وأن', 'أما', 'فأما', 'لكن', 'ولكن', 'بل',
    'إنما', 'حتى', 'كما', 'مما', 'بما', 'فما', 'وما', 'لما',
}


def normalize_arabic(text: str) -> str:
    """Normalize Arabic text for tokenization."""
    text = re.sub(r'[\u0610-\u061A\u064B-\u065F\u0670]', '', text)  # remove diacritics
    text = re.sub(r'[أإآ]', 'ا', text)
    text = re.sub(r'ة', 'ه', text)
    text = re.sub(r'ى', 'ي', text)
    return text


def tokenize_arabic(text: str, remove_stopwords: bool = True) -> List[str]:
    """Tokenize Arabic text into words."""
    text = normalize_arabic(text)
    # Keep only Arabic letters and spaces
    words = re.findall(r'[\u0600-\u06FF]+', text)
    if remove_stopwords:
        words = [w for w in words if w not in ARABIC_STOPWORDS and len(w) > 2]
    return words


# ── TF-IDF ─────────────────────────────────────────────────────────────────────

class TFIDFRetriever:
    """Classical TF-IDF retrieval."""

    def __init__(self, chunks: List[dict]):
        self.chunks = chunks
        self.tokenized_docs = []
        self.idf = {}
        self.tf_idf_matrix = []
        self._build_index()

    def _build_index(self):
        print("[TF-IDF] Building index...")
        N = len(self.chunks)

        # Tokenize all docs
        self.tokenized_docs = [
            tokenize_arabic(c['text']) for c in self.chunks
        ]

        # Build vocabulary and DF
        df = Counter()
        for doc in self.tokenized_docs:
            unique_terms = set(doc)
            for term in unique_terms:
                df[term] += 1

        # Compute IDF
        self.idf = {
            term: math.log((N + 1) / (freq + 1)) + 1
            for term, freq in df.items()
        }

        # Compute TF-IDF vectors (as dicts for sparse representation)
        self.tf_idf_matrix = []
        for doc_tokens in self.tokenized_docs:
            tf = Counter(doc_tokens)
            total = max(len(doc_tokens), 1)
            vec = {
                term: (count / total) * self.idf.get(term, 0)
                for term, count in tf.items()
            }
            # L2 normalize
            norm = math.sqrt(sum(v ** 2 for v in vec.values())) or 1
            vec = {k: v / norm for k, v in vec.items()}
            self.tf_idf_matrix.append(vec)

        print(f"[TF-IDF] Indexed {N} docs, vocabulary size: {len(self.idf)}")

    def search(self, query: str, top_k: int = 5) -> List[Dict]:
        """Search with TF-IDF cosine similarity."""
        query_tokens = tokenize_arabic(query)
        if not query_tokens:
            return []

        # Build query vector
        query_tf = Counter(query_tokens)
        total = len(query_tokens)
        query_vec = {
            term: (count / total) * self.idf.get(term, 0)
            for term, count in query_tf.items()
            if term in self.idf
        }
        norm = math.sqrt(sum(v ** 2 for v in query_vec.values())) or 1
        query_vec = {k: v / norm for k, v in query_vec.items()}

        # Compute cosine similarity
        scores = []
        for i, doc_vec in enumerate(self.tf_idf_matrix):
            score = sum(
                query_vec.get(term, 0) * doc_vec.get(term, 0)
                for term in query_vec
            )
            scores.append((i, score))

        scores.sort(key=lambda x: x[1], reverse=True)
        top = scores[:top_k]

        results = []
        for rank, (idx, score) in enumerate(top, 1):
            results.append({
                'rank': rank,
                'score': round(score, 4),
                'chunk_id': self.chunks[idx]['id'],
                'text': self.chunks[idx]['text'],
                'method': 'TF-IDF',
            })
        return results


# ── BM25 ───────────────────────────────────────────────────────────────────────

class BM25Retriever:
    """BM25 (Okapi BM25) retrieval."""

    def __init__(self, chunks: List[dict], k1: float = 1.5, b: float = 0.75):
        self.chunks = chunks
        self.k1 = k1
        self.b = b
        self.tokenized_docs = []
        self.idf = {}
        self.dl = []       # document lengths
        self.avgdl = 0
        self._build_index()

    def _build_index(self):
        print("[BM25] Building index...")
        N = len(self.chunks)
        self.tokenized_docs = [
            tokenize_arabic(c['text']) for c in self.chunks
        ]
        self.dl = [len(d) for d in self.tokenized_docs]
        self.avgdl = sum(self.dl) / max(N, 1)

        # DF counts
        df = Counter()
        for doc in self.tokenized_docs:
            for term in set(doc):
                df[term] += 1

        # BM25 IDF: log((N - df + 0.5) / (df + 0.5) + 1)
        self.idf = {
            term: math.log((N - freq + 0.5) / (freq + 0.5) + 1)
            for term, freq in df.items()
        }
        print(f"[BM25] Indexed {N} docs, vocab: {len(self.idf)}")

    def search(self, query: str, top_k: int = 5) -> List[Dict]:
        """BM25 search."""
        query_tokens = tokenize_arabic(query)
        if not query_tokens:
            return []

        scores = []
        for i, doc_tokens in enumerate(self.tokenized_docs):
            tf = Counter(doc_tokens)
            dl = self.dl[i]
            score = 0.0
            for term in set(query_tokens):
                if term not in self.idf:
                    continue
                f = tf.get(term, 0)
                idf = self.idf[term]
                numerator = f * (self.k1 + 1)
                denominator = f + self.k1 * (1 - self.b + self.b * dl / self.avgdl)
                score += idf * (numerator / denominator)
            scores.append((i, score))

        scores.sort(key=lambda x: x[1], reverse=True)

        results = []
        for rank, (idx, score) in enumerate(scores[:top_k], 1):
            results.append({
                'rank': rank,
                'score': round(score, 4),
                'chunk_id': self.chunks[idx]['id'],
                'text': self.chunks[idx]['text'],
                'method': 'BM25',
            })
        return results


# ── Semantic Search ────────────────────────────────────────────────────────────

class SemanticRetriever:
    """Embedding-based semantic search using FAISS."""

    def __init__(self, index, chunks: List[dict], embed_model):
        self.index = index
        self.chunks = chunks
        self.model = embed_model

    def search(self, query: str, top_k: int = 5) -> List[Dict]:
        """Search using embedding similarity."""
        # Encode query
        query_emb = self.model.encode(
            [query],
            normalize_embeddings=True,
            show_progress_bar=False,
        ).astype(np.float32)

        # FAISS search
        scores, indices = self.index.search(query_emb, top_k)
        scores = scores[0]
        indices = indices[0]

        results = []
        for rank, (idx, score) in enumerate(zip(indices, scores), 1):
            if idx == -1:
                continue
            results.append({
                'rank': rank,
                'score': round(float(score), 4),
                'chunk_id': self.chunks[idx]['id'],
                'text': self.chunks[idx]['text'],
                'method': 'Semantic',
            })
        return results


# ── Unified Retrieval Interface ────────────────────────────────────────────────

class RetrievalSystem:
    """Unified interface for all retrieval methods."""

    def __init__(self, chunks: List[dict], faiss_index, embed_model):
        self.chunks = chunks
        self.tfidf = TFIDFRetriever(chunks)
        self.bm25 = BM25Retriever(chunks)
        self.semantic = SemanticRetriever(faiss_index, chunks, embed_model)

    def search_all(self, query: str, top_k: int = 5) -> Dict:
        """Run all retrieval methods and return results."""
        return {
            'query': query,
            'tfidf': self.tfidf.search(query, top_k),
            'bm25': self.bm25.search(query, top_k),
            'semantic': self.semantic.search(query, top_k),
        }

    def format_results(self, results: Dict) -> str:
        """Format results for display."""
        lines = [f"\n{'='*70}"]
        lines.append(f"الاستعلام: {results['query']}")
        lines.append('=' * 70)

        for method_key, method_name in [('tfidf', 'TF-IDF'), ('bm25', 'BM25'), ('semantic', 'بحث دلالي')]:
            lines.append(f"\n── {method_name} ──")
            for r in results[method_key]:
                lines.append(f"  [{r['rank']}] Score={r['score']:.4f}")
                lines.append(f"      {r['text'][:150]}...")
                lines.append("")

        return '\n'.join(lines)


if __name__ == "__main__":
    # Quick test
    import sys
    import os
    sys.path.insert(0, os.path.dirname(__file__))
    from book_preparation import prepare_book

    index, chunks, embeddings, model = prepare_book()
    system = RetrievalSystem(chunks, index, model)

    test_queries = [
        "ما هو موقف الغزالي من الفلسفة؟",
        "ما رأي الغزالي في السببية؟",
    ]

    for q in test_queries:
        results = system.search_all(q)
        print(system.format_results(results))
