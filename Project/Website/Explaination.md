## Table of Contents

1. [Why Word Embeddings?](#1-why-word-embeddings)
2. [Similarity in Vector Space](#2-similarity-in-vector-space)
3. [Problems with Naive Word Representations](#3-problems-with-naive-word-representations)
4. [What Are Embeddings?](#4-what-are-embeddings)
5. [Embedding Tables and Learning](#5-embedding-tables-and-learning)
6. [Pre-trained Embeddings and Word2Vec](#6-pre-trained-embeddings-and-word2vec)
7. [GloVe: Global Vectors for Word Representation](#7-glove-global-vectors-for-word-representation)
8. [Advanced Topics: OOV, FastText, ELMo, and Beyond](#8-advanced-topics-oov-fasttext-elmo-and-beyond)

---

## 1. Why Word Embeddings?

### The Core Problem: NLP vs. Computer Vision

Machine learning excels with structured, numerical data. Computer Vision (CV) handles this naturally — pixel values are already numerical (0–255), continuous, and proximity in value has inherent meaning (near pixel values = near intensity). No special transformation is needed before feeding pixels into a model.

Natural Language Processing (NLP) is fundamentally different. Text is **categorical** — words and phonemes are symbols, and their index in a vocabulary is arbitrary. A word assigned index `123` is not "more" or "less" than a word assigned index `1094`. There is no inherent mathematical relationship between word indices.

### The Challenge of Encoding Language

For NLP, the encoding system must:
- Convert discrete symbols into a computer-processable form
- Incorporate similarity operations — **similar words must have similar representations**
- Preserve enough information for sequential and contextual relationships

Without this, models cannot learn anything meaningful from raw categorical tokens. Knowing an actor's ID number (123 vs. 1094) tells you nothing about them; knowing their age and number of movies gives you actual comparable features. The same principle applies to words.

### Why Not Just Use Indices?

Raw vocabulary indices are arbitrary integers. They carry no semantic information. If "hotel" is index 500 and "motel" is index 3820, a model treating these as numbers would consider them completely unrelated — when they are in fact highly similar concepts. The solution is to find a **dense, continuous representation** that encodes meaning.

---

## 2. Similarity in Vector Space

### Vectors Enable Meaningful Comparison

Once words are represented as vectors (lists of numbers = features), we can apply standard mathematical similarity measures:

- **Dot Product:** `dot(A, B)` — captures alignment of two vectors
- **Cosine Similarity:** `dot(A, B) / (|A| × |B|)` — normalized dot product, measures angle between vectors (1 = identical direction, 0 = orthogonal, -1 = opposite)
- **Euclidean Distance:** `||A - B||` — straight-line distance in vector space

All three are related views of the same underlying concept: how "close" two vectors are.

### Neurons as Feature Detectors

In neural networks, a neuron computes `dot(W, X)` — a dot product between the weight vector W and the input X. This is mathematically a **similarity measure**. Neurons with weights that resemble a particular pattern will fire strongly when that pattern appears in the input. This is why the dot product is central to all of deep learning: neurons detect features by measuring similarity.

### The Goal of Word Embeddings

The objective is to learn a **mapping** from the discrete categorical space (word indices) to a continuous **vector space**, where:

> Words with similar meanings are located close to each other in the vector space.

This is the fundamental promise of word embeddings.

---

## 3. Problems with Naive Word Representations

### Bag-of-Words (BoW) Approaches

Early NLP relied on vectorizing text through:
- **Binary (One-Hot Encoding / OHE):** Each word becomes a binary vector with a 1 in its position, 0s everywhere else
- **Count:** Word frequency in a document
- **TF-IDF:** Term frequency weighted by inverse document frequency

While BoW representations can work at the sentence level (sentences with similar words share 1s at the same positions, so their dot product is non-zero), they have serious fundamental flaws at the word level.

### The Sparsity Problem

One-Hot Encoding (OHE) creates vectors of length equal to the entire vocabulary size (typically 10,000–50,000 words). A single word is a vector with exactly one 1 and thousands of 0s. This leads to:

- **High dimensionality** — massive numbers of weights in any downstream model
- **Sparsity** — almost all values are 0, contributing nothing useful
- **Overfitting** — too many parameters relative to useful information

### The Similarity Problem (Most Critical)

The fatal flaw of OHE is that **every pair of words is equally dissimilar**. Mathematically:

```
dot(OHE("hotel"), OHE("motel")) = 0
dot(OHE("hotel"), OHE("the"))   = 0
```

Both dot products are zero — perfectly orthogonal vectors — even though "hotel" and "motel" are semantically nearly identical. OHE encodes zero similarity information, making it useless for tasks requiring semantic understanding.

### Why BoW at Sentence Level Sometimes Works

At the sentence (not word) level, BoW builds one vector per sentence. Sentences sharing many words will share 1s at the same positions, so their dot products are non-zero. This explains why BoW-based models can achieve reasonable accuracy (e.g., 86% on IMDB sentiment). However, they still suffer from sparsity and lack sequence/context information.

---

## 4. What Are Embeddings?

### The Definition

An **Embedding** is a learned mapping from a discrete/categorical space to a continuous vector space:

```
Embedding: Discrete Symbol → Dense Vector
```

Instead of a vocabulary index (integer), each word becomes a **dense vector** of fixed size `emb_sz` (the embedding dimension), where the values encode semantic information.

### The Embedding Matrix

An embedding is implemented as a matrix (called the **Embedding Matrix** or **Look-Up Table, LUT**):

```
          E1     E2     E3
Word_0   W01    W02    W03
Word_1   W11    W12    W13
Word_2   W21    W22    W23
Word_3   W31    W32    W33
```

- **Rows** = words in the vocabulary
- **Columns** = latent factors / features (size = `emb_sz`)
- **Each row** is the embedding vector for that word

### How the Look-Up Works

Looking up a word's embedding is mathematically a dot product:

```
Embedding(word_i) = OHE(word_i) · W
```

The OHE selects exactly the row corresponding to word_i from the matrix. This is equivalent to a lookup table — O(1) access time in practice.

### Who Decides the Columns?

Two approaches:
1. **Manual / Traditional ML:** Manually designed features — Part-of-Speech tags, WordNet relations, co-occurrence counts. Requires expert knowledge and is labor-intensive.
2. **Deep Learning:** Simply choose `emb_sz` (a hyperparameter) and let the network learn the values through backpropagation. The network automatically discovers what features are useful.

### Embeddings for Structured (Tabular) Data

Embeddings are not just for NLP. In structured/tabular data, **numerical variables** already encode similarity (nearby values → nearby representations via weight scaling). But **categorical variables** (like "Material Type" with values 0, 1, 2, 3) have the same problem as word indices — the numbers are arbitrary. Applying an embedding table to categorical features in structured data is standard practice in modern deep learning for tabular data, enabling hierarchical architectures that handle mixed data types.

---

## 5. Embedding Tables and Learning

### Training Embeddings End-to-End

Since embedding values are weight matrices, they participate in standard gradient-based learning:

1. Initialize embedding matrix with random values
2. Pass word indices through the embedding layer
3. Compute a task-specific loss (e.g., cross-entropy for classification)
4. Backpropagate gradients all the way to the embedding weights
5. The optimizer updates the embeddings alongside all other weights

The result: embeddings that are optimized for the target task.

### Shared Weights Across a Sentence

When processing a sentence, each word is passed through an embedding layer. Critically, **all embedding layers share the same weight matrix W**. Each word index selects a different row from this shared matrix. During backpropagation, only the rows corresponding to words that appeared in the current training example are updated — other rows receive no gradient.

### Two Learning Strategies

1. **End-to-End from Current Task:** Train embeddings from scratch on your specific dataset. Works well with sufficient data.
2. **Pre-trained Embeddings:** Transfer embeddings trained on a large generic dataset to your task. This is the approach taken by Word2Vec, GloVe, and ELMo.

The choice follows standard **Transfer Learning (TL) rules**:
- Large, similar dataset → Reuse + Fine-tune
- Small, similar dataset → Reuse, No fine-tune
- Large, different dataset → Fine-tune from scratch
- Small, different dataset → Minimal reuse, careful fine-tuning

---

## 6. Pre-trained Embeddings and Word2Vec

### The Transfer Learning Approach

Pre-training embeddings on a large generic task and transferring them to downstream NLP tasks provides significant advantages. The source task must:
- Be **generic** — applicable to most NLP tasks
- Be trained on **as much data as possible** (e.g., Wikipedia, Common Crawl)
- Be **self-supervised** — require no manual annotation

**Language Modeling** (predicting words from context) satisfies all three criteria.

### Contextualized Word Representations

The key insight, articulated by J.R. Firth (1957):

> "A word is defined by the company that it keeps."

Words take their meaning from context — the surrounding words in a window. Training a model to understand context produces vectors that capture semantic relationships.

### Word2Vec (Mikolov et al., 2013)

Word2Vec is a family of three related algorithms for learning word embeddings from context:

#### 1. CBOW — Continuous Bag of Words
- **Task:** Predict the **center word** given the surrounding **context words**
- Input: all context words → Output: center word
- Architecture: Embedding layers (shared W) → Average/Sum → Dense → Softmax over vocabulary
- The Softmax output size = `vocab_sz` (10k–50k), making it computationally expensive

#### 2. Skip-Gram (SG)
- **Task:** Predict the surrounding **context words** given the **center word**
- Input: center word → Output: context words
- The dataset is built by sliding a window over text and creating (center, context) pairs
- Same computational cost problem as CBOW: Softmax over full vocabulary

#### 3. Skip-Gram with Negative Sampling (SGNS) — Most Widely Used
- **Reformulates the task:** Instead of predicting which word from the full vocabulary, predict whether a (center, context) pair is **valid (1) or invalid (0)**
- **Negative samples:** Random word pairs are introduced and labeled as 0 (invalid)
- **Loss:** Binary Cross-Entropy (BCE) instead of Softmax Cross-Entropy
- **Architecture:** Center word embedding · Context word embedding → Dot product → Sigmoid → Binary output
- **Key benefit:** Output size is 1 (sigmoid), not `vocab_sz` (softmax) — dramatically reduces parameters and prevents overfitting

##### Why Separate Embeddings for Center and Context?
SGNS maintains two separate embedding matrices (W for center words, W̃ for context words):
- Negative samples are not real context — using a single matrix would corrupt embeddings with "fake" co-occurrences
- At inference time, context embeddings are discarded; only the center word embeddings are used
- Optionally, sum W + W̃ for a small performance boost (as done in GloVe)

##### Training Process (SGNS Step by Step)
1. Slide a window of size `window_sz` (default=5) over the corpus
2. Form positive pairs: (center_word, context_word) → label 1
3. Sample `n_negative` (default=5) random words → form negative pairs → label 0
4. Forward pass: look up both embeddings, dot them, apply sigmoid
5. Compute BCE loss
6. Backpropagate: update only the embedding rows that were selected (for center word and sampled context/negative words)

##### Hyperparameters
- **Window size:** Larger → broader thematic context; Smaller → syntactic/closer relationships. Default = 5.
- **Negative samples:** Default = 5. Slightly larger than balanced (50/50) because random negatives might accidentally be valid.

### Remarkable Properties of Learned Embeddings

After training, embeddings capture semantic relationships through vector arithmetic:

```
vector("King") - vector("Man") + vector("Woman") ≈ vector("Queen")
vector("Paris") - vector("France") + vector("Italy") ≈ vector("Rome")
```

These arithmetic relationships emerge entirely from the training objective of predicting word co-occurrence — the model was never explicitly told about gender or geography.

### Applications Beyond NLP

Word embedding techniques extend naturally to **recommender systems**. Items (products, movies, songs) can be treated as "words" and user interaction sequences as "sentences." Training Word2Vec on these sequences produces item embeddings where similar items are close in vector space — enabling nearest-neighbor recommendations.

---

## 7. GloVe: Global Vectors for Word Representation

### Local vs. Global Context

Word2Vec uses a **local** sliding window — it only considers words within a fixed window around the target word. This ignores **global** document-wide statistics.

**Co-occurrence matrices** capture global statistics: a matrix where entry (i, j) counts how many times word i and word j appear in the same document (or window) across the entire corpus. This gives a richer, corpus-wide view of word relationships.

### Problems with Raw Co-occurrence Matrices

Like OHE and BoW, raw co-occurrence matrices are:
- **Huge:** `vocab_sz × vocab_sz` (e.g., 50k × 50k = 2.5 billion entries)
- **Sparse:** Most word pairs never co-occur
- **High-dimensional:** Too large for direct use in models

### Latent Factor Decomposition

The insight: instead of storing raw co-occurrence counts, find **latent factors** (dimensions) that explain the co-occurrence patterns. Words cluster by these factors:
- A word like "President" scores high on a "political" latent factor
- A word like "Game" scores high on a "sports" latent factor
- A word like "Official" scores moderately on both

If we find these latent representations, we have meaningful, dense word vectors — embeddings!

### Singular Value Decomposition (SVD)

SVD is a classical linear algebra technique to factorize a matrix A into components U, Σ, V:

```
A ≈ U · Σ · Vᵀ
```

Where U and V contain the latent factor representations. However, SVD has critical drawbacks:
- **Computational cost:** O(mn²) where n = vocab_sz — scales quadratically with vocabulary size
- **Sparsity issues:** Many zero entries in the co-occurrence matrix cause instability
- **Weighted Matrix Factorization** is used to handle unobserved (zero) entries, treating them separately from truly zero co-occurrences

### Gradient-Based Optimization for GloVe

Instead of SVD, GloVe optimizes the latent factors using gradient descent:

1. Assign each word a random latent vector U (and a context vector V)
2. For each (word_i, word_j) pair with co-occurrence count X_ij:
   - Compute predicted log-count: `dot(U_i, V_j) + bias_i + bias_j`
   - Compare to actual log-count: `log(X_ij)`
   - Compute weighted loss (common pairs weighted higher)
3. Backpropagate and update U and V

The **bias terms** capture each word's overall frequency in the language, independent of specific co-occurrences.

### GloVe Loss Function

```
L = Σ f(X_ij) · (dot(U_i, V_j) + b_i + b̃_j - log(X_ij))²
```

Where `f(X_ij)` is a weighting function that down-weights very frequent co-occurrences (stop words) and gives zero weight to unobserved pairs.

### Why Two Matrices (U and V)?

GloVe (like SGNS) maintains two embedding matrices. Since the co-occurrence matrix X is symmetric, U and V differ only in random initialization. GloVe recommends using the **sum W + W̃** as the final word vector, which gives a small but consistent performance boost — particularly on semantic analogy tasks.

### GloVe vs. Word2Vec

| Property | Word2Vec (SGNS) | GloVe |
|---|---|---|
| Context type | Local (sliding window) | Global (corpus-wide co-occurrence) |
| Training objective | Predict valid/invalid pairs | Reconstruct co-occurrence counts |
| Optimization | SGD on pairs | Weighted least squares |
| Scalability | Very fast | Moderate (SVD slow; gradient GloVe faster) |
| Performance | Strong | Comparable; stronger on semantic analogies |

Both are considered count-based/prediction-based hybrid approaches in practice, and both produce high-quality embeddings used across NLP.

---

## 8. Advanced Topics: OOV, FastText, ELMo, and Beyond

### The Out-of-Vocabulary (OOV) Problem

Standard word-level embeddings (Word2Vec, GloVe) maintain a fixed vocabulary. Any word not seen during training — new words, rare words, typos, proper nouns, domain-specific terms — is **Out-of-Vocabulary (OOV)** and has no embedding. This is a significant practical limitation.

### FastText: Character N-Gram Embeddings

**FastText** (Facebook AI Research) addresses OOV by representing words as sums of their constituent **character n-grams**:

- Instead of embedding "correctly" as a whole word, FastText embeds its n-grams: `cor`, `orr`, `rre`, `rec`, `ect`, `ctl`, `tly`, etc.
- The word vector = sum of its n-gram vectors
- This approach is a middle ground between character-level and word-level representation

**Benefits:**
- Handles OOV words — even unseen words can be approximated by their character n-grams
- Captures morphological similarity — "run", "running", "runner" share n-grams and thus similar embeddings
- Better for morphologically rich languages

**Stems vs. Morphology:** An alternative strategy is to keep stems (root forms) in the vocabulary instead of all inflections, reducing vocabulary size while retaining core semantics.

### ELMo: Contextualized Embeddings

**ELMo (Embeddings from Language Models)** represents a paradigm shift: instead of a single static vector per word, ELMo produces **context-dependent embeddings**. The same word gets different vectors depending on its context sentence.

**Key differences from Word2Vec/GloVe:**
- Uses a **bidirectional LSTM (BiLSTM)** language model as the encoder
- Processes input as a **sequence**, capturing ordering and context
- Combines **character-level and word-level** representations — a sub-word approach
- The embedding for each word is a learned combination of all BiLSTM layer outputs

**Sub-word representation in ELMo:**
1. Characters have entries in the lookup table
2. Sub-words are composed from characters
3. Complete words are in the table when seen; unknown words are decomposed into sub-words or characters
4. This hierarchy dramatically reduces OOV impact

**Why contextual matters:** The word "bank" means very different things in "river bank" vs. "bank account." Static embeddings assign one fixed vector; ELMo assigns different vectors depending on the surrounding words.

### The Embedding Ecosystem at a Glance

| Method | Representation | Context | OOV Handling | Key Idea |
|---|---|---|---|---|
| OHE | Sparse, high-dim | None | No | Baseline: word = binary position |
| BoW/TF-IDF | Sparse, sentence-level | Bag | No | Word frequency features |
| Word2Vec | Dense, static | Local window | No | Predict context from word |
| GloVe | Dense, static | Global corpus | No | Factorize co-occurrence matrix |
| FastText | Dense, static | Local window | Yes (n-grams) | Sub-word character n-grams |
| ELMo | Dense, contextual | Full sentence | Yes (chars) | BiLSTM language model |
| BERT/Transformers | Dense, contextual | Full document | Tokenization | Self-attention over full context |

### Practical Guidelines for Using Embeddings

**When to train from scratch:**
- Large domain-specific corpus available
- Task vocabulary differs substantially from general language (e.g., medical, legal, code)

**When to use pre-trained embeddings:**
- Small dataset
- General language domain
- Reduces training time and data requirements significantly

**Transfer Learning Quadrant:**
- **Small data, similar domain:** Freeze pre-trained embeddings, train only task layers
- **Large data, similar domain:** Fine-tune pre-trained embeddings with task layers
- **Small data, different domain:** Use only lower-layer embeddings (more general), freeze them
- **Large data, different domain:** Fine-tune from scratch, or train entirely new embeddings

### Code Resources Referenced

The lecture includes practical implementations covering:
- Keras Embedding layers (end-to-end training)
- BoW vectors model with IMDB sentiment classification
- Playing with word vectors using gensim
- TensorBoard Embedding Projector for visualization
- Loading pre-trained GloVe embeddings
- GloVe training from scratch
- FastText on IMDB classification

---

## Key Takeaways

1. **Words cannot be arbitrary indices** — similarity operations require continuous, dense vector representations.

2. **Embeddings = learned lookup tables** — a matrix where each row is a word's dense feature vector, learned via backpropagation.

3. **Word2Vec** learns embeddings by training a model to predict word context (CBOW, Skip-Gram, SGNS). SGNS is the most computationally efficient variant.

4. **GloVe** learns embeddings by factorizing global co-occurrence statistics, combining the strengths of count-based and prediction-based methods.

5. **Both Word2Vec and GloVe produce static embeddings** — one vector per word regardless of context. This limits their ability to capture polysemy (words with multiple meanings).

6. **FastText and ELMo** address OOV and context respectively, paving the way for modern Transformer-based models (BERT, GPT) that produce fully contextualized, sub-word representations at scale.

7. **Transfer Learning is central** — pre-trained embeddings on massive corpora provide strong starting points for almost any NLP task.

