type QuizQuestion = {
  q: string;
  options: string[];
  answer: number;
  explain: string;
};

const questions: QuizQuestion[] = [
  {
    q: "Why can't raw vocabulary indices (e.g., word_0 = 0, word_1 = 1) be used directly as meaningful word representations?",
    options: [
      "They are too large to store",
      "They are arbitrary integers that encode no semantic information",
      "They are not compatible with neural networks",
      "They require too much memory",
    ],
    answer: 1,
    explain:
      "Indices are arbitrary — index 123 is no more 'similar' to index 124 than to index 9999. They carry no semantic meaning, making operations like similarity measurement invalid.",
  },
  {
    q: "What is the key difference between how CV and NLP handle their 'alphabet'?",
    options: [
      "CV uses larger datasets than NLP",
      "Pixel values are continuous and inherently meaningful; word symbols are categorical and arbitrary",
      "NLP models are deeper than CV models",
      "CV doesn't require any preprocessing",
    ],
    answer: 1,
    explain:
      "Pixel values (0–255) already encode intensity — near values mean near intensity. Word symbols are arbitrary indices with no inherent numerical meaning.",
  },
  {
    q: "What does One-Hot Encoding (OHE) produce for the word 'hotel' in a vocabulary of 10,000 words?",
    options: [
      "A dense 300-dimensional vector",
      "A vector of length 10,000 with a single 1 and all other values 0",
      "A probability distribution over the vocabulary",
      "A binary hash of the characters in the word",
    ],
    answer: 1,
    explain:
      "OHE creates a binary vector of length = vocab_sz, with a 1 only at the word's index position and 0s everywhere else.",
  },
  {
    q: "What is the dot product of two different one-hot encoded word vectors?",
    options: [
      "1, because they share a vocabulary",
      "A number between 0 and 1 representing similarity",
      "0, making all word pairs equally dissimilar",
      "Undefined — dot product cannot be applied to OHE vectors",
    ],
    answer: 2,
    explain:
      "Two different OHE vectors are orthogonal — their dot product is always 0. This means 'hotel' and 'motel' appear equally unrelated as 'hotel' and 'the', which is semantically wrong.",
  },
  {
    q: "An Embedding is best defined as:",
    options: [
      "A compression algorithm for text",
      "A mapping from a discrete categorical space to a continuous vector space",
      "A tokenizer that splits words into characters",
      "A loss function for language models",
    ],
    answer: 1,
    explain:
      "Embeddings are the learned mapping from discrete symbols (word indices) to dense continuous vectors, where similar meanings map to nearby vectors.",
  },
  {
    q: "In an Embedding Matrix of shape (vocab_sz × emb_sz), what do the rows and columns represent?",
    options: [
      "Rows = latent factors; Columns = vocabulary words",
      "Rows = vocabulary words; Columns = latent factors/features",
      "Rows = training examples; Columns = model layers",
      "Rows = sentences; Columns = word positions",
    ],
    answer: 1,
    explain:
      "Each row is one word's embedding vector. The columns are the learned latent dimensions (features), whose number (emb_sz) is a hyperparameter.",
  },
  {
    q: "In deep learning, how are the values in an embedding matrix determined?",
    options: [
      "They are set manually based on linguistic rules",
      "They are computed using SVD on a co-occurrence matrix",
      "They are learned via backpropagation as part of the model's training",
      "They are copied from a dictionary",
    ],
    answer: 2,
    explain:
      "In DL, you set emb_sz as a hyperparameter and let the network learn the embedding values through gradient descent, end-to-end with the rest of the model.",
  },
  {
    q: "When processing a sentence with N words, all N embedding layers in a Word Embedding model share:",
    options: [
      "Different weight matrices, one per position",
      "The same weight matrix W",
      "The same bias vector but different weight matrices",
      "No parameters — they are frozen",
    ],
    answer: 1,
    explain:
      "All embedding layers share the same weight matrix W. Each word index selects a different row from this shared matrix. Only the selected rows receive gradient updates for a given training example.",
  },
  {
    q: "A Lookup Table (LUT) operation in an embedding layer is mathematically equivalent to:",
    options: [
      "A softmax over the full vocabulary",
      "A dot product between the OHE of the word and the embedding matrix",
      "A convolution over the word sequence",
      "A cosine similarity between two word vectors",
    ],
    answer: 1,
    explain:
      "OHE(word_i) · W selects the i-th row of W, which is exactly the embedding for word_i. This is the mathematical equivalent of a lookup table.",
  },
  {
    q: "What is the main advantage of sparse representations (BoW, OHE) at the sentence level compared to word level?",
    options: [
      "They have lower dimensionality",
      "Sentences with shared words have non-zero dot products, capturing some similarity",
      "They capture word order information",
      "They solve the OOV problem",
    ],
    answer: 1,
    explain:
      "At the sentence level, two sentences sharing words will have 1s at the same positions, making their dot product non-zero. This is why BoW sentence models can still achieve reasonable accuracy (e.g., 86% on IMDB), despite word-level orthogonality.",
  },
  {
    q: "What is the key principle behind contextual word representations as formalized by J.R. Firth (1957)?",
    options: [
      "Words are best understood through their morphological structure",
      "A word is defined by the company it keeps — its surrounding context words",
      "The frequency of a word determines its meaning",
      "Words should be represented at the character level",
    ],
    answer: 1,
    explain:
      "'A word is defined by the company that it keeps.' Training models to predict surrounding context words produces embeddings that capture semantic relationships.",
  },
  {
    q: "Which of the following is NOT one of the three Word2Vec algorithms?",
    options: [
      "Continuous Bag-of-Words (CBOW)",
      "Skip-Gram (SG)",
      "Skip-Gram with Negative Sampling (SGNS)",
      "Global Vector Factorization (GVF)",
    ],
    answer: 3,
    explain:
      "Word2Vec (Mikolov et al., 2013) comprises CBOW, Skip-Gram, and SGNS. 'Global Vector Factorization' is not a Word2Vec variant — GloVe is a separate method.",
  },
  {
    q: "In the CBOW model, what is the training objective?",
    options: [
      "Predict context words given the center word",
      "Predict the center word given the surrounding context words",
      "Determine if two words are valid co-occurrence pairs",
      "Minimize the Euclidean distance between all word vectors",
    ],
    answer: 1,
    explain:
      "CBOW: Context → Center. The model takes surrounding context words as input and predicts the center word. Skip-Gram is the reverse.",
  },
  {
    q: "Why is Skip-Gram with Negative Sampling (SGNS) preferred over standard Skip-Gram?",
    options: [
      "SGNS uses a larger training corpus",
      "SGNS replaces the Softmax over the full vocabulary with a binary sigmoid, dramatically reducing parameters",
      "SGNS uses character-level inputs instead of words",
      "SGNS requires no training data",
    ],
    answer: 1,
    explain:
      "Standard SG requires a Softmax output of size vocab_sz (10k–50k), causing overfitting. SGNS reformulates the task as binary classification (valid pair or not?), using sigmoid with output size 1 — far fewer parameters.",
  },
  {
    q: "In SGNS, what is the role of 'negative samples'?",
    options: [
      "They are misspelled words that the model learns to correct",
      "They are randomly sampled word pairs labeled as invalid (0), providing the negative class for binary classification",
      "They are stop words that are excluded from training",
      "They are words outside the vocabulary",
    ],
    answer: 1,
    explain:
      "SGNS has only positive (valid) pairs by default. Negative samples — random word pairs marked as 0 (invalid) — are introduced to create a balanced binary classification problem.",
  },
  {
    q: "Why does SGNS maintain two separate embedding matrices (W for center words and W̃ for context words)?",
    options: [
      "To double the model capacity",
      "Because context embeddings are corrupted by negative samples and should not be used at inference",
      "Because W̃ is used for visualization only",
      "There is no reason — a single matrix would work identically",
    ],
    answer: 1,
    explain:
      "Negative samples are fake context, so W̃ gets trained on invalid pairs. Using a single matrix would corrupt the inference embeddings with these bad vectors. Context embeddings are discarded at inference.",
  },
  {
    q: "The default window size in Word2Vec is 5. What does a larger window size capture?",
    options: [
      "Narrower, syntactically related context",
      "Broader, thematic/topical context",
      "Only stopwords and function words",
      "Character-level morphological patterns",
    ],
    answer: 1,
    explain:
      "Larger windows capture broader, more topical relationships. Smaller windows tend to capture tighter syntactic relationships between words.",
  },
  {
    q: "Which arithmetic relationship does a well-trained Word2Vec embedding model exhibit?",
    options: [
      "vector('King') + vector('Man') = vector('Queen')",
      "vector('King') - vector('Man') + vector('Woman') ≈ vector('Queen')",
      "vector('King') × vector('Woman') = vector('Queen')",
      "dot(vector('King'), vector('Queen')) = 0",
    ],
    answer: 1,
    explain:
      "The famous analogy: King − Man + Woman ≈ Queen. This emergent property shows that embeddings capture gender relationships as directions in vector space.",
  },
  {
    q: "What fundamental difference distinguishes GloVe from Word2Vec in terms of the context it uses?",
    options: [
      "GloVe uses character-level context; Word2Vec uses word-level",
      "GloVe uses global corpus-wide co-occurrence statistics; Word2Vec uses a local sliding window",
      "GloVe uses supervised labels; Word2Vec is unsupervised",
      "GloVe only works for English; Word2Vec is language-agnostic",
    ],
    answer: 1,
    explain:
      "Word2Vec's sliding window gives local context. GloVe builds a global co-occurrence matrix counting how often each word pair appears anywhere in the corpus, then factorizes this matrix.",
  },
  {
    q: "What are the two main problems with raw co-occurrence matrices?",
    options: [
      "They are too slow to compute and require labeled data",
      "They are huge (vocab_sz × vocab_sz) and extremely sparse",
      "They only work at the character level and lose word order",
      "They are not differentiable and cannot be optimized",
    ],
    answer: 1,
    explain:
      "A co-occurrence matrix is vocab_sz × vocab_sz (e.g., 50k × 50k = 2.5B entries), and most word pairs never co-occur, making the matrix very sparse — the same problems as OHE and BoW.",
  },
  {
    q: "In GloVe's training objective, what does the model try to minimize?",
    options: [
      "Cross-entropy between predicted and actual next words",
      "The weighted squared difference between predicted log co-occurrence and actual log co-occurrence counts",
      "The Euclidean distance between all word pairs",
      "Binary cross-entropy between valid and invalid word pairs",
    ],
    answer: 1,
    explain:
      "GloVe minimizes a weighted least-squares loss: f(X_ij) × (dot(U_i, V_j) + b_i + b̃_j − log(X_ij))², where f down-weights very frequent co-occurrences.",
  },
  {
    q: "What does GloVe recommend using as the final word vector when two matrices U and V are trained?",
    options: [
      "Only U, discarding V",
      "Only V, discarding U",
      "The average of U and V",
      "The sum W + W̃, which typically gives a small performance boost",
    ],
    answer: 3,
    explain:
      "GloVe paper (section 4.2) recommends using the sum W + W̃ as the final word vectors, noting it 'typically gives a small boost in performance, with the biggest increase in the semantic analogy task.'",
  },
  {
    q: "What is the Out-of-Vocabulary (OOV) problem in word-level embedding models?",
    options: [
      "Words that appear too frequently dominate the embeddings",
      "Words not seen during training have no embedding vector",
      "Words with multiple meanings get averaged embeddings",
      "Very long words cannot be tokenized",
    ],
    answer: 1,
    explain:
      "Standard word-level embeddings maintain a fixed vocabulary. Any word not seen at training time (new words, rare words, typos, proper nouns) has no corresponding embedding — this is the OOV problem.",
  },
  {
    q: "How does FastText address the OOV problem?",
    options: [
      "By using a much larger vocabulary covering all possible words",
      "By representing words as sums of their character n-gram vectors, so unseen words can be approximated",
      "By replacing OOV words with the closest known word",
      "By using a transformer-based architecture",
    ],
    answer: 1,
    explain:
      "FastText breaks each word into character n-grams and sums their vectors. Even a completely unseen word can be represented using its constituent n-gram vectors, which were learned during training.",
  },
  {
    q: "What is the key advancement of ELMo over Word2Vec and GloVe?",
    options: [
      "ELMo is faster to train",
      "ELMo produces context-dependent embeddings — the same word gets different vectors depending on its surrounding context",
      "ELMo uses a larger vocabulary",
      "ELMo requires less training data",
    ],
    answer: 1,
    explain:
      "Word2Vec and GloVe are static — one fixed vector per word regardless of context. ELMo uses a bidirectional LSTM to produce different embeddings for the same word depending on its sentence context, capturing polysemy (e.g., 'bank' in a financial vs. river context).",
  },
  {
    q: "Why do dense embeddings usually outperform one-hot vectors for semantic tasks?",
    options: [
      "They use more storage per word",
      "They can place related words near each other in a continuous space",
      "They eliminate the need for tokenization",
      "They always preserve word order exactly",
    ],
    answer: 1,
    explain:
      "Dense embeddings learn geometry: similar words are close together, so downstream models can exploit semantic neighborhood structure that one-hot vectors cannot express.",
  },
  {
    q: "What is a major limitation of static embeddings like Word2Vec and GloVe?",
    options: [
      "They cannot represent any words at all",
      "The same word always gets the same vector, even when its meaning changes by context",
      "They are only useful for classification tasks",
      "They require labeled training data",
    ],
    answer: 1,
    explain:
      "Static embeddings collapse all senses of a word into one vector. That works for many tasks, but it cannot separate meanings like 'bank' as a river edge versus a financial institution.",
  },
  {
    q: "Why is the embedding dimension emb_sz usually much smaller than the vocabulary size?",
    options: [
      "To make the vectors easier to sort alphabetically",
      "To force a compact latent representation that captures useful structure without one dimension per word",
      "Because neural networks cannot handle large matrices",
      "To make all words orthogonal to each other",
    ],
    answer: 1,
    explain:
      "The embedding dimension is a compression choice. A smaller dense space encourages the model to learn shared semantic features instead of memorizing a separate coordinate for every word.",
  },
  {
    q: "What does it mean when two word vectors are close in embedding space?",
    options: [
      "They have identical spellings",
      "They tend to appear in similar contexts or have related meanings",
      "They always have the same part of speech",
      "They were trained in the same mini-batch",
    ],
    answer: 1,
    explain:
      "In a well-trained embedding space, distance reflects distributional similarity: words that appear in similar contexts often end up near each other.",
  },
  {
    q: "Why are contextual models like ELMo helpful for words with multiple senses?",
    options: [
      "They assign a single global vector to each word",
      "They derive the representation from the surrounding sentence, so different senses can map to different vectors",
      "They remove the need for a vocabulary",
      "They always use one-hot vectors internally",
    ],
    answer: 1,
    explain:
      "Contextual models condition on the sentence around the word, so the representation changes with usage. That lets them separate multiple senses of the same surface form.",
  },
];

export const QUIZ_QUESTIONS: Record<"en" | "ar", QuizQuestion[]> = {
  en: questions,
  ar: questions,
};
