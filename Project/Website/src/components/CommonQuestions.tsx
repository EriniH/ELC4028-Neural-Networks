import { useState } from "react";
import { ChevronDown, HelpCircle } from "lucide-react";

const FAQS = [
  {
    q: "Why not just use raw word indices (0, 1, 2...) instead of embeddings?",
    a: "Raw indices are arbitrary integers with no semantic meaning. Index 500 is not inherently 'more similar' to 501 than to 5000. Neural networks rely on measuring similarity through dot products—without meaningful representations, similarity operations are meaningless. Embeddings map words to a vector space where proximity reflects semantic similarity.",
    tag: "Fundamentals",
  },
  {
    q: "Why do we need dense vectors instead of just using sparse one-hot encoded vectors?",
    a: "One-hot encoding creates ultra-sparse vectors (one 1, thousands of 0s) where every different word pair has dot product = 0. This means 'hotel' appears equally unrelated to both 'motel' and 'the'—semantically wrong. Dense embeddings are compact and allow the network to learn meaningful similarity relationships.",
    tag: "Representation",
  },
  {
    q: "How do embeddings learn semantic meaning without being explicitly told?",
    a: "Through self-supervised learning. Word2Vec trains to predict word co-occurrence: given 'the dog barks', predict 'dog' from context words. Over billions of examples, the network learns that 'dog' and 'cat' often appear in similar contexts, so their embeddings become similar. The semantic structure emerges from predicting these patterns.",
    tag: "Training",
  },
  {
    q: "Can embeddings really represent abstract concepts like 'royalty' or 'gender'?",
    a: "Yes, but indirectly. Embeddings don't have explicit 'royalty dimension'—instead, multiple dimensions combine to form abstract patterns. When you compute king - man + woman ≈ queen, you're performing arithmetic that happens to align with the gender concept the network learned from data. It's emergent structure, not hard-coded.",
    tag: "Interpretation",
  },
  {
    q: "Why do we need pre-trained embeddings instead of training from scratch?",
    a: "Training good embeddings requires massive data (billions of words) to capture semantic relationships. Pre-training on Wikipedia/web gives you embeddings trained on universal language patterns. You can then fine-tune on your small task-specific dataset, leveraging the pre-learned structure. This dramatically reduces data and training requirements.",
    tag: "Transfer Learning",
  },
  {
    q: "What's the difference between Word2Vec and GloVe?",
    a: "Word2Vec uses a local sliding window (nearby words) and predicts context. GloVe uses global co-occurrence statistics (all pairwise word co-occurrence counts). Word2Vec is faster; GloVe uses more information and often performs slightly better on semantic tasks. Both produce high-quality static embeddings—the choice is usually task-dependent.",
    tag: "Methods",
  },
  {
    q: "Why can't static embeddings (Word2Vec/GloVe) handle ambiguity like 'bank'?",
    a: "Static embeddings assign one fixed vector per word. 'Bank' gets the same vector in 'river bank' and 'bank account'—but these mean different things semantically. Contextualized embeddings (ELMo, BERT) produce different vectors depending on surrounding words, solving this ambiguity.",
    tag: "Advanced",
  },
  {
    q: "How does FastText handle words that weren't in the training data?",
    a: "FastText represents words as sums of character n-grams instead of whole-word vectors. An unknown word like 'spaceship' is decomposed into n-grams ('spa', 'pac', 'ace', etc.) which were trained. Since these sub-word units appeared in many training words, you can approximate the unseen word's embedding from its parts.",
    tag: "Advanced",
  },
];

export function CommonQuestions() {
  const [expanded, setExpanded] = useState<number | null>(null);
  const [selectedTag, setSelectedTag] = useState<string | null>(null);

  const allTags = Array.from(new Set(FAQS.map((q) => q.tag)));
  const filtered = selectedTag ? FAQS.filter((q) => q.tag === selectedTag) : FAQS;

  return (
    <div className="space-y-6">
      <div className="flex items-center gap-2 mb-6">
        <HelpCircle className="h-5 w-5 text-primary" />
        <h3 className="text-lg font-semibold">Common Questions & Misconceptions</h3>
      </div>

      {/* Tag Filter */}
      <div className="flex flex-wrap gap-2">
        <button
          onClick={() => setSelectedTag(null)}
          className={`rounded-full px-3 py-1 text-xs font-medium transition ${
            selectedTag === null
              ? "bg-primary text-primary-foreground"
              : "border border-border bg-background hover:bg-secondary"
          }`}
        >
          All
        </button>
        {allTags.map((tag) => (
          <button
            key={tag}
            onClick={() => setSelectedTag(tag)}
            className={`rounded-full px-3 py-1 text-xs font-medium transition ${
              selectedTag === tag
                ? "bg-primary text-primary-foreground"
                : "border border-border bg-background hover:bg-secondary"
            }`}
          >
            {tag}
          </button>
        ))}
      </div>

      {/* Q&A Cards */}
      <div className="space-y-2">
        {filtered.map((faq, idx) => (
          <div key={idx} className="rounded-lg border bg-card shadow-[var(--shadow-soft)] overflow-hidden">
            <button
              onClick={() => setExpanded(expanded === idx ? null : idx)}
              className="w-full px-5 py-4 flex items-start justify-between gap-3 hover:bg-secondary/50 transition text-left"
            >
              <div className="flex-1 text-start">
                <div className="font-semibold text-foreground">{faq.q}</div>
                <div className="text-xs text-primary/70 mt-1">{faq.tag}</div>
              </div>
              <ChevronDown
                className={`h-5 w-5 text-muted-foreground flex-shrink-0 mt-0.5 transition-transform ${
                  expanded === idx ? "rotate-180" : ""
                }`}
              />
            </button>

            {expanded === idx && (
              <div className="border-t bg-secondary/20 px-5 py-4">
                <p className="text-sm leading-relaxed text-foreground/90">{faq.a}</p>
              </div>
            )}
          </div>
        ))}
      </div>

      {filtered.length === 0 && (
        <div className="text-center py-8 text-muted-foreground text-sm">
          No questions found for this category.
        </div>
      )}
    </div>
  );
}
