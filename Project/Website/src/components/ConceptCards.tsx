import { useState } from "react";
import { ChevronDown, Lightbulb } from "lucide-react";

const CONCEPTS = [
  {
    title: "The Fundamental Problem",
    shortDesc: "Why can't we just use word indices?",
    fullDesc:
      "Raw vocabulary indices (0, 1, 2...) are arbitrary integers with no semantic meaning. The index 123 is mathematically no more 'similar' to 124 than to 9999. Since machine learning relies on measuring similarity (via dot products in neurons), we need representations where proximity in vector space reflects semantic similarity.",
    example: `Index("hotel") = 500
Index("motel") = 3820
Dot product: 0 (completely orthogonal!)

But hotel and motel are semantically very similar.`,
  },
  {
    title: "Sparsity & One-Hot Encoding",
    shortDesc: "Why OHE doesn't capture meaning",
    fullDesc:
      "One-Hot Encoding creates a vector of length vocab_size with exactly one 1 and all other 0s. This is extremely sparse and, critically, every pair of different words is orthogonal (dot product = 0). So 'hotel' and 'motel' appear equally unrelated as 'hotel' and 'the'.",
    example: `OHE("hotel")  = [0,0,...,1,...,0]  (one 1 at index 500)
OHE("motel")  = [0,0,...,1,...,0]  (one 1 at index 3820)
OHE("the")    = [0,0,...,1,...,0]  (one 1 at some other index)

dot(OHE("hotel"), OHE("motel")) = 0
dot(OHE("hotel"), OHE("the")) = 0

Same similarity! Both pairs are equally unrelated.`,
  },
  {
    title: "What Embeddings Solve",
    shortDesc: "Dense, learned semantic vectors",
    fullDesc:
      "An embedding is a learned mapping from discrete symbols to dense continuous vectors. Each word becomes a small vector (e.g., 300 dimensions) of real numbers. The network learns these values through backpropagation such that semantically similar words end up close in vector space.",
    example: `Embedding("hotel")  ≈ [0.2, -0.5, 0.8, ..., 0.1]  (300 dims)
Embedding("motel")  ≈ [0.21, -0.48, 0.79, ..., 0.12] (very close!)
Embedding("the")    ≈ [-0.1, 0.3, -0.2, ..., 0.9]  (very different)

Cosine Similarity:
sim("hotel", "motel") ≈ 0.98  (highly similar!)
sim("hotel", "the") ≈ 0.12    (very different)`,
  },
  {
    title: "Transfer Learning & Pre-training",
    shortDesc: "Why pre-trained embeddings are powerful",
    fullDesc:
      "Training embeddings from scratch requires large datasets. Instead, we can pre-train embeddings on a massive generic corpus (Wikipedia, news, web), then transfer them to our specific task. This works because semantic relationships are universal across domains.",
    example: `Pre-training corpus: Wikipedia + Common Crawl (billions of words)
Learn embeddings for ALL words
Transfer to your task: e.g., medical text classification

Result: You get embeddings trained on massive data, fine-tuned for your task. Much better than training from scratch on small data.`,
  },
  {
    title: "Vector Arithmetic & Analogies",
    shortDesc: "Emergent semantic operations",
    fullDesc:
      "A remarkable property: embeddings support vector arithmetic that captures semantic relationships. For example, King - Man + Woman ≈ Queen. This emerges entirely from the training objective (predicting context) without explicit instruction. It shows that embeddings encode genuine semantic structure.",
    example: `vec("king") - vec("man") + vec("woman") ≈ vec("queen")
vec("paris") - vec("france") + vec("italy") ≈ vec("rome")
vec("running") - vec("run") + vec("sleeping") ≈ vec("sleep")

The model was NEVER told about royalty or geography explicitly.
It discovered these relationships by predicting word co-occurrence!`,
  },
  {
    title: "Static vs. Contextual Embeddings",
    shortDesc: "The evolution of representation",
    fullDesc:
      "Early methods (Word2Vec, GloVe) produce static embeddings: one fixed vector per word regardless of context. But 'bank' means different things in 'river bank' vs 'bank account'. Modern methods like ELMo and BERT produce contextual embeddings: different vectors depending on the surrounding words.",
    example: `Static (Word2Vec):
vec("bank") = [0.1, -0.2, 0.5, ...] (always the same)

Contextual (ELMo/BERT):
vec("bank") in "river bank" ≈ [0.15, -0.25, 0.45, ...]
vec("bank") in "bank account" ≈ [0.08, -0.18, 0.52, ...]

Context changes the representation!`,
  },
];

export function ConceptCards() {
  const [expanded, setExpanded] = useState<number | null>(0);

  return (
    <div className="space-y-3">
      <div className="flex items-center gap-2 mb-4">
        <Lightbulb className="h-5 w-5 text-primary" />
        <h3 className="text-lg font-semibold">Key Concepts</h3>
      </div>

      {CONCEPTS.map((concept, idx) => (
        <div key={idx} className="rounded-lg border bg-card shadow-[var(--shadow-soft)] overflow-hidden">
          <button
            onClick={() => setExpanded(expanded === idx ? null : idx)}
            className="w-full px-5 py-4 flex items-center justify-between gap-3 hover:bg-secondary/50 transition text-left"
          >
            <div className="flex-1">
              <h4 className="font-semibold text-foreground">{concept.title}</h4>
              <p className="text-sm text-muted-foreground mt-0.5">{concept.shortDesc}</p>
            </div>
            <ChevronDown
              className={`h-5 w-5 text-muted-foreground flex-shrink-0 transition-transform ${
                expanded === idx ? "rotate-180" : ""
              }`}
            />
          </button>

          {expanded === idx && (
            <div className="border-t bg-secondary/20 px-5 py-4 space-y-4">
              <p className="text-sm leading-relaxed text-foreground/90">{concept.fullDesc}</p>

              <div className="bg-background rounded-lg p-4 font-mono text-xs leading-relaxed whitespace-pre-wrap overflow-x-auto border border-border">
                <code className="text-foreground/80">{concept.example}</code>
              </div>
            </div>
          )}
        </div>
      ))}
    </div>
  );
}
