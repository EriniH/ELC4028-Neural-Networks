import { useState } from "react";
import { GitCompare } from "lucide-react";

const METHODS = [
  {
    name: "Word2Vec",
    year: "2013",
    context: "Local Window",
    training: "Predict context from word",
    oov: "❌ No",
    speed: "⚡ Very Fast",
    key: "SGNS (Skip-Gram with Negative Sampling)",
    pros: ["Simple & fast", "Works well in practice", "Widely available"],
    cons: ["OOV problem", "Static vectors", "Only local context"],
  },
  {
    name: "GloVe",
    year: "2014",
    context: "Global Corpus",
    training: "Reconstruct co-occurrence matrix",
    oov: "❌ No",
    speed: "⚙️ Moderate",
    key: "Weighted matrix factorization",
    pros: ["Uses global statistics", "Explicit similarity objective", "Strong on semantics"],
    cons: ["OOV problem", "Static vectors", "More complex setup"],
  },
  {
    name: "FastText",
    year: "2016",
    context: "Local Window",
    training: "Char n-grams + word context",
    oov: "✅ Yes",
    speed: "⚡ Fast",
    key: "Sub-word character n-grams",
    pros: ["Handles OOV", "Morphological awareness", "Good for rich languages"],
    cons: ["Vectors still static", "Slightly more parameters"],
  },
  {
    name: "ELMo",
    year: "2018",
    context: "Full Sentence",
    training: "BiLSTM language model",
    oov: "✅ Yes",
    speed: "🐢 Slower",
    key: "Contextualized (different per sentence)",
    pros: ["Context-aware", "Handles OOV", "Multi-layer combination"],
    cons: ["Much slower", "Higher memory", "More complex"],
  },
];

export function EmbeddingMethodsComparison() {
  const [selectedIdx, setSelectedIdx] = useState(0);
  const selected = METHODS[selectedIdx];

  return (
    <div className="rounded-xl border bg-card p-6 shadow-[var(--shadow-soft)]">
      <div className="mb-6 flex items-center gap-2">
        <GitCompare className="h-5 w-5 text-primary" />
        <h3 className="text-lg font-semibold">Embedding Methods Timeline</h3>
      </div>

      {/* Method Selector Buttons */}
      <div className="mb-6 flex gap-2 overflow-x-auto pb-2">
        {METHODS.map((m, idx) => (
          <button
            key={m.name}
            onClick={() => setSelectedIdx(idx)}
            className={`px-3 py-2 rounded-lg whitespace-nowrap text-sm font-medium transition ${
              selectedIdx === idx
                ? "bg-primary text-primary-foreground"
                : "border border-border bg-background hover:bg-secondary"
            }`}
          >
            {m.name} <span className="text-xs opacity-70">({m.year})</span>
          </button>
        ))}
      </div>

      {/* Method Details Grid */}
      <div className="grid gap-4 mb-6 md:grid-cols-2">
        {[
          { label: "Context Type", value: selected.context },
          { label: "Training Goal", value: selected.training },
          { label: "Handles OOV", value: selected.oov },
          { label: "Speed", value: selected.speed },
        ].map((item) => (
          <div key={item.label} className="rounded-lg bg-secondary/30 p-3">
            <div className="text-xs uppercase tracking-wider text-muted-foreground mb-1">
              {item.label}
            </div>
            <div className="text-sm font-medium">{item.value}</div>
          </div>
        ))}
      </div>

      {/* Key Innovation */}
      <div className="mb-6 rounded-lg bg-primary/10 border border-primary/30 p-4">
        <div className="text-xs uppercase tracking-wider text-primary font-semibold mb-1">
          Key Innovation
        </div>
        <div className="text-sm text-foreground">{selected.key}</div>
      </div>

      {/* Pros and Cons */}
      <div className="grid gap-4 md:grid-cols-2 mb-6">
        <div>
          <h4 className="text-xs uppercase tracking-wider font-semibold text-success mb-3">
            Strengths
          </h4>
          <ul className="space-y-2">
            {selected.pros.map((pro, idx) => (
              <li key={idx} className="flex gap-2 text-sm text-foreground/90">
                <span className="text-success mt-0.5">✓</span>
                <span>{pro}</span>
              </li>
            ))}
          </ul>
        </div>
        <div>
          <h4 className="text-xs uppercase tracking-wider font-semibold text-amber-600 mb-3">
            Limitations
          </h4>
          <ul className="space-y-2">
            {selected.cons.map((con, idx) => (
              <li key={idx} className="flex gap-2 text-sm text-foreground/90">
                <span className="text-amber-600 mt-0.5">✕</span>
                <span>{con}</span>
              </li>
            ))}
          </ul>
        </div>
      </div>

      {/* Timeline Context */}
      <div className="border-t pt-4">
        <p className="text-xs text-muted-foreground leading-relaxed">
          💡 <strong>Progression:</strong> From static local-context embeddings (Word2Vec/GloVe) → handling out-of-vocab (FastText) → full contextualization (ELMo) → modern transformers like BERT that attend to the entire document and produce different embeddings per context.
        </p>
      </div>
    </div>
  );
}
