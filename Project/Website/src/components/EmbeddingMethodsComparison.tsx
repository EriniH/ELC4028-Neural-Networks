import { useState } from "react";
import { GitCompare } from "lucide-react";
import { useLang } from "@/lib/i18n";

const METHODS = [
  {
    id: "word2vec",
    name: "Word2Vec",
    year: "2013",
    key: "SGNS (Skip-Gram with Negative Sampling)",
  },
  {
    id: "glove",
    name: "GloVe",
    year: "2014",
    key: "Weighted matrix factorization",
  },
  {
    id: "fasttext",
    name: "FastText",
    year: "2016",
    key: "Sub-word character n-grams",
  },
  {
    id: "elmo",
    name: "ELMo",
    year: "2018",
    key: "Contextualized (different per sentence)",
  },
];

export function EmbeddingMethodsComparison() {
  const { t } = useLang();
  const [selectedIdx, setSelectedIdx] = useState(0);
  const selected = METHODS[selectedIdx];

  const details = {
    context: t(`methods.${selected.id}.context`),
    training: t(`methods.${selected.id}.training`),
    oov: t(`methods.${selected.id}.oov`),
    speed: t(`methods.${selected.id}.speed`),
    pros: (t(`methods.${selected.id}.pros`) || "").split(","),
    cons: (t(`methods.${selected.id}.cons`) || "").split(","),
  };

  return (
    <div className="rounded-xl border bg-card p-6 shadow-[var(--shadow-soft)]">
      <div className="mb-6 flex items-center gap-2">
        <GitCompare className="h-5 w-5 text-primary" />
        <h3 className="text-lg font-semibold">{t("methods.title")}</h3>
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
          { label: t("methods.contextType"), value: details.context },
          { label: t("methods.trainingGoal"), value: details.training },
          { label: t("methods.handlesOOV"), value: details.oov },
          { label: t("methods.speed"), value: details.speed },
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
          {t("methods.keyInnovation")}
        </div>
        <div className="text-sm text-foreground">{selected.key}</div>
      </div>

      {/* Pros and Cons */}
      <div className="grid gap-4 md:grid-cols-2 mb-6">
        <div>
          <h4 className="text-xs uppercase tracking-wider font-semibold text-success mb-3">
            {t("methods.strengths")}
          </h4>
          <ul className="space-y-2">
            {details.pros.map((pro, idx) => (
              <li key={idx} className="flex gap-2 text-sm text-foreground/90">
                <span className="text-success mt-0.5">✓</span>
                <span>{pro}</span>
              </li>
            ))}
          </ul>
        </div>
        <div>
          <h4 className="text-xs uppercase tracking-wider font-semibold text-amber-600 mb-3">
            {t("methods.limitations")}
          </h4>
          <ul className="space-y-2">
            {details.cons.map((con, idx) => (
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
        <p className="text-xs text-muted-foreground leading-relaxed text-start">
          💡 <strong>Progression:</strong> {t("methods.progression")}
        </p>
      </div>
    </div>
  );
}
