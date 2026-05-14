import { useState } from "react";
import { ChevronDown, Lightbulb } from "lucide-react";
import { useLang } from "@/lib/i18n";

export function ConceptCards() {
  const { t } = useLang();
  const [expanded, setExpanded] = useState<number | null>(0);

  const CONCEPTS = Array.from({ length: 6 }).map((_, i) => {
    return {
      title: t(`concepts.card${i + 1}.title`),
      shortDesc: t(`concepts.card${i + 1}.short`),
      fullDesc: t(`concepts.card${i + 1}.full`),
      example: t(`concepts.card${i + 1}.example`),
    };
  });

  return (
    <div className="space-y-3">
      <div className="flex items-center gap-2 mb-4">
        <Lightbulb className="h-5 w-5 text-primary" />
        <h3 className="text-lg font-semibold">{t("concepts.title")}</h3>
      </div>

      {CONCEPTS.map((concept, idx) => (
        <div key={idx} className="rounded-lg border bg-card shadow-[var(--shadow-soft)] overflow-hidden">
          <button
            onClick={() => setExpanded(expanded === idx ? null : idx)}
            className="w-full px-5 py-4 flex items-center justify-between gap-3 hover:bg-secondary/50 transition text-start"
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

