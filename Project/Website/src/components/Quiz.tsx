import { useState } from "react";
import { Check, X, RotateCcw } from "lucide-react";
import { useLang, QUIZ_QUESTIONS } from "@/lib/i18n";

export function Quiz() {
  const { t, lang } = useLang();
  const QUESTIONS = QUIZ_QUESTIONS[lang];

  const [idx, setIdx] = useState(0);
  const [picked, setPicked] = useState<number | null>(null);
  const [score, setScore] = useState(0);
  const [done, setDone] = useState(false);

  const q = QUESTIONS[idx];

  function choose(i: number) {
    if (picked !== null) return;
    setPicked(i);
    if (i === q.answer) setScore((s) => s + 1);
  }

  function next() {
    if (idx + 1 >= QUESTIONS.length) {
      setDone(true);
    } else {
      setIdx(idx + 1);
      setPicked(null);
    }
  }

  function reset() {
    setIdx(0);
    setPicked(null);
    setScore(0);
    setDone(false);
  }

  if (done) {
    const pct = Math.round((score / QUESTIONS.length) * 100);
    return (
      <div className="rounded-xl border bg-card p-8 text-center shadow-[var(--shadow-soft)]">
        <div className="font-display text-5xl">{score}/{QUESTIONS.length}</div>
        <p className="mt-2 text-muted-foreground">
          {pct === 100 ? t("quiz.perfect") : pct >= 60 ? t("quiz.nice") : t("quiz.again")}
        </p>
        <button
          onClick={reset}
          className="mt-6 inline-flex items-center gap-2 rounded-lg bg-primary px-5 py-2.5 text-sm font-medium text-primary-foreground transition hover:opacity-90"
        >
          <RotateCcw className="h-4 w-4" /> {t("quiz.try")}
        </button>
      </div>
    );
  }

  return (
    <div className="rounded-xl border bg-card p-6 shadow-[var(--shadow-soft)] sm:p-8">
      <div className="mb-3 flex items-center justify-between text-xs uppercase tracking-wider text-muted-foreground">
        <span>{t("quiz.q")} {idx + 1} {t("quiz.of")} {QUESTIONS.length}</span>
        <span>{t("quiz.score")} {score}</span>
      </div>
      <div className="h-1.5 w-full overflow-hidden rounded-full bg-secondary">
        <div
          className="h-full bg-primary transition-all"
          style={{ width: `${((idx + (picked !== null ? 1 : 0)) / QUESTIONS.length) * 100}%` }}
        />
      </div>

      <h3 className="mt-5 text-2xl">{q.q}</h3>

      <div className="mt-5 space-y-2.5">
        {q.options.map((opt, i) => {
          const isCorrect = i === q.answer;
          const isPicked = i === picked;
          let cls = "border-border bg-background hover:border-ring hover:bg-secondary";
          if (picked !== null) {
            if (isCorrect) cls = "border-success bg-success/10 text-foreground";
            else if (isPicked) cls = "border-destructive bg-destructive/10 text-foreground";
            else cls = "border-border bg-background opacity-60";
          }
          return (
            <button
              key={i}
              onClick={() => choose(i)}
              disabled={picked !== null}
              className={`flex w-full items-center justify-between rounded-lg border px-4 py-3 text-start text-sm transition ${cls}`}
            >
              <span>{opt}</span>
              {picked !== null && isCorrect && <Check className="h-4 w-4 text-success" />}
              {picked !== null && isPicked && !isCorrect && <X className="h-4 w-4 text-destructive" />}
            </button>
          );
        })}
      </div>

      {picked !== null && (
        <div className="mt-5 rounded-lg bg-secondary p-4 text-sm text-secondary-foreground">
          {q.explain}
        </div>
      )}

      {picked !== null && (
        <div className="mt-5 flex justify-end">
          <button
            onClick={next}
            className="rounded-lg bg-primary px-5 py-2.5 text-sm font-medium text-primary-foreground transition hover:opacity-90"
          >
            {idx + 1 >= QUESTIONS.length ? t("quiz.results") : t("quiz.next")}
          </button>
        </div>
      )}
    </div>
  );
}
