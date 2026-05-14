import { useEffect, useState } from "react";
import { Check, X, RotateCcw } from "lucide-react";
import { useLang, QUIZ_QUESTIONS } from "@/lib/i18n";

const QUIZ_LENGTH = 5;

type QuizView = "practice" | "all";

function pickQuestions<T>(questions: T[], count: number) {
  const shuffled = [...questions];

  for (let i = shuffled.length - 1; i > 0; i -= 1) {
    const j = Math.floor(Math.random() * (i + 1));
    [shuffled[i], shuffled[j]] = [shuffled[j], shuffled[i]];
  }

  return shuffled.slice(0, Math.min(count, shuffled.length));
}

export function Quiz() {
  const { t, lang } = useLang();
  const QUESTION_POOL = QUIZ_QUESTIONS[lang];
  const [view, setView] = useState<QuizView>("practice");
  const [activeQuestions, setActiveQuestions] = useState(() => pickQuestions(QUESTION_POOL, QUIZ_LENGTH));

  const [idx, setIdx] = useState(0);
  const [picked, setPicked] = useState<number | null>(null);
  const [score, setScore] = useState(0);
  const [done, setDone] = useState(false);

  useEffect(() => {
    setActiveQuestions(pickQuestions(QUESTION_POOL, QUIZ_LENGTH));
    setIdx(0);
    setPicked(null);
    setScore(0);
    setDone(false);
  }, [lang, QUESTION_POOL]);

  const q = activeQuestions[idx];
  const totalQuestions = QUESTION_POOL.length;

  function choose(i: number) {
    if (picked !== null) return;
    setPicked(i);
    if (i === q.answer) setScore((s) => s + 1);
  }

  function next() {
    if (idx + 1 >= activeQuestions.length) {
      setDone(true);
    } else {
      setIdx(idx + 1);
      setPicked(null);
    }
  }

  function reset() {
    setActiveQuestions(pickQuestions(QUESTION_POOL, QUIZ_LENGTH));
    setIdx(0);
    setPicked(null);
    setScore(0);
    setDone(false);
  }

  const pct = Math.round((score / activeQuestions.length) * 100);

  return (
    <div className="rounded-xl border bg-card p-4 shadow-[var(--shadow-soft)] sm:p-6">
      <div className="flex flex-col gap-4 sm:flex-row sm:items-center sm:justify-between">
        <div className="inline-flex w-full max-w-sm rounded-lg bg-muted p-1">
          <button
            type="button"
            onClick={() => setView("practice")}
            className={`flex-1 rounded-md px-3 py-1.5 text-sm font-medium transition ${view === "practice" ? "bg-background text-foreground shadow" : "text-muted-foreground hover:text-foreground"}`}
          >
            {t("quiz.tab.play")}
          </button>
          <button
            type="button"
            onClick={() => setView("all")}
            className={`flex-1 rounded-md px-3 py-1.5 text-sm font-medium transition ${view === "all" ? "bg-background text-foreground shadow" : "text-muted-foreground hover:text-foreground"}`}
          >
            {t("quiz.tab.all")}
          </button>
        </div>
        <div className="text-xs uppercase tracking-wider text-muted-foreground">
          {t("quiz.score")} {score}/{activeQuestions.length}
        </div>
      </div>
      <p className="mt-2 text-sm text-muted-foreground">
        Practice mode draws 5 random questions from the full {totalQuestions}-question bank each run.
      </p>

      {view === "practice" ? (
      <div className="mt-4">
        {done ? (
          <div className="rounded-xl border bg-background p-8 text-center">
            <div className="font-display text-5xl">
              {score}/{activeQuestions.length}
            </div>
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
        ) : (
          <>
            <div className="mb-3 flex items-center justify-between text-xs uppercase tracking-wider text-muted-foreground">
              <span>
                {t("quiz.q")} {idx + 1} {t("quiz.of")} {activeQuestions.length}
              </span>
              <span>
                {t("quiz.score")} {score}
              </span>
            </div>
            <div className="h-1.5 w-full overflow-hidden rounded-full bg-secondary">
              <div
                className="h-full bg-primary transition-all"
                style={{ width: `${((idx + (picked !== null ? 1 : 0)) / activeQuestions.length) * 100}%` }}
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
                  {idx + 1 >= activeQuestions.length ? t("quiz.results") : t("quiz.next")}
                </button>
              </div>
            )}
          </>
        )}
      </div>
      ) : (
      <div className="mt-4">
        <div className="mb-4">
          <div className="text-xs uppercase tracking-wider text-muted-foreground">{t("quiz.all.title")}</div>
          <p className="mt-1 text-sm text-muted-foreground">
            All {totalQuestions} questions are listed here, with a new random set of {QUIZ_LENGTH} in the practice tab.
          </p>
        </div>
        <div className="space-y-4">
          {QUESTION_POOL.map((question, questionIndex) => (
            <article key={question.q} className="rounded-xl border bg-background p-4 sm:p-5">
              <div className="text-xs uppercase tracking-wider text-muted-foreground">
                {t("quiz.q")} {questionIndex + 1}
              </div>
              <h4 className="mt-2 text-lg leading-snug">{question.q}</h4>
              <div className="mt-4 space-y-2">
                {question.options.map((option, optionIndex) => {
                  const isAnswer = optionIndex === question.answer;
                  return (
                    <div
                      key={option}
                      className={`rounded-lg border px-3 py-2 text-sm ${isAnswer ? "border-success bg-success/10" : "border-border bg-card"}`}
                    >
                      <span className="mr-2 font-medium">{String.fromCharCode(65 + optionIndex)}.</span>
                      {option}
                    </div>
                  );
                })}
              </div>
              <p className="mt-4 text-sm text-muted-foreground">
                <span className="font-medium text-foreground">{t("quiz.all.answer")}: </span>
                {question.options[question.answer]}
              </p>
              <p className="mt-2 text-sm text-muted-foreground">
                <span className="font-medium text-foreground">{t("quiz.all.explain")}: </span>
                {question.explain}
              </p>
            </article>
          ))}
        </div>
      </div>
      )}
    </div>
  );
}
