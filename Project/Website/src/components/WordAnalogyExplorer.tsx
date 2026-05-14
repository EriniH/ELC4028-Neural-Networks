import { useMemo, useState } from "react";
import { WORDS, similarity, type WordVec } from "@/lib/embeddings";
import { Zap } from "lucide-react";

const ANALOGIES = [
  { a: "king", b: "man", c: "woman", expected: "queen" },
  { a: "paris", b: "france", c: "italy", expected: "rome" },
  { a: "doctor", b: "man", c: "woman", expected: "nurse" },
];

function getClosestWord(targetX: number, targetY: number): { word: string; distance: number } {
  let closest = WORDS[0];
  let minDist = Infinity;

  for (const w of WORDS) {
    const dist = Math.hypot(w.x - targetX, w.y - targetY);
    if (dist < minDist) {
      minDist = dist;
      closest = w;
    }
  }

  return { word: closest.word, distance: minDist };
}

function performAnalogy(a: string, b: string, c: string) {
  const wA = WORDS.find((w) => w.word === a)!;
  const wB = WORDS.find((w) => w.word === b)!;
  const wC = WORDS.find((w) => w.word === c)!;

  // Vector arithmetic: A - B + C
  const resultX = wA.x - wB.x + wC.x;
  const resultY = wA.y - wB.y + wC.y;

  return getClosestWord(resultX, resultY);
}

export function WordAnalogyExplorer() {
  const [currentIdx, setCurrentIdx] = useState(0);
  const [showResult, setShowResult] = useState(false);

  const analogy = ANALOGIES[currentIdx];
  const result = useMemo(() => performAnalogy(analogy.a, analogy.b, analogy.c), [analogy]);
  const isCorrect = result.word === analogy.expected;

  const handleNext = () => {
    setShowResult(false);
    setCurrentIdx((prev) => (prev + 1) % ANALOGIES.length);
  };

  return (
    <div className="rounded-xl border bg-card p-6 shadow-[var(--shadow-soft)]">
      <div className="mb-4 flex items-center gap-2">
        <Zap className="h-5 w-5 text-primary" />
        <h3 className="text-lg font-semibold">Vector Arithmetic Explorer</h3>
      </div>

      <p className="mb-6 text-sm text-muted-foreground">
        Embeddings capture relationships through vector arithmetic. Watch how subtracting and adding vectors reveals semantic patterns:
      </p>

      {/* Analogy Display */}
      <div className="mb-6 rounded-lg bg-secondary/40 p-4 font-mono text-sm sm:text-base" dir="ltr">
        <div className="space-y-2">
          <div>
            vec(<span className="text-amber-500 font-semibold">{analogy.a}</span>)
            <span className="text-primary"> −</span> vec(<span className="text-blue-500 font-semibold">{analogy.b}</span>)
          </div>
          <div className="text-muted-foreground">+</div>
          <div>
            vec(<span className="text-pink-500 font-semibold">{analogy.c}</span>) <span className="text-primary">≈</span> ?
          </div>
        </div>
      </div>

      {/* Result Button */}
      {!showResult ? (
        <button
          onClick={() => setShowResult(true)}
          className="w-full rounded-lg bg-primary px-4 py-3 font-medium text-primary-foreground transition hover:opacity-90 mb-4"
        >
          Reveal Answer
        </button>
      ) : (
        <div className="mb-4 space-y-3 rounded-lg bg-secondary/50 p-4">
          <div className="flex items-center gap-2">
            <span className="text-sm font-medium">Result:</span>
            <span className={`font-display text-lg ${isCorrect ? "text-success" : "text-amber-500"}`}>
              {result.word}
            </span>
            {isCorrect && <span className="text-sm">✓ Correct!</span>}
          </div>
          <p className="text-xs text-muted-foreground">
            {isCorrect
              ? `Perfect! The arithmetic naturally produces "${result.word}" — the model learned this relationship from co-occurrence patterns.`
              : `Got "${result.word}" instead of "${analogy.expected}". The 2D projection loses some information, but the concept still applies!`}
          </p>
        </div>
      )}

      {/* Navigation */}
      <div className="flex items-center justify-between">
        <span className="text-xs text-muted-foreground">
          {currentIdx + 1} / {ANALOGIES.length}
        </span>
        <button
          onClick={handleNext}
          className="rounded-lg border border-border px-4 py-2 text-sm font-medium transition hover:bg-secondary"
        >
          Next Analogy
        </button>
      </div>

      <div className="mt-4 border-t pt-4">
        <p className="text-xs text-muted-foreground leading-relaxed">
          💡 <strong>Key insight:</strong> This vector arithmetic emerges entirely from the training objective — the model was never explicitly told "king means royalty" or "paris means a specific city." It learned these relationships by optimizing co-occurrence predictions!
        </p>
      </div>
    </div>
  );
}
