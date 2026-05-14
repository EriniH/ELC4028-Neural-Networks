import { useMemo, useState } from "react";
import { WORDS, GROUP_COLORS, similarity, type WordVec } from "@/lib/embeddings";
import { useLang } from "@/lib/i18n";

const W = 560;
const H = 420;
const PAD = 40;

function project(v: WordVec) {
  // map x,y from [-4,4] to canvas
  const sx = ((v.x + 4) / 8) * (W - PAD * 2) + PAD;
  const sy = H - (((v.y + 4) / 8) * (H - PAD * 2) + PAD);
  return { sx, sy };
}

export function EmbeddingDemo() {
  const { t } = useLang();
  const [selected, setSelected] = useState<string>("king");

  const sel = useMemo(() => WORDS.find((w) => w.word === selected)!, [selected]);

  const neighbors = useMemo(() => {
    return WORDS.filter((w) => w.word !== sel.word)
      .map((w) => ({ ...w, sim: similarity(sel, w) }))
      .sort((a, b) => b.sim - a.sim)
      .slice(0, 5);
  }, [sel]);

  return (
    <div className="grid gap-6 lg:grid-cols-[1fr_280px]">
      <div className="rounded-xl border bg-card p-4 shadow-[var(--shadow-soft)]">
        <svg viewBox={`0 0 ${W} ${H}`} className="h-auto w-full">
          {/* axes */}
          <line x1={PAD} y1={H / 2} x2={W - PAD} y2={H / 2} stroke="oklch(0.9 0.01 260)" strokeDasharray="3 4" />
          <line x1={W / 2} y1={PAD} x2={W / 2} y2={H - PAD} stroke="oklch(0.9 0.01 260)" strokeDasharray="3 4" />

          {/* lines from selected to neighbors */}
          {neighbors.map((n) => {
            const a = project(sel);
            const b = project(n);
            return (
              <line
                key={`l-${n.word}`}
                x1={a.sx}
                y1={a.sy}
                x2={b.sx}
                y2={b.sy}
                stroke="oklch(0.45 0.18 265)"
                strokeOpacity={n.sim * 0.6}
                strokeWidth={1.5}
              />
            );
          })}

          {WORDS.map((w) => {
            const { sx, sy } = project(w);
            const isSel = w.word === sel.word;
            return (
              <g
                key={w.word}
                className="cursor-pointer"
                onClick={() => setSelected(w.word)}
              >
                <circle
                  cx={sx}
                  cy={sy}
                  r={isSel ? 9 : 6}
                  fill={GROUP_COLORS[w.group]}
                  stroke={isSel ? "oklch(0.18 0.03 260)" : "white"}
                  strokeWidth={isSel ? 2.5 : 1.5}
                />
                <text
                  x={sx + 10}
                  y={sy + 4}
                  fontSize={12}
                  fill="oklch(0.25 0.04 260)"
                  fontWeight={isSel ? 700 : 500}
                >
                  {w.word}
                </text>
              </g>
            );
          })}
        </svg>
        <p className="mt-2 text-xs text-muted-foreground">
          {t("demo.hint")}
        </p>
      </div>

      <div className="rounded-xl border bg-card p-5 shadow-[var(--shadow-soft)]" dir="ltr">
        <div className="mb-1 text-xs uppercase tracking-wider text-muted-foreground">
          {t("demo.selected")}
        </div>
        <div className="font-display text-3xl">{sel.word}</div>
        <div className="mt-1 text-sm text-muted-foreground">
          vector ≈ ({sel.x.toFixed(2)}, {sel.y.toFixed(2)})
        </div>
        <div className="mt-5 mb-2 text-xs uppercase tracking-wider text-muted-foreground">
          {t("demo.neighbors")}
        </div>
        <ul className="space-y-2">
          {neighbors.map((n) => (
            <li key={n.word} className="flex items-center gap-2">
              <span
                className="h-2.5 w-2.5 rounded-full"
                style={{ background: GROUP_COLORS[n.group] }}
              />
              <span className="flex-1 text-sm">{n.word}</span>
              <span className="font-mono text-xs text-muted-foreground">
                {n.sim.toFixed(3)}
              </span>
            </li>
          ))}
        </ul>
      </div>
    </div>
  );
}
