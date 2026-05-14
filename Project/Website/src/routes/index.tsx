import { createFileRoute } from "@tanstack/react-router";
import { EmbeddingDemo } from "@/components/EmbeddingDemo";
import { Quiz } from "@/components/Quiz";
import { Sparkles, Brain, Target, BookOpen, Languages } from "lucide-react";
import { useLang } from "@/lib/i18n";

export const Route = createFileRoute("/")({
  head: () => ({
    meta: [
      { title: "Word Embeddings — An Interactive Tutorial" },
      {
        name: "description",
        content:
          "Learn what word embeddings are with clear explanations, an interactive 2D demo, and a quiz to test your understanding.",
      },
      { property: "og:title", content: "Word Embeddings — An Interactive Tutorial" },
      {
        property: "og:description",
        content:
          "Visualize word vectors, explore nearest neighbors, and quiz yourself on Word2Vec, GloVe, and contextual embeddings.",
      },
    ],
  }),
  component: Index,
});

function Section({ id, eyebrow, title, children }: { id: string; eyebrow: string; title: string; children: React.ReactNode }) {
  return (
    <section id={id} className="mx-auto max-w-5xl px-6 py-20 scroll-mt-16">
      <div className="mb-8">
        <div className="mb-2 text-xs font-medium uppercase tracking-[0.2em] text-primary">{eyebrow}</div>
        <h2 className="text-4xl sm:text-5xl">{title}</h2>
      </div>
      {children}
    </section>
  );
}

function Index() {
  const { t, lang, setLang } = useLang();
  return (
    <div className="min-h-screen">
      {/* nav */}
      <header className="sticky top-0 z-20 border-b border-border/60 bg-background/80 backdrop-blur">
        <nav className="mx-auto flex max-w-5xl items-center justify-between px-6 py-3 text-sm">
          <a href="#top" className="flex items-center gap-2 font-display text-lg">
            <Sparkles className="h-4 w-4 text-primary" /> {t("nav.brand")}
          </a>
          <div className="flex items-center gap-4 sm:gap-6">
            <div className="hidden gap-6 text-muted-foreground sm:flex">
              <a href="#learn" className="hover:text-foreground">{t("nav.learn")}</a>
              <a href="#demo" className="hover:text-foreground">{t("nav.demo")}</a>
              <a href="#arabic" className="hover:text-foreground">{t("nav.arabic")}</a>
              <a href="#quiz" className="hover:text-foreground">{t("nav.quiz")}</a>
            </div>
            <button
              onClick={() => setLang(lang === "en" ? "ar" : "en")}
              className="inline-flex items-center gap-1.5 rounded-lg border border-border bg-background px-3 py-1.5 text-xs font-medium transition hover:bg-secondary"
              aria-label="Toggle language"
            >
              <Languages className="h-3.5 w-3.5" /> {t("nav.lang")}
            </button>
          </div>
        </nav>
      </header>

      {/* hero */}
      <section id="top" className="relative overflow-hidden">
        <div
          className="absolute inset-0 -z-10 opacity-[0.18]"
          style={{ background: "var(--gradient-hero)" }}
        />
        <div className="mx-auto max-w-5xl px-6 py-24 sm:py-32">
          <div className="inline-flex items-center gap-2 rounded-full border bg-background/70 px-3 py-1 text-xs text-muted-foreground backdrop-blur">
            <Brain className="h-3.5 w-3.5" /> {t("hero.badge")}
          </div>
          <h1 className="mt-5 text-5xl leading-[1.05] sm:text-7xl">
            {t("hero.title1")}
            <br />
            <span className="italic text-primary">{t("hero.title2")}</span>
          </h1>
          <p className="mt-6 max-w-2xl text-lg text-muted-foreground">
            {t("hero.desc")}
          </p>
          <div className="mt-8 flex flex-wrap gap-3">
            <a
              href="#learn"
              className="inline-flex items-center gap-2 rounded-lg bg-primary px-5 py-2.5 text-sm font-medium text-primary-foreground transition hover:opacity-90"
            >
              <BookOpen className="h-4 w-4" /> {t("hero.start")}
            </a>
            <a
              href="#demo"
              className="inline-flex items-center gap-2 rounded-lg border border-border bg-background px-5 py-2.5 text-sm font-medium transition hover:bg-secondary"
            >
              <Target className="h-4 w-4" /> {t("hero.skip")}
            </a>
          </div>
        </div>
      </section>

      {/* explanation */}
      <Section id="learn" eyebrow={t("learn.eyebrow")} title={t("learn.title")}>
        <div className="prose-like space-y-5 text-[17px] leading-relaxed text-foreground/90">
          <p>{t("learn.p1")}</p>
          <p>{t("learn.p2")}</p>
        </div>

        <div className="mt-10 grid gap-5 md:grid-cols-3">
          {[
            { t: t("learn.card1.t"), d: t("learn.card1.d") },
            { t: t("learn.card2.t"), d: t("learn.card2.d") },
            { t: t("learn.card3.t"), d: t("learn.card3.d") },
          ].map((c) => (
            <div key={c.t} className="rounded-xl border bg-card p-5 shadow-[var(--shadow-soft)]">
              <div className="font-display text-xl">{c.t}</div>
              <p className="mt-2 text-sm text-muted-foreground">{c.d}</p>
            </div>
          ))}
        </div>

        <div className="mt-10 rounded-xl border bg-secondary/60 p-6">
          <div className="mb-2 text-xs uppercase tracking-wider text-muted-foreground">{t("learn.example")}</div>
          <div className="font-mono text-lg sm:text-xl" dir="ltr">
            vec(<span className="text-primary">king</span>) − vec(man) + vec(woman) ≈ vec(<span className="text-primary">queen</span>)
          </div>
          <p className="mt-2 text-sm text-muted-foreground">
            {t("learn.example.note")}
          </p>
        </div>
      </Section>

      {/* demo */}
      <Section id="demo" eyebrow={t("demo.eyebrow")} title={t("demo.title")}>
        <p className="mb-8 max-w-2xl text-muted-foreground">{t("demo.desc")}</p>
        <EmbeddingDemo />
      </Section>

      {/* arabic demo */}
      <Section id="arabic" eyebrow={t("arabic.eyebrow")} title={t("arabic.title")}>
        <p className="mb-6 max-w-2xl text-muted-foreground">{t("arabic.desc")}</p>
        <div className="overflow-hidden rounded-xl border bg-card shadow-[var(--shadow-soft)]">
          <iframe
            src="/arabic-embeddings-demo.html"
            title="Arabic Word Embeddings Demo"
            className="block h-[860px] w-full border-0 md:h-[900px]"
            loading="lazy"
          />
        </div>
        <div className="mt-3 text-right">
          <a
            href="/arabic-embeddings-demo.html"
            target="_blank"
            rel="noreferrer"
            className="text-sm text-primary hover:underline"
          >
            {t("arabic.open")} ↗
          </a>
        </div>
      </Section>

      {/* quiz */}
      <Section id="quiz" eyebrow={t("quiz.eyebrow")} title={t("quiz.title")}>
        <Quiz />
      </Section>

      <footer className="border-t border-border/60">
        <div className="mx-auto max-w-5xl px-6 py-10 text-sm text-muted-foreground">
          {t("footer")}
        </div>
      </footer>
    </div>
  );
}
