import { createFileRoute, Link } from "@tanstack/react-router";
import { Download, ArrowLeft, Languages } from "lucide-react";
import { useLang } from "@/lib/i18n";

export const Route = createFileRoute("/presentation")({
  head: () => ({
    meta: [
      { title: "Presentation — Word Embeddings" },
      { name: "description", content: "View and download the Word Embeddings presentation." },
    ],
  }),
  component: Presentation,
});

function Presentation() {
  const { t, lang, setLang } = useLang();
  return (
    <div className="min-h-screen flex flex-col">
      {/* nav */}
      <header className="sticky top-0 z-20 border-b border-border/60 bg-background/80 backdrop-blur">
        <nav className="mx-auto flex w-full max-w-5xl items-center justify-between px-6 py-3 text-sm">
          <Link to="/" className="flex items-center gap-2 font-display text-lg hover:text-primary transition">
            <ArrowLeft className="h-4 w-4" /> {t("nav.brand")}
          </Link>
          <div className="flex items-center gap-4">
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

      {/* content */}
      <main className="flex-1 max-w-6xl w-full mx-auto p-6 flex flex-col mt-4">
        <div className="mb-6 flex flex-col sm:flex-row items-start sm:items-center justify-between gap-4">
          <div>
            <h1 className="text-3xl font-display">{t("presentation.title")}</h1>
            <p className="text-muted-foreground mt-2">{t("presentation.desc")}</p>
          </div>
          <a
            href={`${import.meta.env.BASE_URL}WordEmbedding.pptx`}
            download
            className="inline-flex items-center gap-2 rounded-lg bg-primary px-5 py-2.5 text-sm font-medium text-primary-foreground transition hover:opacity-90 whitespace-nowrap"
          >
            <Download className="h-4 w-4" /> {t("presentation.download")}
          </a>
        </div>
        
        <div className="flex-1 rounded-xl overflow-hidden border shadow-sm bg-muted/30">
          <object
            data={`${import.meta.env.BASE_URL}WordEmbedding.pdf#toolbar=0&view=FitH`}
            type="application/pdf"
            className="w-full h-[75vh] min-h-[500px]"
            title="Word Embeddings Presentation"
          >
            <embed 
              src={`${import.meta.env.BASE_URL}WordEmbedding.pdf#toolbar=0&view=FitH`} 
              type="application/pdf" 
              className="w-full h-full" 
            />
            <p className="p-6 text-center text-muted-foreground">
              Your browser does not support inline PDFs. 
              <a href={`${import.meta.env.BASE_URL}WordEmbedding.pdf`} target="_blank" rel="noreferrer" className="text-primary hover:underline ml-1">
                Click here to view the PDF
              </a>
            </p>
          </object>
        </div>
      </main>
    </div>
  );
}
