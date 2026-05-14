"""
PySimpleGUI-based desktop interface for Arabic RAG system.
Replaces the Gradio web interface with a desktop GUI.
"""

import PySimpleGUI as sg
import threading
from typing import Dict, List
import os

# ── Theme Setup ────────────────────────────────────────────────────────────
sg.theme('DarkBlue3')
sg.set_options(
    font=('Arial', 10),
    text_justification='right',
)


def build_interface(retrieval_system, rag_system):
    """Build the PySimpleGUI interface."""

    # ── Retrieval Tab ──────────────────────────────────────────────────────────
    def do_retrieval(query: str, top_k: int):
        """Perform retrieval search."""
        if not query.strip():
            return {"error": "يرجى إدخال استعلام"}

        results = retrieval_system.search_all(query, top_k=int(top_k))

        def fmt_results(res_list, method_name):
            if not res_list:
                return "لا توجد نتائج."
            parts = []
            for r in res_list:
                parts.append(
                    f"[{r['rank']}] درجة: {r['score']:.4f}\n"
                    f"{r['text'][:300]}...\n"
                    f"{'-'*80}"
                )
            return f"\n{'='*5}\n{method_name}\n{'='*5}\n\n" + "\n\n".join(parts)

        return {
            'tfidf': fmt_results(results['tfidf'], "TF-IDF"),
            'bm25': fmt_results(results['bm25'], "BM25"),
            'semantic': fmt_results(results['semantic'], "البحث الدلالي (Semantic)"),
        }

    # ── RAG Tab ────────────────────────────────────────────────────────────────
    def do_rag(query: str):
        """Perform RAG comparison."""
        if not query.strip():
            return {"error": "يرجى إدخال سؤال"}

        comparison = rag_system.compare(query)

        # Format retrieved context
        chunks = comparison['rag'].get('retrieved_chunks', [])
        context_parts = []
        for c in chunks:
            context_parts.append(
                f"[{c['rank']}] (تشابه: {c['score']:.4f})\n{c['text']}"
            )
        context_out = "\n\n" + "="*80 + "\n\n".join(context_parts) if context_parts else "لا توجد مقاطع مسترجعة."

        rag_answer = comparison['rag']['answer']
        llm_answer = comparison['llm_only']['answer']

        return {
            'context': context_out,
            'rag_answer': rag_answer,
            'llm_answer': llm_answer,
        }

    # ── Sample Queries ────────────────────────────────────────────────────────
    sample_queries = [
        "من هو الغزالي وما لقبه؟",
        "ما تعريف الغزالي للتصوف؟",
        "ما رأي الغزالي في قدم العالم؟",
        "كيف يختلف الغزالي عن الفلاسفة في مسألة الزمان؟",
        "ما موقف الغزالي من السببية؟",
    ]

    # ── Retrieval Tab Layout ───────────────────────────────────────────────
    retrieval_layout = [
        [sg.Text("استرجاع المعلومات - البحث في الكتاب", font=("Arial", 14, "bold"))],
        [sg.Text("")],
        [
            sg.Text("الاستعلام:", size=(12, 1)),
            sg.InputText(
                size=(70, 1),
                key="-RETRIEVAL_QUERY-",
                tooltip="مثال: ما هو موقف الغزالي من الفلسفة؟",
            ),
        ],
        [
            sg.Text("عدد النتائج (Top-K):", size=(12, 1)),
            sg.Slider(
                range=(1, 10),
                default_value=5,
                orientation="h",
                size=(30, 15),
                key="-TOP_K-",
            ),
        ],
        [sg.Button("🔍 بحث", key="-RETRIEVAL_SEARCH-", size=(15, 1)), sg.Text("")],
        [sg.Text("أمثلة على الاستعلامات:")],
        [
            sg.Listbox(
                values=sample_queries,
                size=(80, 3),
                key="-SAMPLE_QUERIES-",
                select_mode=sg.LISTBOX_SELECT_MODE_SINGLE,
            )
        ],
        [sg.Button("استخدام المثال", key="-USE_EXAMPLE-", size=(15, 1))],
        [sg.Text("")],
        [sg.Text("النتائج:", font=("Arial", 11, "bold"))],
        [
            sg.Column(
                [
                    [sg.Multiline(size=(30, 25), key="-TFIDF_OUT-", disabled=True)],
                    [sg.Text("TF-IDF", font=("Arial", 10, "bold"))],
                ],
                vertical_alignment="top",
            ),
            sg.Column(
                [
                    [sg.Multiline(size=(30, 25), key="-BM25_OUT-", disabled=True)],
                    [sg.Text("BM25", font=("Arial", 10, "bold"))],
                ],
                vertical_alignment="top",
            ),
            sg.Column(
                [
                    [sg.Multiline(size=(30, 25), key="-SEMANTIC_OUT-", disabled=True)],
                    [sg.Text("البحث الدلالي", font=("Arial", 10, "bold"))],
                ],
                vertical_alignment="top",
            ),
        ],
    ]

    # ── RAG Tab Layout ─────────────────────────────────────────────────────
    rag_layout = [
        [sg.Text("الإجابة على الأسئلة - RAG vs LLM", font=("Arial", 14, "bold"))],
        [sg.Text("")],
        [
            sg.Text("السؤال:", size=(12, 1)),
            sg.InputText(
                size=(70, 1),
                key="-RAG_QUERY-",
                tooltip="مثال: ما رأي الغزالي في السببية؟",
            ),
        ],
        [sg.Button("💬 إجابة", key="-RAG_ANSWER-", size=(15, 1)), sg.Text("")],
        [sg.Text("أمثلة على الأسئلة:")],
        [
            sg.Listbox(
                values=sample_queries,
                size=(80, 3),
                key="-SAMPLE_QUESTIONS-",
                select_mode=sg.LISTBOX_SELECT_MODE_SINGLE,
            )
        ],
        [sg.Button("استخدام المثال", key="-USE_QUESTION-", size=(15, 1))],
        [sg.Text("")],
        [sg.Text("المقاطع المسترجعة من الكتاب:", font=("Arial", 11, "bold"))],
        [sg.Multiline(size=(95, 6), key="-RAG_CONTEXT-", disabled=True)],
        [sg.Text("")],
        [sg.Text("المقارنة بين الإجابات:", font=("Arial", 11, "bold"))],
        [
            sg.Column(
                [
                    [sg.Text("إجابة RAG (مع الاسترجاع)", font=("Arial", 10, "bold"))],
                    [sg.Multiline(size=(45, 15), key="-RAG_ANSWER_OUT-", disabled=True)],
                ],
            ),
            sg.Column(
                [
                    [sg.Text("إجابة LLM فقط (بدون استرجاع)", font=("Arial", 10, "bold"))],
                    [sg.Multiline(size=(45, 15), key="-LLM_ANSWER_OUT-", disabled=True)],
                ],
            ),
        ],
    ]

    # ── System Info Tab Layout ─────────────────────────────────────────────
    system_info = """📚 الكتاب
─────────────────────────────────────────────────────────────────────
• العنوان: فلسفة الغزالي
• المؤلف: عباس محمود العقاد
• الموضوع: الفلسفة الإسلامية، الإمام الغزالي، ما بعد الطبيعة
• اللغة: العربية الفصحى

⚙️ معالجة النص
─────────────────────────────────────────────────────────────────────
• تطبيع الحروف العربية (الألف، التاء المربوطة، الألف المقصورة)
• إزالة التشكيل
• تقسيم إلى مقاطع من 2-4 جمل مع تداخل جزئي
• إزالة كلمات التوقف العربية للبحث الكلاسيكي

🤖 نماذج الذكاء الاصطناعي
─────────────────────────────────────────────────────────────────────
• نموذج التمثيل الشعاعي: paraphrase-multilingual-MiniLM-L12-v2
  (متعدد اللغات، يدعم العربية، خفيف الحجم)
• نموذج اللغة: SILMA-Kashif-2B-Instruct-v1.0
  (مفتوح المصدر، يدعم العربية)

🔍 طرق الاسترجاع
─────────────────────────────────────────────────────────────────────
┌──────────┬─────────────┬───────────────────────────────────────┐
│ الطريقة  │ النوع        │ المبدأ                                 │
├──────────┼─────────────┼───────────────────────────────────────┤
│ TF-IDF   │ كلاسيكي     │ تكرار الكلمة × ندرتها               │
│ BM25     │ كلاسيكي     │ تحسين TF-IDF مع طول المستند        │
│ Semantic │ دلالي       │ تشابه جيب التمام بين التمثيلات الشعاعية│
└──────────┴─────────────┴───────────────────────────────────────┘

📋 فهرس FAISS
─────────────────────────────────────────────────────────────────────
• نوع الفهرس: IndexFlatIP
  (الضرب الداخلي = تشابه جيب التمام للمتجهات المعيارية)
• الأبعاد: 384 بُعدًا (MiniLM-L12)
════════════════════════════════════════════════════════════════════"""

    system_layout = [
        [sg.Multiline(system_info, size=(100, 40), disabled=True, key="-SYSTEM_INFO-")]
    ]

    # ── Main Layout ────────────────────────────────────────────────────────
    layout = [
        [
            sg.Text(
                "📚 نظام استرجاع المعلومات والإجابة على الأسئلة بالعربية",
                font=("Arial", 16, "bold"),
                justification="center",
            )
        ],
        [
            sg.Text(
                "الكتاب: فلسفة الغزالي — تأليف عباس محمود العقاد",
                font=("Arial", 12),
                justification="center",
            )
        ],
        [sg.Text("")],
        [
            sg.TabGroup(
                [
                    [
                        sg.Tab("🔍 استرجاع المعلومات", retrieval_layout, key="-TAB_RETRIEVAL-"),
                        sg.Tab("🤖 الإجابة على الأسئلة", rag_layout, key="-TAB_RAG-"),
                        sg.Tab("ℹ️ معلومات النظام", system_layout, key="-TAB_INFO-"),
                    ]
                ],
                tab_location="top",
                selected_title_color="white",
                font=("Arial", 11),
            )
        ],
        [sg.Button("إغلاق", key="-EXIT-", size=(15, 1))],
    ]

    # ── Create Window ──────────────────────────────────────────────────────
    window = sg.Window(
        "نظام RAG للعربية - فلسفة الغزالي",
        layout,
        size=(1000, 750),
        finalize=True,
        element_justification="center",
    )

    # ── Event Loop ─────────────────────────────────────────────────────────
    while True:
        event, values = window.read()

        if event == sg.WINDOW_CLOSED or event == "-EXIT-":
            break

        # ── Retrieval Tab Events ───────────────────────────────────────────
        if event == "-USE_EXAMPLE-":
            if values["-SAMPLE_QUERIES-"]:
                selected = values["-SAMPLE_QUERIES-"][0]
                window["-RETRIEVAL_QUERY-"].update(selected)

        elif event == "-RETRIEVAL_SEARCH-":
            query = values["-RETRIEVAL_QUERY-"]
            top_k = int(values["-TOP_K-"])

            window["-TFIDF_OUT-"].update("جاري البحث...")
            window.refresh()

            try:
                results = do_retrieval(query, top_k)
                if "error" in results:
                    window["-TFIDF_OUT-"].update(results["error"])
                else:
                    window["-TFIDF_OUT-"].update(results["tfidf"])
                    window["-BM25_OUT-"].update(results["bm25"])
                    window["-SEMANTIC_OUT-"].update(results["semantic"])
            except Exception as e:
                window["-TFIDF_OUT-"].update(f"خطأ: {str(e)}")

        # ── RAG Tab Events ────────────────────────────────────────────────
        if event == "-USE_QUESTION-":
            if values["-SAMPLE_QUESTIONS-"]:
                selected = values["-SAMPLE_QUESTIONS-"][0]
                window["-RAG_QUERY-"].update(selected)

        elif event == "-RAG_ANSWER-":
            query = values["-RAG_QUERY-"]

            window["-RAG_CONTEXT-"].update("جاري معالجة السؤال...")
            window.refresh()

            # Run in thread to prevent UI freezing
            def run_rag():
                try:
                    results = do_rag(query)
                    if "error" in results:
                        window["-RAG_CONTEXT-"].update(results["error"])
                    else:
                        window["-RAG_CONTEXT-"].update(results["context"])
                        window["-RAG_ANSWER_OUT-"].update(results["rag_answer"])
                        window["-LLM_ANSWER_OUT-"].update(results["llm_answer"])
                except Exception as e:
                    window["-RAG_CONTEXT-"].update(f"خطأ: {str(e)}")

            thread = threading.Thread(target=run_rag, daemon=True)
            thread.start()

    window.close()


if __name__ == "__main__":
    from book_preparation import prepare_book
    from retrieval import RetrievalSystem
    from rag import ArabicLLM, RAGSystem

    # Configuration
    OUTPUT_DIR = "outputs"
    BOOK_PATH = "data/ghazali_philosophy.txt"
    EMBED_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    LLM_MODEL = "silma-ai/SILMA-Kashif-2B-Instruct-v1.0"

    print("=" * 60)
    print("Arabic RAG System - PySimpleGUI Desktop Interface")
    print("=" * 60)

    print("\n[1/3] Loading book and building index...")
    index, chunks, embeddings, embed_model = prepare_book(
        book_path=BOOK_PATH,
        output_dir=OUTPUT_DIR,
        embedding_model_name=EMBED_MODEL,
    )
    print(f"✓ Book prepared: {len(chunks)} chunks")

    print("\n[2/3] Building retrieval system...")
    retrieval_system = RetrievalSystem(chunks, index, embed_model)
    print("✓ Retrieval system ready")

    print("\n[3/3] Loading LLM...")
    llm = ArabicLLM(model_name=LLM_MODEL)
    rag_system = RAGSystem(llm, retrieval_system.semantic, top_k=3)
    print("✓ LLM loaded")

    print("\n" + "=" * 60)
    print("🖥️ Launching PySimpleGUI desktop interface...")
    print("=" * 60 + "\n")

    build_interface(retrieval_system, rag_system)
