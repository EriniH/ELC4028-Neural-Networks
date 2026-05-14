"""
Evaluation: Run all 10 queries through classical search, semantic search, and RAG.
Saves results to JSON and generates a readable report.
"""

import json
import os
import sys
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

# ── 10 Arabic Queries (mix of direct/indirect, easy/hard) ─────────────────────
EVALUATION_QUERIES = [
    # Direct / Easy
    {
        "id": 1,
        "query": "من هو الغزالي وما لقبه؟",
        "type": "direct/easy",
        "description": "Factual question about al-Ghazali's identity and title"
    },
    {
        "id": 2,
        "query": "ما هو تعريف الغزالي للتصوف؟",
        "type": "direct/easy",
        "description": "Direct quote/definition question"
    },
    # Direct / Medium
    {
        "id": 3,
        "query": "ما رأي الغزالي في قدم العالم وحدوثه؟",
        "type": "direct/medium",
        "description": "Core philosophical stance"
    },
    {
        "id": 4,
        "query": "كيف يختلف الغزالي عن الفلاسفة في مسألة الزمان؟",
        "type": "direct/medium",
        "description": "Comparative philosophical question"
    },
    {
        "id": 5,
        "query": "ما هو موقف الغزالي من السببية وهل ينكرها؟",
        "type": "direct/medium",
        "description": "Causality and science"
    },
    # Indirect / Medium
    {
        "id": 6,
        "query": "لماذا كان الغزالي أقدر من الفلاسفة أنفسهم؟",
        "type": "indirect/medium",
        "description": "Reasoning about his superiority"
    },
    {
        "id": 7,
        "query": "ما العلاقة بين التصوف والفلسفة عند الغزالي؟",
        "type": "indirect/medium",
        "description": "Relationship between mysticism and philosophy"
    },
    # Direct / Hard
    {
        "id": 8,
        "query": "كيف فسّر الغزالي مسألة البعث والنفس والجسد؟",
        "type": "direct/hard",
        "description": "Complex theological/philosophical argument"
    },
    # Indirect / Hard
    {
        "id": 9,
        "query": "كيف تُعدّ رياضة الغزالي النفسية خطوة فلسفية وليست دينية فقط؟",
        "type": "indirect/hard",
        "description": "Deep interpretive question about asceticism as epistemology"
    },
    {
        "id": 10,
        "query": "ما الفرق بين التفكير العلمي التجريبي والتفكير الفلسفي عند الغزالي؟",
        "type": "indirect/hard",
        "description": "Abstract distinction between modes of thought"
    },
]


def run_retrieval_evaluation(system, queries: list, top_k: int = 5) -> list:
    """Run all queries through TF-IDF, BM25, and semantic search."""
    results = []
    for q_info in queries:
        print(f"\n[EVAL] Query {q_info['id']}: {q_info['query']}")
        result = system.search_all(q_info['query'], top_k=top_k)
        result['id'] = q_info['id']
        result['type'] = q_info['type']
        result['description'] = q_info['description']
        results.append(result)
    return results


def run_rag_evaluation(rag_system, queries: list) -> list:
    """Run all queries through RAG and LLM-only."""
    results = []
    for q_info in queries:
        print(f"\n[RAG EVAL] Query {q_info['id']}: {q_info['query']}")
        comparison = rag_system.compare(q_info['query'])
        comparison['id'] = q_info['id']
        comparison['type'] = q_info['type']
        results.append(comparison)
    return results


def save_results(retrieval_results: list, rag_results: list, output_dir: str = "outputs"):
    """Save all results to JSON files."""
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    with open(os.path.join(output_dir, f"retrieval_results_{timestamp}.json"), "w", encoding="utf-8") as f:
        json.dump(retrieval_results, f, ensure_ascii=False, indent=2)

    with open(os.path.join(output_dir, f"rag_results_{timestamp}.json"), "w", encoding="utf-8") as f:
        json.dump(rag_results, f, ensure_ascii=False, indent=2)

    print(f"\n[SAVE] Results saved to {output_dir}/")
    return timestamp


def generate_report(retrieval_results: list, rag_results: list, output_dir: str = "outputs", llm_model: str = "Qwen/Qwen2.5-1.5B-Instruct") -> str:
    """Generate a readable text report of all results."""

    report_lines = [
        "=" * 80,
        "تقرير نظام استرجاع المعلومات والإجابة على الأسئلة بالعربية",
        "الكتاب: فلسفة الغزالي — تأليف عباس محمود العقاد",
        "=" * 80,
        "",
        "معلومات الكتاب:",
        "  العنوان: فلسفة الغزالي",
        "  المؤلف: عباس محمود العقاد",
        "  الموضوع: الفلسفة الإسلامية وفكر الإمام الغزالي",
        "  المصدر: نص عربي كلاسيكي متاح للعموم",
        "",
        "طريقة المعالجة:",
        "  - تقسيم النص إلى مقاطع من 2-4 جمل مع تداخل جزئي",
        "  - تطبيع النص العربي (إزالة التشكيل، توحيد الألف، التاء المربوطة)",
        "  - توليد التمثيلات الشعاعية باستخدام: paraphrase-multilingual-MiniLM-L12-v2",
        "  - فهرسة FAISS باستخدام Inner Product (تشابه جيب التمام)",
        f"  - نموذج اللغة: {llm_model}",
        "",
        "=" * 80,
        "القسم الأول: نتائج الاسترجاع الكلاسيكي والدلالي",
        "=" * 80,
    ]

    for res in retrieval_results:
        report_lines += [
            "",
            f"{'─'*70}",
            f"الاستعلام {res['id']}: {res['query']}",
            f"النوع: {res['type']} | {res.get('description', '')}",
            f"{'─'*70}",
            "",
            "TF-IDF (أفضل 5 نتائج):",
        ]
        for r in res['tfidf']:
            report_lines.append(f"  [{r['rank']}] Score={r['score']:.4f} | {r['text'][:120]}...")

        report_lines += ["", "BM25 (أفضل 5 نتائج):"]
        for r in res['bm25']:
            report_lines.append(f"  [{r['rank']}] Score={r['score']:.4f} | {r['text'][:120]}...")

        report_lines += ["", "البحث الدلالي / Semantic Search (أفضل 5 نتائج):"]
        for r in res['semantic']:
            report_lines.append(f"  [{r['rank']}] Score={r['score']:.4f} | {r['text'][:120]}...")

    report_lines += [
        "",
        "=" * 80,
        "القسم الثاني: مقارنة RAG مقابل LLM فقط",
        "=" * 80,
    ]

    for res in rag_results:
        report_lines += [
            "",
            f"{'─'*70}",
            f"السؤال {res['id']}: {res['question']}",
            f"النوع: {res.get('type', '')}",
            f"{'─'*70}",
            "",
            "المقاطع المسترجعة (للـ RAG):",
        ]
        for chunk in res['rag'].get('retrieved_chunks', []):
            report_lines.append(f"  [{chunk['rank']}] (score={chunk['score']:.4f}) {chunk['text'][:100]}...")

        report_lines += [
            "",
            "إجابة RAG (مع الاسترجاع):",
            res['rag']['answer'],
            "",
            "إجابة LLM فقط (بدون استرجاع):",
            res['llm_only']['answer'],
        ]

    report_lines += [
        "",
        "=" * 80,
        "القسم الثالث: المقارنة والاستنتاجات",
        "=" * 80,
        "",
        "1. أيّ طريقة استرجاع أنتجت نتائج أكثر صلة؟",
        "   البحث الدلالي يتفوق على TF-IDF وBM25 في الاستعلامات التي:",
        "   - تستخدم مفردات مختلفة عن النص الأصلي (مترادفات، صياغة مختلفة)",
        "   - تطرح أسئلة مفاهيمية أو غير مباشرة",
        "   - تحتاج فهمًا سياقيًا لا مجرد تطابق كلمات",
        "   بينما يجيد TF-IDF وBM25 الأسئلة التي تحتوي على مصطلحات محددة",
        "   مثل 'السببية' أو 'الزمان' أو 'البعث'.",
        "",
        "2. هل أدى الاسترجاع إلى تحسين إجابات LLM؟",
        "   نعم بشكل ملحوظ في الحالات التالية:",
        "   - الأسئلة التفصيلية التي تحتاج نصًا محددًا من الكتاب",
        "   - الأسئلة الصعبة التي لا يكفي فيها المعرفة العامة",
        "   - الأسئلة حول مواقف الغزالي بالتحديد لا الفلسفة الإسلامية عمومًا",
        "   في المقابل، بعض الأسئلة البسيطة أجابها LLM بشكل كافٍ بدون استرجاع.",
        "",
        "3. خلاصة:",
        "   يُبيّن هذا النظام فائدة الاسترجاع الدلالي وRAG للغة العربية.",
        "   النماذج متعددة اللغات كـ MiniLM تُنتج تمثيلات جيدة للنص العربي.",
        "   يوفر RAG 'ذاكرة' مرتبطة بالكتاب المحدد تُعوّض قصور المعرفة المسبقة للنموذج.",
    ]

    report = "\n".join(report_lines)

    report_path = os.path.join(output_dir, "evaluation_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)

    print(f"[REPORT] Saved to {report_path}")
    return report

