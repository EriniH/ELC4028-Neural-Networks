#!/usr/bin/env python3
"""
Main entry point for the Arabic IR & RAG system.
Book: فلسفة الغزالي (Al-Ghazali's Philosophy) by Abbas Mahmoud Al-Aqqad

Usage:
    python main.py --mode evaluate     # Run full evaluation on 10 queries
    python main.py --mode prepare      # Prepare book index only
    python main.py --query "سؤالك"    # Quick single query (saves results to JSON file)
"""

import argparse
import os
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(__file__))

BOOK_PATH   = "data/ghazali_philosophy.txt"
#OUTPUT_DIR  = "outputs"
OUTPUT_DIR  = "outputs_Silma"
EMBED_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
# LLM_MODEL   = "Qwen/Qwen2.5-1.5B-Instruct"
LLM_MODEL   = "silma-ai/SILMA-Kashif-2B-Instruct-v1.0"


def load_system(skip_llm: bool = False):
    """Load all components."""
    from book_preparation import prepare_book
    from retrieval import RetrievalSystem

    print("\n" + "="*60)
    print("Arabic Information Retrieval & RAG System")
    print("Book: فلسفة الغزالي — عباس محمود العقاد")
    print("="*60)

    # Prepare book
    index, chunks, embeddings, embed_model = prepare_book(
        book_path=BOOK_PATH,
        output_dir=OUTPUT_DIR,
        embedding_model_name=EMBED_MODEL,
    )
    print(f"[INFO] Book prepared: {len(chunks)} chunks")

    # Retrieval system
    retrieval_system = RetrievalSystem(chunks, index, embed_model)

    if skip_llm:
        return retrieval_system, None, chunks

    # LLM + RAG
    from rag import ArabicLLM, RAGSystem
    llm = ArabicLLM(model_name=LLM_MODEL)
    rag_system = RAGSystem(llm, retrieval_system.semantic, top_k=3)

    return retrieval_system, rag_system, chunks


def mode_prepare():
    """Just prepare the book index."""
    from book_preparation import prepare_book
    index, chunks, embeddings, model = prepare_book(
        book_path=BOOK_PATH,
        output_dir=OUTPUT_DIR,
        embedding_model_name=EMBED_MODEL,
        force_rebuild=True,
    )
    print(f"\n✓ Book prepared: {len(chunks)} chunks, index saved to {OUTPUT_DIR}/")





def mode_evaluate():
    """Run full evaluation."""
    retrieval_system, rag_system, chunks = load_system(skip_llm=False)
    from evaluation import (
        EVALUATION_QUERIES,
        run_retrieval_evaluation,
        run_rag_evaluation,
        save_results,
        generate_report,
    )

    print("\n[EVAL] Running retrieval evaluation...")
    retrieval_results = run_retrieval_evaluation(retrieval_system, EVALUATION_QUERIES)

    print("\n[EVAL] Running RAG evaluation...")
    rag_results = run_rag_evaluation(rag_system, EVALUATION_QUERIES)

    print("\n[EVAL] Generating report...")
    save_results(retrieval_results, rag_results, OUTPUT_DIR)
    report = generate_report(retrieval_results, rag_results, OUTPUT_DIR, llm_model=LLM_MODEL)
    print("\n" + report[:3000])   # print first part
    print(f"\n[DONE] Full report saved to {OUTPUT_DIR}/evaluation_report.txt")


def mode_query(query: str):
    """Quick single query through all methods, save to file."""
    import json
    from datetime import datetime
    
    retrieval_system, rag_system, chunks = load_system(skip_llm=True)

    print(f"\nQuery: {query}")
    print("="*60)

    results = retrieval_system.search_all(query, top_k=5)
    print(retrieval_system.format_results(results))
    
    # Save results to JSON file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(OUTPUT_DIR, f"query_results_{timestamp}.json")
    with open(output_file, 'w', encoding='utf-8') as f:
        # Convert results to JSON-serializable format
        json_results = {
            'query': results['query'],
            'tfidf': results['tfidf'],
            'bm25': results['bm25'],
            'semantic': results['semantic'],
        }
        json.dump(json_results, f, ensure_ascii=False, indent=2)
    print(f"\n✓ Results saved to {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Arabic IR & RAG System — فلسفة الغزالي")
    parser.add_argument(
        "--mode",
        choices=["evaluate", "prepare"],
        default="evaluate",
        help="Operating mode",
    )
    parser.add_argument("--query", type=str, help="Quick single query (saves results to JSON file)")
    args = parser.parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    if args.query:
        mode_query(args.query)
    elif args.mode == "prepare":
        mode_prepare()
    else:
        mode_evaluate()


if __name__ == "__main__":
    main()
