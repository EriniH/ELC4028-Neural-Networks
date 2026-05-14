"""
Task 3: RAG System
- Small open-source LLM (~1-3B params) from HuggingFace
- RAG: retrieve relevant chunks → feed to LLM with context
- LLM-only: answer without retrieval
- Compare both outputs
"""

import torch
from typing import List, Dict, Optional
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline


# ── Model Configuration ────────────────────────────────────────────────────────
# We use Qwen2.5-1.5B-Instruct: multilingual, supports Arabic, ~1.5B params
# Alternative: "silma-ai/SILMA-Kashif-2B-Instruct-v1.0" (2B, good Arabic support)
# Alternative: "microsoft/Phi-3-mini-4k-instruct" (3.8B, good Arabic support) (not tested yet)

DEFAULT_LLM_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"

# Prompt templates
SYSTEM_PROMPT_AR = """أنت مساعد ذكي متخصص في الفلسفة الإسلامية. أجب على الأسئلة بدقة وباللغة العربية."""

RAG_PROMPT_TEMPLATE = """\
أنت مساعد متخصص في الفلسفة الإسلامية. استخدم المقاطع التالية من كتاب "فلسفة الغزالي" للإجابة على السؤال.

المقاطع المسترجعة:
{context}

السؤال: {question}

الإجابة (بالعربية):"""

LLM_ONLY_PROMPT_TEMPLATE = """\
أنت مساعد متخصص في الفلسفة الإسلامية. أجب على السؤال التالي من معرفتك العامة.

السؤال: {question}

الإجابة (بالعربية):"""


# ── LLM Loader ────────────────────────────────────────────────────────────────

class ArabicLLM:
    """Wrapper for a small multilingual LLM that supports Arabic."""

    def __init__(
        self,
        model_name: str = DEFAULT_LLM_MODEL,
        max_new_tokens: int = 300,
        device: Optional[str] = None,
    ):
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens

        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        print(f"[LLM] Loading {model_name} on {self.device}...")
        self._load_model()

    def _load_model(self):
        """Load tokenizer and model."""
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=True,
        )

        # Load in float32 for CPU compatibility; use float16 on GPU
        dtype = torch.float16 if self.device == "cuda" else torch.float32

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=dtype,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
        )
        self.model.eval()

        if self.device == "cuda":
            self.model = self.model.to(self.device)

        print(f"[LLM] Model loaded. Parameters: ~{sum(p.numel() for p in self.model.parameters())/1e9:.1f}B")

    def generate(self, prompt: str, max_new_tokens: Optional[int] = None) -> str:
        """Generate a response for the given prompt."""
        max_tokens = max_new_tokens or self.max_new_tokens

        # Use chat template if available (Qwen, Phi support this)
        if hasattr(self.tokenizer, 'apply_chat_template'):
            try:
                messages = [
                    {"role": "system", "content": SYSTEM_PROMPT_AR},
                    {"role": "user", "content": prompt},
                ]
                formatted = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
            except Exception:
                # Fallback if chat template doesn't support system role
                messages = [
                    {"role": "user", "content": f"{SYSTEM_PROMPT_AR}\n\n{prompt}"},
                ]
                formatted = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
        else:
            formatted = f"{SYSTEM_PROMPT_AR}\n\n{prompt}"

        inputs = self.tokenizer(
            formatted,
            return_tensors="pt",
            truncation=True,
            max_length=2048,
        )

        if self.device == "cuda":
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=False,          # greedy for determinism
                temperature=1.0,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        # Decode only the newly generated tokens
        input_len = inputs['input_ids'].shape[1]
        generated_ids = output_ids[0][input_len:]
        response = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
        return response.strip()


# ── RAG Pipeline ───────────────────────────────────────────────────────────────

class RAGSystem:
    """Retrieval-Augmented Generation system."""

    def __init__(self, llm: ArabicLLM, semantic_retriever, top_k: int = 3):
        self.llm = llm
        self.retriever = semantic_retriever
        self.top_k = top_k

    def _build_context(self, retrieved_chunks: List[Dict]) -> str:
        """Format retrieved chunks into a context string."""
        parts = []
        for i, chunk in enumerate(retrieved_chunks, 1):
            parts.append(f"[مقطع {i}]\n{chunk['text']}")
        return "\n\n".join(parts)

    def answer_with_rag(self, question: str) -> Dict:
        """Answer using RAG: retrieve then generate."""
        # Step 1: Retrieve relevant chunks
        retrieved = self.retriever.search(question, top_k=self.top_k)

        # Step 2: Build context
        context = self._build_context(retrieved)

        # Step 3: Build RAG prompt
        prompt = RAG_PROMPT_TEMPLATE.format(
            context=context,
            question=question,
        )

        # Step 4: Generate
        answer = self.llm.generate(prompt)

        return {
            'question': question,
            'method': 'RAG',
            'retrieved_chunks': retrieved,
            'context_used': context,
            'answer': answer,
        }

    def answer_without_retrieval(self, question: str) -> Dict:
        """Answer using LLM only, no retrieval."""
        prompt = LLM_ONLY_PROMPT_TEMPLATE.format(question=question)
        answer = self.llm.generate(prompt)

        return {
            'question': question,
            'method': 'LLM-Only',
            'answer': answer,
        }

    def compare(self, question: str) -> Dict:
        """Run both RAG and LLM-only for the same question."""
        print(f"\n[RAG] Processing: {question}")

        print("  → LLM-only answer...")
        llm_result = self.answer_without_retrieval(question)

        print("  → RAG answer...")
        rag_result = self.answer_with_rag(question)

        return {
            'question': question,
            'rag': rag_result,
            'llm_only': llm_result,
        }

    def format_comparison(self, comparison: Dict) -> str:
        """Format the comparison output."""
        q = comparison['question']
        rag = comparison['rag']
        llm = comparison['llm_only']

        lines = [
            f"\n{'='*70}",
            f"السؤال: {q}",
            f"{'='*70}",
            "",
            "── المقاطع المسترجعة ──",
        ]
        for chunk in rag['retrieved_chunks']:
            lines.append(f"  [{chunk['rank']}] (score={chunk['score']:.4f}) {chunk['text'][:120]}...")

        lines += [
            "",
            "── إجابة RAG (بالاسترجاع) ──",
            rag['answer'],
            "",
            "── إجابة LLM فقط (بدون استرجاع) ──",
            llm['answer'],
            "",
        ]
        return '\n'.join(lines)


if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
    from book_preparation import prepare_book
    from retrieval import SemanticRetriever

    index, chunks, embeddings, embed_model = prepare_book()
    semantic = SemanticRetriever(index, chunks, embed_model)
    llm = ArabicLLM()
    rag = RAGSystem(llm, semantic)

    result = rag.compare("ما هو موقف الغزالي من السببية؟")
    print(rag.format_comparison(result))
