"""Example entry point for building a knowledge base and answering questions."""
from __future__ import annotations

from agentic_rag.scraping import fetch_url
from agentic_rag.rag_pipeline import RAGPipeline
from agentic_rag.llm_interface import load_llm


def build_kb(urls: list[str]) -> RAGPipeline:
    docs = [{"text": fetch_url(u), "metadata": {"source": u}} for u in urls]
    rag = RAGPipeline()
    rag.ingest(docs)
    return rag


def answer_question(question: str, rag: RAGPipeline) -> str:
    hits = rag.query(question)
    context = "\n".join([h.page_content for h in hits])
    llm = load_llm()
    prompt = f"Context:\n{context}\n\nQuestion:\n{question}\nAnswer:"
    response = llm(prompt, max_length=512, do_sample=False)
    return response[0]["generated_text"]


if __name__ == "__main__":  # pragma: no cover - manual execution
    urls = ["https://example.com"]
    rag_pipeline = build_kb(urls)
    print(answer_question("What is example.com about?", rag_pipeline))
