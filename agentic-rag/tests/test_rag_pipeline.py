import pytest

chromadb = pytest.importorskip("chromadb")
langchain = pytest.importorskip("langchain")

from agentic_rag.rag_pipeline import RAGPipeline


def test_ingest_and_query(tmp_path):
    rag = RAGPipeline(persist_dir=tmp_path)
    docs = [{"text": "Chroma DB is a vector database", "metadata": {"source": "test"}}]
    rag.ingest(docs)
    results = rag.query("vector database")
    assert results, "No results returned"
    assert "vector database" in results[0].page_content
