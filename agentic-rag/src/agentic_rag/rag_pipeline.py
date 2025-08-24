"""Retrieval-augmented generation pipeline using Chroma DB."""
from __future__ import annotations

from typing import Iterable

from chromadb import Client
from chromadb.config import Settings
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma


class RAGPipeline:
    """A minimal wrapper around Chroma for text retrieval."""

    def __init__(self, persist_dir: str = "chroma_db") -> None:
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        self.client = Client(
            Settings(chroma_db_impl="duckdb+parquet", persist_directory=persist_dir)
        )
        self.db = Chroma(client=self.client, embedding_function=self.embeddings)

    def ingest(self, docs: Iterable[dict]) -> None:
        """Add documents to the database."""
        for doc in docs:
            self.db.add_texts([doc["text"]], metadatas=doc.get("metadata"))

    def query(self, question: str, top_k: int = 3):
        """Search similar documents."""
        return self.db.similarity_search(question, k=top_k)
