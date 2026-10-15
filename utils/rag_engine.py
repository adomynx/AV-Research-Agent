"""
RAG Engine — hybrid retrieval + reranking over uploaded research papers.

Pipeline:
    query
      ├─ BM25 (sparse / keyword)  ─┐
      │                            ├─ Ensemble (weighted union)
      └─ FAISS (dense / semantic) ─┘            │
                                                ▼
                                   FlashRank cross-encoder rerank
                                                │
                                                ▼
                                   top-k grounded chunks (+ scores)

Why hybrid: dense vectors capture meaning but miss exact tokens (e.g. "nuScenes
NDS", "ISO 26262"); BM25 nails those literals. Combining them, then reranking
with a cross-encoder, gives markedly better retrieval than dense-only — the
kind of measurable upgrade the eval harness is built to quantify.

Embeddings are provider-agnostic (free local model by default), so RAG needs no
API key.
"""

from __future__ import annotations

from langchain_classic.retrievers import ContextualCompressionRetriever, EnsembleRetriever
from langchain_community.document_compressors import FlashrankRerank
from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

from utils.llm import get_embeddings

# Retrieve this many from each retriever before reranking narrows the set.
_CANDIDATE_K = 8
# Keep this many reranked chunks for the final context.
_FINAL_K = 4
# Ensemble weights: dense slightly favoured over sparse.
_WEIGHTS = (0.4, 0.6)  # (bm25, dense)


class RAGEngine:
    """
    Hybrid (BM25 + dense) retriever with cross-encoder reranking over uploaded
    research papers. Returns grounded chunks with source + page + score.
    """

    def __init__(self, documents: list[Document]):
        self.embeddings = get_embeddings()
        self.retriever = None
        self._vectorstore = None

        if not documents:
            return

        dense = FAISS.from_documents(documents, self.embeddings).as_retriever(
            search_kwargs={"k": _CANDIDATE_K}
        )
        self._vectorstore = dense.vectorstore

        sparse = BM25Retriever.from_documents(documents)
        sparse.k = _CANDIDATE_K

        ensemble = EnsembleRetriever(
            retrievers=[sparse, dense], weights=list(_WEIGHTS)
        )

        # Cross-encoder reranker (local ONNX model, downloaded once).
        reranker = FlashrankRerank(top_n=_FINAL_K)
        self.retriever = ContextualCompressionRetriever(
            base_compressor=reranker, base_retriever=ensemble
        )

    def query(self, question: str) -> str:
        """
        Retrieve and rerank the most relevant chunks for a question.

        Returns concatenated text with inline source/page citations and the
        reranker's relevance score, or "" if nothing is indexed.
        """
        if not self.retriever:
            return ""

        try:
            docs = self.retriever.invoke(question)
        except Exception as e:  # noqa: BLE001
            print(f"[RAGEngine] Query error: {e}")
            return ""

        if not docs:
            return ""

        results = []
        for doc in docs:
            source = doc.metadata.get("source", "Unknown paper")
            page = doc.metadata.get("page", "?")
            score = doc.metadata.get("relevance_score")
            score_str = f", score={score:.2f}" if isinstance(score, (int, float)) else ""
            results.append(f"[{source}, p.{page}{score_str}]: {doc.page_content}")
        return "\n\n".join(results)

    def get_doc_count(self) -> int:
        """Return the number of indexed document chunks."""
        if self._vectorstore is not None:
            return self._vectorstore.index.ntotal
        return 0
