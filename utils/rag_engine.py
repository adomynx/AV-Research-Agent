"""
RAG Engine — Builds a vector store from uploaded documents and enables
semantic search for the Research Agent to query uploaded papers.
"""

from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document


class RAGEngine:
    """
    RAG Engine that indexes uploaded research papers and provides
    semantic search capabilities for the Research Agent.

    Uses FAISS for fast local vector similarity search and
    OpenAI embeddings for document/query encoding.
    """

    def __init__(self, documents: list[Document], api_key: str):
        """
        Initialize the RAG engine with documents.

        Args:
            documents: List of chunked LangChain Documents
            api_key: OpenAI API key for embeddings
        """
        self.embeddings = OpenAIEmbeddings(
            api_key=api_key,
            model="text-embedding-3-small"
        )

        if documents:
            self.vectorstore = FAISS.from_documents(
                documents,
                self.embeddings
            )
            self.retriever = self.vectorstore.as_retriever(
                search_type="similarity",
                search_kwargs={"k": 4}
            )
        else:
            self.vectorstore = None
            self.retriever = None

    def query(self, question: str, k: int = 4) -> str:
        """
        Query the vector store for relevant document chunks.

        Args:
            question: The search query
            k: Number of results to return

        Returns:
            Concatenated text of relevant document chunks
        """
        if not self.retriever:
            return ""

        try:
            docs = self.retriever.invoke(question)

            if not docs:
                return ""

            results = []
            for i, doc in enumerate(docs, 1):
                source = doc.metadata.get("source", "Unknown paper")
                page = doc.metadata.get("page", "?")
                results.append(
                    f"[{source}, p.{page}]: {doc.page_content}"
                )

            return "\n\n".join(results)

        except Exception as e:
            print(f"[RAGEngine] Query error: {e}")
            return ""

    def get_doc_count(self) -> int:
        """Return the number of indexed document chunks."""
        if self.vectorstore:
            return self.vectorstore.index.ntotal
        return 0
