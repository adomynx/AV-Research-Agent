"""
PDF Loader — Handles ingestion and chunking of uploaded research papers.
Uses PyPDF2 for extraction and LangChain text splitters for chunking.
"""

import io
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document


def load_pdf_documents(uploaded_files) -> list[Document]:
    """
    Load and chunk uploaded PDF files into LangChain Documents.

    Args:
        uploaded_files: List of Streamlit UploadedFile objects

    Returns:
        List of chunked LangChain Document objects
    """
    all_docs = []
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", ". ", " ", ""]
    )

    for uploaded_file in uploaded_files:
        try:
            # Write to temp file for PyPDF processing
            import tempfile
            import os

            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(uploaded_file.read())
                tmp_path = tmp.name

            # Load and extract text
            loader = PyPDFLoader(tmp_path)
            pages = loader.load()

            # Add source metadata
            for page in pages:
                page.metadata["source"] = uploaded_file.name

            # Chunk the documents
            chunks = splitter.split_documents(pages)
            all_docs.extend(chunks)

            # Cleanup temp file
            os.unlink(tmp_path)

        except Exception as e:
            print(f"[PDFLoader] Error processing {uploaded_file.name}: {e}")
            continue

    return all_docs
