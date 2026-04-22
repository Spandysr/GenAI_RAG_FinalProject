"""
ingestion.py — Document Ingestion Pipeline
============================================
Handles PDF loading and chunking using LangChain's PyPDFLoader
and RecursiveCharacterTextSplitter. This module is the entry point
for converting raw PDF documents into chunk-sized text fragments
ready for embedding.

Design decision: chunk_size=500 balances capturing enough context
per chunk (a full paragraph) against keeping chunks small enough
for precise retrieval. overlap=50 prevents hard semantic breaks
at chunk boundaries.
"""

import os
import glob
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


# --------------- Configuration Constants ---------------
CHUNK_SIZE = 500      # Characters per chunk — tuned for FAQ-style content
CHUNK_OVERLAP = 50    # Overlap prevents losing context at boundaries


def load_pdf(file_path: str):
    """
    Load a single PDF file and return a list of LangChain Document objects.
    Each Document corresponds to one page of the PDF and carries page-level
    metadata (source path, page number).

    Args:
        file_path: Absolute or relative path to the PDF file.

    Returns:
        List[Document]: One Document per page, with .page_content and .metadata.

    Raises:
        FileNotFoundError: If the specified PDF does not exist.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"❌ PDF not found: {file_path}")

    print(f"📄 Loading PDF: {file_path}")
    loader = PyPDFLoader(file_path)
    pages = loader.load()
    print(f"   ✅ Loaded {len(pages)} page(s)")
    return pages


def load_pdfs_from_directory(directory: str):
    """
    Discover and load all PDF files from a directory (non-recursive).

    Args:
        directory: Path to the folder containing PDF files.

    Returns:
        List[Document]: Combined list of page-level Documents from all PDFs.

    Raises:
        FileNotFoundError: If the directory does not exist.
        ValueError: If no PDFs are found in the directory.
    """
    if not os.path.isdir(directory):
        raise FileNotFoundError(f"❌ Directory not found: {directory}")

    pdf_paths = glob.glob(os.path.join(directory, "*.pdf"))
    if not pdf_paths:
        raise ValueError(f"❌ No PDF files found in: {directory}")

    print(f"📂 Found {len(pdf_paths)} PDF(s) in {directory}")

    all_pages = []
    for path in pdf_paths:
        all_pages.extend(load_pdf(path))

    print(f"📚 Total pages loaded: {len(all_pages)}")
    return all_pages


def chunk_documents(documents, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP):
    """
    Split page-level Documents into smaller, retrieval-friendly chunks.

    Uses RecursiveCharacterTextSplitter which tries to split on natural
    boundaries (paragraphs → sentences → words) before falling back to
    raw character splits. This preserves semantic coherence within chunks.

    Args:
        documents: List of LangChain Document objects (page-level).
        chunk_size: Maximum characters per chunk (default: 500).
        chunk_overlap: Character overlap between consecutive chunks (default: 50).

    Returns:
        List[Document]: Chunked Documents with inherited metadata.
    """
    print(f"\n✂️  Chunking documents (size={chunk_size}, overlap={chunk_overlap})")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        # Split hierarchy: double newline → single newline → sentence → word
        separators=["\n\n", "\n", ". ", " ", ""]
    )

    chunks = splitter.split_documents(documents)
    print(f"   ✅ Created {len(chunks)} chunks from {len(documents)} page(s)")

    # Display a preview of the first chunk for demo visibility
    if chunks:
        preview = chunks[0].page_content[:120].replace("\n", " ")
        print(f"   📌 First chunk preview: \"{preview}...\"")

    return chunks


def ingest_pdf(file_path: str):
    """
    End-to-end convenience function: load a single PDF and chunk it.

    Args:
        file_path: Path to the PDF file.

    Returns:
        List[Document]: Chunked Documents ready for embedding.
    """
    pages = load_pdf(file_path)
    chunks = chunk_documents(pages)
    return chunks


def ingest_directory(directory: str):
    """
    End-to-end convenience function: load all PDFs in a directory and chunk them.

    Args:
        directory: Path to the directory containing PDF files.

    Returns:
        List[Document]: Combined chunked Documents from all PDFs.
    """
    pages = load_pdfs_from_directory(directory)
    chunks = chunk_documents(pages)
    return chunks
