"""
vector_store.py — ChromaDB Vector Store Management
====================================================
Handles embedding generation and ChromaDB operations: creating collections,
storing embedded chunks, and performing similarity search with scores.

Design decision: ChromaDB was chosen over FAISS (no persistence by default,
no metadata filtering) and Pinecone (requires cloud account, paid tier for
production). ChromaDB provides local persistence, metadata filtering, and
a clean Python API — ideal for a demo-to-production pathway.

Embedding model: all-MiniLM-L6-v2 produces 384-dimensional vectors. It
ranks in the top tier on the MTEB leaderboard for its size class (22M
params), offering an excellent speed/quality ratio for semantic search
over short text chunks.
"""

import os
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

# Load environment variables from .env
load_dotenv()

# --------------- Configuration ---------------
CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")
CHROMA_COLLECTION_NAME = os.getenv("CHROMA_COLLECTION_NAME", "customer_support_docs")
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
RETRIEVAL_TOP_K = int(os.getenv("RETRIEVAL_TOP_K", "3"))


def get_embedding_function():
    """
    Initialize and return the HuggingFace embedding function.

    Uses all-MiniLM-L6-v2 which runs locally (no API calls), ensuring:
    - Zero marginal cost per embedding
    - No data leaving the local machine (privacy)
    - Fast inference (~14ms per sentence on CPU)

    Returns:
        HuggingFaceEmbeddings: Configured embedding function.
    """
    print(f"🔢 Initializing embedding model: {EMBEDDING_MODEL_NAME}")
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        model_kwargs={"device": "cpu"},          # Explicit CPU — portable across environments
        encode_kwargs={"normalize_embeddings": True}  # L2-normalize for cosine similarity
    )
    print(f"   ✅ Embedding model loaded (384-dim vectors)")
    return embeddings


def create_vector_store(chunks):
    """
    Embed document chunks and persist them to ChromaDB.

    This function creates a new ChromaDB collection (or overwrites an existing
    one) with the provided chunks. Each chunk is embedded using
    all-MiniLM-L6-v2 and stored alongside its metadata.

    Args:
        chunks: List of LangChain Document objects (from ingestion.chunk_documents).

    Returns:
        Chroma: The initialized and populated vector store instance.
    """
    print(f"\n💾 Creating vector store in: {CHROMA_PERSIST_DIR}")
    print(f"   Collection: {CHROMA_COLLECTION_NAME}")

    embeddings = get_embedding_function()

    # Create the Chroma vector store from documents
    # persist_directory ensures data survives process restarts
    # collection_metadata specifies cosine distance for proper relevance scoring
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=CHROMA_COLLECTION_NAME,
        persist_directory=CHROMA_PERSIST_DIR,
        collection_metadata={"hnsw:space": "cosine"}
    )

    print(f"   ✅ Stored {len(chunks)} chunks in ChromaDB")
    return vector_store


def load_vector_store():
    """
    Load an existing ChromaDB vector store from disk.

    Used during query time — avoids re-embedding documents on every run.

    Returns:
        Chroma: The loaded vector store instance.

    Raises:
        FileNotFoundError: If the persist directory does not exist.
    """
    if not os.path.exists(CHROMA_PERSIST_DIR):
        raise FileNotFoundError(
            f"❌ Vector store not found at {CHROMA_PERSIST_DIR}. "
            f"Run ingestion first."
        )

    print(f"📂 Loading vector store from: {CHROMA_PERSIST_DIR}")
    embeddings = get_embedding_function()

    vector_store = Chroma(
        collection_name=CHROMA_COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=CHROMA_PERSIST_DIR
    )

    # Verify the collection is not empty
    count = vector_store._collection.count()
    print(f"   ✅ Loaded vector store ({count} chunks available)")
    return vector_store


def retrieve_relevant_chunks(vector_store, query: str, top_k: int = RETRIEVAL_TOP_K):
    """
    Perform similarity search against the vector store and return
    the top-k most relevant chunks with their similarity scores.

    Uses cosine similarity (via L2-normalized embeddings). Scores are
    returned as distances — lower is more similar. We convert to a
    0-1 similarity score for downstream confidence calculation.

    Args:
        vector_store: Chroma vector store instance.
        query: The user's natural language question.
        top_k: Number of results to return (default: 3).

    Returns:
        tuple: (documents, scores)
            - documents: List of Document objects
            - scores: List of float similarity scores (0-1, higher = more similar)
    """
    print(f"\n🔍 Retrieving top-{top_k} chunks for query:")
    print(f"   \"{query}\"")

    # similarity_search_with_relevance_scores returns (doc, score) tuples
    # Score is normalized to [0, 1] where 1 = most similar
    results = vector_store.similarity_search_with_relevance_scores(query, k=top_k)

    documents = []
    scores = []

    for i, (doc, score) in enumerate(results):
        documents.append(doc)
        scores.append(score)
        # Truncate preview for readability in demo
        preview = doc.page_content[:80].replace("\n", " ")
        print(f"   📌 Chunk {i+1} (score: {score:.3f}): \"{preview}...\"")

    return documents, scores
