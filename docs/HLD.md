# High-Level Design (HLD) — RAG Customer Support Assistant

| Field | Value |
|-------|-------|
| **Project** | RAG-Based Customer Support Assistant with LangGraph & HITL |
| **Role** | Senior AI Engineer |
| **Date** | April 2026 |
| **Version** | 1.0 |

---

## 1. Problem Definition

Traditional rule-based chatbots operate on intent classification and slot filling — they match user utterances to predefined patterns and return scripted responses. This architecture fails in three fundamental ways for customer support:

1. **Brittleness**: Any query that falls outside the scripted intent taxonomy receives a generic fallback response. Real customer queries are messy, ambiguous, and context-dependent — they rarely map cleanly to a fixed set of intents.

2. **Knowledge staleness**: The knowledge is baked into the code or training data. Updating product policies, shipping rules, or pricing requires redeploying the bot. In organizations where support documentation changes weekly, this creates an unacceptable maintenance burden.

3. **Lack of graceful degradation**: When a rule-based bot cannot answer, it simply fails. There is no mechanism to recognize uncertainty and route to a human — the customer either gets a wrong answer or a dead end.

**What RAG solves**: Retrieval-Augmented Generation decouples the knowledge source from the reasoning engine. Instead of encoding knowledge into model weights (fine-tuning) or stuffing it into prompts (prompt engineering), RAG retrieves relevant documents at query time and presents them as context to an LLM. This means:

- Knowledge updates require only re-indexing documents — no model retraining
- The LLM reasons over fresh, authoritative source material rather than parametric memory
- Retrieval scores provide a natural confidence signal for routing decisions

**Scope and boundaries**: This system handles single-turn customer support queries over a fixed document corpus (PDF-based knowledge base). It does not handle multi-turn conversations, real-time data, or concurrent multi-user sessions. It includes a Human-in-the-Loop escalation path for queries the system cannot confidently answer.

---

## 2. System Architecture Overview

The system follows a three-stage pipeline: **Ingestion** (offline, batch), **Retrieval + Generation** (online, per-query), and **Escalation** (conditional, human-in-the-loop). These stages are decoupled — ingestion populates the vector store independently of the query-serving path.

The query-serving path is orchestrated by a LangGraph StateGraph, which provides typed state management, conditional branching, and a clear execution trace.

### Architecture Diagram

```
┌─────────────────────────── OFFLINE INGESTION PIPELINE ───────────────────────────┐
│                                                                                    │
│   ┌──────────┐     ┌────────────────────┐     ┌──────────────┐     ┌───────────┐ │
│   │ PDF Docs │────▶│ PyPDFLoader        │────▶│ Text Splitter│────▶│ Embedding │ │
│   │ (data/)  │     │ (page extraction)  │     │ (500/50)     │     │ MiniLM-L6 │ │
│   └──────────┘     └────────────────────┘     └──────────────┘     └─────┬─────┘ │
│                                                                           │       │
│                                                                    ┌──────▼──────┐│
│                                                                    │  ChromaDB   ││
│                                                                    │  (persist)  ││
│                                                                    └──────┬──────┘│
└───────────────────────────────────────────────────────────────────────────┼────────┘
                                                                           │
┌──────────────────────────── ONLINE QUERY PIPELINE ───────────────────────┼────────┐
│                                                                          │        │
│   ┌──────────┐     ┌─────────────────────────────────────────────────────┘        │
│   │ User     │     │                                                              │
│   │ Query    │     │  LangGraph StateGraph (SupportState)                         │
│   └────┬─────┘     │  ┌──────────────────────────────────────────────────────┐    │
│        │           │  │                                                      │    │
│        ▼           │  │  ┌───────────────┐    ┌───────────────┐              │    │
│   ┌─────────┐      │  │  │ retrieve_node │───▶│ generate_node │              │    │
│   │ Embed   │──────┘  │  │               │    │               │              │    │
│   │ Query   │         │  │ ChromaDB      │    │ LLM (GPT-3.5/ │              │    │
│   └─────────┘         │  │ top-3 search  │    │  Gemini)       │              │    │
│                       │  └───────────────┘    └───────┬───────┘              │    │
│                       │                               │                      │    │
│                       │                    ┌──────────▼──────────┐            │    │
│                       │                    │ Conditional Router  │            │    │
│                       │                    │                     │            │    │
│                       │                    │ confidence < 0.4?   │            │    │
│                       │                    │ "escalate" in resp? │            │    │
│                       │                    └──────┬────────┬─────┘            │    │
│                       │                    YES    │        │   NO             │    │
│                       │                    ┌──────▼─────┐  │  ┌──────┐       │    │
│                       │                    │ hitl_node  │  └─▶│ END  │       │    │
│                       │                    │            │     └──────┘       │    │
│                       │                    │ Human      │                    │    │
│                       │                    │ Terminal    │                    │    │
│                       │                    │ Input      │                    │    │
│                       │                    └──────┬─────┘                    │    │
│                       │                           │                          │    │
│                       │                    ┌──────▼─────┐                    │    │
│                       │                    │    END     │                    │    │
│                       │                    └────────────┘                    │    │
│                       │                                                      │    │
│                       └──────────────────────────────────────────────────────┘    │
│                                                                                   │
│   ┌──────────────────┐                                                            │
│   │ Final Response   │◀─── response field from SupportState                       │
│   │ (to customer)    │                                                            │
│   └──────────────────┘                                                            │
└───────────────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Component Descriptions

### 3.1 Document Loader (PyPDFLoader)
The document loader uses LangChain's `PyPDFLoader` to extract text from PDF files on a per-page basis. Each page becomes a `Document` object carrying the raw text content and metadata (source file path, page number). This metadata is preserved through chunking and stored in ChromaDB, enabling source attribution in responses. PyPDFLoader was chosen over alternatives like `UnstructuredPDFLoader` because it produces clean text output for text-heavy documents without requiring heavyweight dependencies (poppler, tesseract).

### 3.2 Chunking Module (RecursiveCharacterTextSplitter)
The chunking module splits page-level documents into 500-character chunks with 50-character overlap. `RecursiveCharacterTextSplitter` is used because it attempts to split on natural language boundaries (paragraphs, then sentences, then words) before resorting to raw character splits. This preserves semantic coherence within each chunk. The 500-character size is tuned for FAQ-style content where individual answers typically span 2-4 sentences — large enough to capture a complete answer, small enough for precise retrieval. The 50-character overlap (10% of chunk size) prevents hard semantic breaks at boundaries.

### 3.3 Embedding Model (all-MiniLM-L6-v2)
The embedding model maps text chunks and queries into a shared 384-dimensional vector space where semantic similarity corresponds to vector proximity. `all-MiniLM-L6-v2` was selected because it ranks in the top tier of the MTEB (Massive Text Embedding Benchmark) for its size class (22M parameters, 80MB). It runs locally on CPU with ~14ms inference per sentence, eliminating API costs and data privacy concerns. For this customer support use case, where chunks are short and semantically distinct, its quality is comparable to larger models while being 10x faster.

### 3.4 ChromaDB Vector Store
ChromaDB serves as the persistent vector store for embedded document chunks. It was chosen over FAISS (in-memory only by default, no metadata filtering, requires explicit serialization), Pinecone (cloud-hosted, requires account provisioning, introduces network latency and cost), and Weaviate (heavyweight infrastructure, over-engineered for a single-collection use case). ChromaDB provides local file-based persistence, built-in metadata filtering, and a clean Python API. Its `persist_directory` parameter ensures indexed data survives process restarts without manual serialization code.

### 3.5 Retriever
The retriever performs similarity search against ChromaDB, returning the top-3 most relevant chunks along with their relevance scores. The search uses cosine similarity (via L2-normalized embeddings) to measure semantic proximity between the query embedding and stored chunk embeddings. Relevance scores are used downstream as a confidence signal for the routing decision. The `top_k=3` parameter limits context to the three most relevant chunks, preventing context dilution in the LLM prompt.

### 3.6 LLM Layer (GPT-3.5-turbo / Gemini)
The LLM generates natural language responses grounded in the retrieved context. GPT-3.5-turbo is the default because it offers the best cost-per-token ratio for the support use case, where responses are short (2-4 sentences) and the reasoning task is straightforward (information extraction, not complex inference). Temperature is set to 0.3 — low enough to minimize hallucination, high enough to avoid robotic repetition. The system prompt explicitly instructs the LLM to flag uncertainty with "ESCALATE", creating a feedback signal for the routing layer. Google Gemini is supported as an alternative via a simple environment variable toggle.

### 3.7 LangGraph Workflow Engine
LangGraph orchestrates the three-node processing pipeline as a `StateGraph`. It was chosen over LangChain's `SequentialChain` and `LCEL` (LangChain Expression Language) because this system requires conditional branching — the HITL node is only invoked when confidence is low. LangChain chains are inherently linear; bolting conditional logic onto them requires workarounds (`RunnableBranch`, custom callbacks) that obscure the control flow. LangGraph makes branching a first-class concept: nodes, edges, and conditional edges are declared explicitly, producing a graph that is inspectable, testable, and extensible without refactoring.

### 3.8 Routing Layer (Conditional Edge)
The routing layer implements a conditional edge after `generate_node` that inspects the graph state and routes to either `hitl_node` (escalation) or `END` (direct response). The routing function evaluates three conditions: (1) average retrieval confidence below 0.4, (2) presence of "escalate" in the LLM response, (3) presence of uncertainty phrases ("I don't know", "not sure"). This multi-signal approach is more robust than relying on any single indicator — it catches both retrieval failures (low confidence) and generation failures (LLM-detected uncertainty).

### 3.9 HITL Module
The Human-in-the-Loop module presents escalated queries to a human operator via terminal input. The operator sees the original query, the LLM's attempted response, and the confidence score — enough context to provide an informed answer without re-reading the source documents. The human response replaces the LLM response in the graph state and is returned to the customer. This design acknowledges that AI systems have failure modes, and provides a structured fallback path that maintains service quality.

---

## 4. Data Flow

### 4.1 Ingestion Flow (Offline)

1. **PDF Loading**: `PyPDFLoader` reads each page of the PDF, producing a `Document` object per page with `page_content` (raw text) and `metadata` (source path, page number).
2. **Chunking**: `RecursiveCharacterTextSplitter` splits each page into ~500-character chunks with 50-character overlap, producing smaller `Document` objects that inherit the parent's metadata.
3. **Embedding**: Each chunk's `page_content` is passed through `all-MiniLM-L6-v2`, producing a 384-dimensional normalized vector.
4. **Storage**: The vector, text content, and metadata are stored together in ChromaDB's persistent collection. The collection is written to disk at `CHROMA_PERSIST_DIR`.

### 4.2 Query Flow (Online)

1. **Query Embedding**: The user's query string is embedded using the same `all-MiniLM-L6-v2` model, producing a 384-dimensional query vector.
2. **Retrieval**: ChromaDB performs cosine similarity search, returning the top-3 chunks and their relevance scores. Scores and chunk texts are written to `SupportState`.
3. **Average Confidence**: The mean of the top-3 relevance scores is computed and stored as `state["confidence"]`.
4. **Generation**: The query and retrieved chunks are formatted into a prompt and passed to the LLM. The response is written to `state["response"]`.
5. **Routing**: `route_after_generate` inspects `confidence` and `response`. If escalation criteria are met, control flows to `hitl_node`; otherwise, the graph terminates.
6. **HITL (conditional)**: The human operator provides a response, which overwrites `state["response"]` and sets `state["escalate"] = True`.
7. **Output**: The final `SupportState` is returned, containing the response, confidence, and escalation metadata.

---

## 5. Technology Choices

| Component | Technology | Reason |
|-----------|-----------|--------|
| Language | Python 3.10+ | De-facto standard for ML/AI; first-class support from all libraries used |
| Document Loader | PyPDFLoader (LangChain) | Clean text extraction for text-heavy PDFs; no external system dependencies |
| Text Splitter | RecursiveCharacterTextSplitter | Splits on natural language boundaries; preserves semantic coherence |
| Embedding Model | all-MiniLM-L6-v2 (HuggingFace) | Top-tier MTEB ranking for size class; runs locally; 384-dim; zero API cost |
| Vector Store | ChromaDB | Local persistence; metadata filtering; clean Python API; no cloud dependency |
| LLM | GPT-3.5-turbo / Gemini | Best cost/quality for short factual responses; Gemini as zero-cost alternative |
| Orchestration | LangGraph (StateGraph) | First-class conditional branching; typed state; inspectable graph topology |
| Configuration | python-dotenv | Simple .env-based config; no external config server needed |
| CLI | argparse | Standard library; no additional dependencies |

---

## 6. Scalability Considerations

### Large Document Handling
The current system ingests all PDFs into a single ChromaDB collection. For large document corpora (10,000+ chunks), ChromaDB's HNSW index provides sub-linear search time (O(log n) vs. brute-force O(n)). For corpora exceeding local memory, the system can be extended to use ChromaDB's client-server mode, which offloads storage to a dedicated server process. Metadata filtering can be added to scope retrieval to specific document categories, reducing the search space.

### High Query Load
The current implementation is single-threaded and synchronous. For concurrent query handling, the architecture supports horizontal scaling: the graph execution is stateless (all state is in `SupportState`), so multiple graph instances can run in parallel behind a FastAPI/uvicorn server. ChromaDB supports concurrent reads. The embedding model can be shared across threads (it is thread-safe after loading). The primary bottleneck under load is the LLM API call, which can be managed via rate limiting, request queuing, and provider-side batching.

### Latency Optimization
End-to-end latency is dominated by two components: embedding generation (~14ms on CPU for MiniLM) and LLM inference (~800ms–2s for GPT-3.5-turbo). ChromaDB retrieval adds <5ms for collections under 100K vectors. To reduce latency: (1) pre-compute and cache query embeddings for common queries, (2) use streaming LLM responses to reduce perceived latency, (3) use GPU-accelerated embedding if available, (4) consider local LLM inference (e.g., Ollama with Mistral-7B) to eliminate network round-trip.

---

*Document version 1.0 — April 2026*
