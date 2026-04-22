# RAG Customer Support Assistant

> A production-grade Retrieval-Augmented Generation (RAG) system for customer support, built with LangGraph, ChromaDB, and Human-in-the-Loop escalation.

## 🏗️ Architecture

```
User Query → LangGraph Workflow → ChromaDB Retrieval → LLM Generation → Response
                                                                ↓
                                                    [Low Confidence?]
                                                         ↓ Yes
                                                    HITL Escalation
```

## 📁 Project Structure

```
RAG_project/
├── src/
│   ├── __init__.py         # Package init
│   ├── ingestion.py        # PDF loading & chunking
│   ├── vector_store.py     # ChromaDB operations & embeddings
│   ├── graph.py            # LangGraph workflow (nodes, routing, state)
│   ├── hitl.py             # Human-in-the-Loop escalation logic
│   └── app.py              # CLI application entry point
├── data/                   # PDF documents for ingestion
├── docs/                   # Project documentation
│   ├── HLD.md              # High-Level Design
│   ├── LLD.md              # Low-Level Design
│   └── Technical_Doc.md    # Technical Documentation
├── create_sample_pdf.py    # Generate sample support PDF
├── requirements.txt        # Python dependencies
├── .env.example            # Environment variables template
└── README.md               # This file
```

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
pip install reportlab   # Only needed to generate the sample PDF
```

### 2. Configure Environment

```bash
cp .env.example .env
# Edit .env with your API keys
```

### 3. Generate Sample PDF (Optional)

```bash
python create_sample_pdf.py
```

### 4. Ingest Documents

```bash
# Single PDF
python -m src.app --ingest data/sample_support_docs.pdf

# Directory of PDFs
python -m src.app --ingest data/
```

### 5. Run Queries

```bash
# Interactive mode
python -m src.app

# Demo mode (pre-defined queries)
python -m src.app --demo
```

## 🔧 Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `LLM_PROVIDER` | `openai` | LLM backend: `openai` or `gemini` |
| `OPENAI_API_KEY` | — | OpenAI API key |
| `GOOGLE_API_KEY` | — | Google Gemini API key |
| `CHROMA_PERSIST_DIR` | `./chroma_db` | ChromaDB storage path |
| `RETRIEVAL_TOP_K` | `3` | Number of chunks to retrieve |
| `CONFIDENCE_THRESHOLD` | `0.4` | Below this → HITL escalation |

## 📐 Tech Stack

| Component | Technology | Rationale |
|-----------|-----------|-----------|
| Embeddings | all-MiniLM-L6-v2 | 384-dim, fast, runs locally |
| Vector Store | ChromaDB | Local persistence, metadata filtering |
| LLM | GPT-3.5-turbo / Gemini | Cost-effective for support use case |
| Orchestration | LangGraph | Conditional routing, typed state |
| Document Loading | PyPDFLoader | Robust PDF parsing |

## 📝 License

MIT — built as part of Innomatics Research Labs internship project.
