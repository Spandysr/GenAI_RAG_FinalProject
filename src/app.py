"""
app.py — RAG Customer Support Assistant CLI Application
========================================================
Main entry point that provides a demo-friendly interactive loop.

Usage:
    python -m src.app                  # Interactive query mode (vector store must exist)
    python -m src.app --ingest <path>  # Ingest a PDF or directory of PDFs first
    python -m src.app --demo           # Run pre-defined demo queries

The application assumes ChromaDB has been populated via --ingest before
running queries. This separation of concerns mirrors production systems
where ingestion and serving are independent pipelines.
"""

import sys
import os
import argparse

# Add project root to path for clean imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.ingestion import ingest_pdf, ingest_directory
from src.vector_store import create_vector_store
from src.graph import run_query


def print_banner():
    """Display the application banner."""
    print("""
╔══════════════════════════════════════════════════════════╗
║                                                          ║
║   🤖 RAG Customer Support Assistant                      ║
║   ─────────────────────────────────                      ║
║   Powered by LangGraph + ChromaDB + LLM                  ║
║   Human-in-the-Loop Escalation Enabled                   ║
║                                                          ║
╚══════════════════════════════════════════════════════════╝
    """)


def handle_ingestion(path: str):
    """
    Ingest PDF document(s) and store embeddings in ChromaDB.

    Args:
        path: Path to a single PDF file or a directory of PDFs.
    """
    print("\n" + "=" * 60)
    print("📥 DOCUMENT INGESTION PIPELINE")
    print("=" * 60)

    if os.path.isfile(path) and path.lower().endswith(".pdf"):
        # Single PDF file
        chunks = ingest_pdf(path)
    elif os.path.isdir(path):
        # Directory of PDFs
        chunks = ingest_directory(path)
    else:
        print(f"❌ Invalid path: {path}")
        print("   Provide a .pdf file or a directory containing PDFs.")
        sys.exit(1)

    # Store chunks in ChromaDB
    vector_store = create_vector_store(chunks)

    print(f"\n{'=' * 60}")
    print(f"✅ Ingestion complete! {len(chunks)} chunks stored in ChromaDB.")
    print(f"   You can now run queries with: python -m src.app")
    print(f"{'=' * 60}")

    return vector_store


def handle_demo():
    """
    Run a set of pre-defined demo queries to showcase the system.
    Includes normal queries and edge cases designed to trigger HITL.
    """
    print("\n" + "=" * 60)
    print("🎬 DEMO MODE — Running pre-defined queries")
    print("=" * 60)

    demo_queries = [
        # Normal queries — should be answerable from typical support docs
        "What is your return policy?",
        "How do I reset my password?",
        "What are your business hours?",

        # Edge case — likely to trigger HITL due to low confidence
        "Can you help me with a billing dispute for order #98765?",
    ]

    results = []
    for i, query in enumerate(demo_queries, 1):
        print(f"\n{'─' * 60}")
        print(f"📌 Demo Query {i}/{len(demo_queries)}")
        print(f"{'─' * 60}")

        result = run_query(query)
        results.append(result)

        # Pause between queries for readability
        if i < len(demo_queries):
            print("\n   ⏳ Moving to next query...\n")

    # Summary table
    print(f"\n{'═' * 60}")
    print("📊 DEMO SUMMARY")
    print(f"{'═' * 60}")
    print(f"{'Query':<45} {'Confidence':<12} {'Escalated'}")
    print(f"{'─' * 45} {'─' * 12} {'─' * 9}")
    for q, r in zip(demo_queries, results):
        truncated = q[:42] + "..." if len(q) > 45 else q
        print(f"{truncated:<45} {r['confidence']:<12.3f} {r.get('escalate', False)}")
    print(f"{'═' * 60}")


def handle_interactive():
    """
    Run the interactive query loop.
    The user can type queries and get responses until they type 'quit'.
    """
    print("\n💬 Interactive mode — type your questions below.")
    print("   Type 'quit' or 'exit' to stop.\n")

    while True:
        try:
            query = input("🙋 Your question: ").strip()

            if not query:
                continue
            if query.lower() in ("quit", "exit", "q"):
                print("\n👋 Goodbye!")
                break

            result = run_query(query)

        except KeyboardInterrupt:
            print("\n\n👋 Interrupted. Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")
            print("   Please try again or check your configuration.\n")


def main():
    """
    CLI entry point with argument parsing.

    Subcommands:
        --ingest <path>  : Ingest PDF(s) into ChromaDB
        --demo           : Run demo queries
        (default)        : Interactive query mode
    """
    parser = argparse.ArgumentParser(
        description="RAG Customer Support Assistant — LangGraph + ChromaDB + HITL",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m src.app --ingest data/support_docs.pdf
  python -m src.app --ingest data/
  python -m src.app --demo
  python -m src.app
        """
    )
    parser.add_argument(
        "--ingest",
        type=str,
        metavar="PATH",
        help="Path to a PDF file or directory of PDFs to ingest"
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Run pre-defined demo queries"
    )

    args = parser.parse_args()

    print_banner()

    if args.ingest:
        handle_ingestion(args.ingest)
    elif args.demo:
        handle_demo()
    else:
        handle_interactive()


if __name__ == "__main__":
    main()
