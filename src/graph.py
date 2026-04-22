"""
graph.py — LangGraph Workflow Orchestration
=============================================
Defines the StateGraph that orchestrates the RAG pipeline:

    retrieve_node  →  generate_node  →  [conditional routing]
                                              ├── hitl_node → END
                                              └── END

State is passed between nodes via SupportState (a TypedDict).
Each node reads from and writes to specific fields, maintaining
a clean contract that makes the graph testable and debuggable.

Design decision: LangGraph over LangChain's sequential chains because:
  - Conditional branching (HITL routing) is a first-class concept
  - State is explicit and typed, not hidden in chain internals
  - Graph structure is inspectable and visualizable
  - Adding new nodes (e.g., feedback, logging) requires zero refactoring
"""

import os
from typing import TypedDict, List, Optional
from dotenv import load_dotenv

from langgraph.graph import StateGraph, END

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from src.vector_store import load_vector_store, retrieve_relevant_chunks
from src.hitl import should_escalate, get_human_response

load_dotenv()


# ============================================================
# STATE SCHEMA
# ============================================================
class SupportState(TypedDict):
    """
    Typed state schema for the customer support graph.

    Fields:
        query:            The customer's natural language question.
        retrieved_chunks: Text content of top-k retrieved chunks.
        confidence:       Average similarity score from retrieval (0-1).
        response:         LLM-generated answer (or human response if escalated).
        escalate:         Whether the query was escalated to a human.
        human_response:   The human operator's response (if escalated).
    """
    query: str
    retrieved_chunks: List[str]
    confidence: float
    response: str
    escalate: bool
    human_response: Optional[str]


# ============================================================
# LLM INITIALIZATION
# ============================================================
def get_llm():
    """
    Initialize the LLM based on the configured provider.

    Supports two providers via LLM_PROVIDER env var:
      - "openai" → GPT-3.5-turbo (default, best cost/quality for support)
      - "gemini" → Google Gemini via langchain-google-genai

    Returns:
        BaseChatModel: Configured LLM instance.
    """
    provider = os.getenv("LLM_PROVIDER", "openai").lower()

    if provider == "gemini":
        # Import only when needed — keeps openai as the default path
        from langchain_google_genai import ChatGoogleGenerativeAI
        print("🤖 LLM Provider: Google Gemini")
        return ChatGoogleGenerativeAI(
            model="gemini-2.0-flash",
            google_api_key=os.getenv("GOOGLE_API_KEY"),
            temperature=0.3   # Low temperature for factual support responses
        )
    else:
        print("🤖 LLM Provider: OpenAI GPT-3.5-turbo")
        return ChatOpenAI(
            model="gpt-3.5-turbo",
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            temperature=0.3   # Low temperature for factual support responses
        )


# ============================================================
# NODE DEFINITIONS
# ============================================================

def retrieve_node(state: SupportState) -> dict:
    """
    NODE 1: Retrieve relevant document chunks from ChromaDB.

    Reads:  state["query"]
    Writes: state["retrieved_chunks"], state["confidence"]

    Performs similarity search against the vector store and computes
    an average confidence score across the top-k results. This score
    drives the downstream escalation decision.
    """
    print("\n" + "━" * 50)
    print("🔄 NODE: retrieve_node")
    print("━" * 50)

    query = state["query"]
    vector_store = load_vector_store()
    documents, scores = retrieve_relevant_chunks(vector_store, query)

    # Extract text content from Document objects
    chunk_texts = [doc.page_content for doc in documents]

    # Average similarity score as our confidence metric
    # If no results, confidence is 0.0 (guaranteed escalation)
    confidence = sum(scores) / len(scores) if scores else 0.0

    print(f"\n   📊 Average confidence: {confidence:.3f}")

    return {
        "retrieved_chunks": chunk_texts,
        "confidence": confidence
    }


def generate_node(state: SupportState) -> dict:
    """
    NODE 2: Generate a response using the LLM with retrieved context.

    Reads:  state["query"], state["retrieved_chunks"]
    Writes: state["response"]

    Constructs a prompt that includes the retrieved chunks as context
    and instructs the LLM to:
      - Answer based ONLY on the provided context
      - Say "I don't know" if the context is insufficient
      - Include "ESCALATE" if the query requires human expertise

    The system prompt is carefully designed to prevent hallucination
    and to trigger HITL escalation when appropriate.
    """
    print("\n" + "━" * 50)
    print("🔄 NODE: generate_node")
    print("━" * 50)

    query = state["query"]
    chunks = state["retrieved_chunks"]

    # Format retrieved chunks into a numbered context block
    context = "\n\n".join(
        [f"[Chunk {i+1}]: {chunk}" for i, chunk in enumerate(chunks)]
    )

    # System prompt engineered for factual, context-grounded responses
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a professional customer support assistant.
Answer the customer's question using ONLY the context provided below.

RULES:
1. If the context contains the answer, respond clearly and helpfully.
2. If the context does NOT contain enough information, respond with:
   "I don't know based on the available information. ESCALATE"
3. If the question involves billing disputes, account security,
   or legal matters, respond with: "This requires human review. ESCALATE"
4. Keep responses concise (2-4 sentences) and professional.
5. Do NOT make up information that is not in the context.

CONTEXT:
{context}"""),
        ("human", "{query}")
    ])

    llm = get_llm()

    # Chain: prompt → LLM → string parser
    chain = prompt | llm | StrOutputParser()

    print(f"   📝 Generating response with {len(chunks)} context chunks...")
    response = chain.invoke({"context": context, "query": query})

    print(f"   🤖 LLM Response: {response[:150]}{'...' if len(response) > 150 else ''}")

    return {"response": response}


def hitl_node(state: SupportState) -> dict:
    """
    NODE 3: Human-in-the-Loop escalation handler.

    Reads:  state["query"], state["response"], state["confidence"]
    Writes: state["escalate"], state["human_response"], state["response"]

    This node is only reached via conditional routing when the system
    determines it cannot confidently answer the query. It presents
    the full context to a human operator and captures their response.
    """
    print("\n" + "━" * 50)
    print("🔄 NODE: hitl_node")
    print("━" * 50)

    query = state["query"]
    llm_response = state["response"]
    confidence = state["confidence"]

    # Delegate to the HITL module for human interaction
    human_response = get_human_response(query, llm_response, confidence)

    return {
        "escalate": True,
        "human_response": human_response,
        # Override the LLM response with the human's authoritative answer
        "response": f"[Human Agent] {human_response}"
    }


# ============================================================
# CONDITIONAL ROUTING
# ============================================================

def route_after_generate(state: SupportState) -> str:
    """
    Conditional edge function: decides whether to escalate or finish.

    Routing logic:
      - If should_escalate() returns True → route to "hitl_node"
      - Otherwise → route to END

    This function is called by LangGraph after generate_node completes.
    The return value must match a node name or the END sentinel.
    """
    print("\n   🔀 Routing decision...")
    confidence = state["confidence"]
    response = state["response"]

    if should_escalate(confidence, response):
        print("   → Routing to: hitl_node (escalation)")
        return "hitl_node"
    else:
        print("   → Routing to: END (confident response)")
        return "end"


# ============================================================
# GRAPH CONSTRUCTION
# ============================================================

def build_graph():
    """
    Construct and compile the LangGraph StateGraph.

    Graph topology:
        START → retrieve_node → generate_node → [conditional]
                                                    ├── hitl_node → END
                                                    └── END

    Returns:
        CompiledGraph: Ready-to-invoke graph instance.
    """
    print("\n🏗️  Building LangGraph workflow...")

    # Initialize the graph with our typed state schema
    graph = StateGraph(SupportState)

    # --- Add nodes ---
    graph.add_node("retrieve_node", retrieve_node)
    graph.add_node("generate_node", generate_node)
    graph.add_node("hitl_node", hitl_node)

    # --- Define edges ---
    # Entry point: always start with retrieval
    graph.set_entry_point("retrieve_node")

    # retrieve_node → generate_node (always)
    graph.add_edge("retrieve_node", "generate_node")

    # generate_node → conditional routing
    graph.add_conditional_edges(
        "generate_node",
        route_after_generate,
        {
            "hitl_node": "hitl_node",   # Low confidence → human
            "end": END                   # High confidence → done
        }
    )

    # hitl_node → END (after human responds, we're done)
    graph.add_edge("hitl_node", END)

    # Compile the graph into an executable
    compiled = graph.compile()
    print("   ✅ Graph compiled successfully")

    return compiled


def run_query(query: str) -> dict:
    """
    Execute a complete query through the RAG pipeline.

    This is the main entry point for processing a customer query.
    It builds the graph, initializes the state, and invokes the
    full workflow.

    Args:
        query: The customer's natural language question.

    Returns:
        dict: Final SupportState containing the response and metadata.
    """
    graph = build_graph()

    # Initialize state with defaults for all fields
    initial_state: SupportState = {
        "query": query,
        "retrieved_chunks": [],
        "confidence": 0.0,
        "response": "",
        "escalate": False,
        "human_response": None
    }

    print(f"\n{'═' * 60}")
    print(f"🚀 Processing query: \"{query}\"")
    print(f"{'═' * 60}")

    # Invoke the graph — LangGraph handles node sequencing and routing
    final_state = graph.invoke(initial_state)

    print(f"\n{'═' * 60}")
    print(f"✅ FINAL RESULT")
    print(f"{'═' * 60}")
    print(f"   📝 Response:   {final_state['response']}")
    print(f"   📊 Confidence: {final_state['confidence']:.3f}")
    print(f"   🧑‍💼 Escalated:  {final_state.get('escalate', False)}")
    print(f"{'═' * 60}")

    return final_state
