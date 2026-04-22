# LinkedIn Post

---

Most customer support bots break the moment a question falls outside their script. I spent the last few weeks building one that doesn't.

For my final project at Innomatics Research Labs, I designed and built a RAG-based Customer Support Assistant — a system that retrieves answers from real documentation instead of guessing, and knows when to step aside and let a human take over.

Here's how it works:

📄 PDF documents are loaded, chunked (500-char with overlap), and embedded using all-MiniLM-L6-v2 — a lightweight model that runs entirely locally with zero API cost.

🔍 When a customer asks a question, ChromaDB performs semantic similarity search to find the three most relevant document chunks. These chunks become the LLM's context — not the entire document, not a pre-trained memory, just the specific paragraphs that matter.

🤖 GPT-3.5-turbo generates a response grounded in that context. If the context is insufficient, the system doesn't hallucinate — it escalates.

🧑‍💼 A Human-in-the-Loop (HITL) module catches low-confidence responses and routes them to a human operator. The trigger is based on retrieval confidence scores and LLM self-assessment — not hardcoded rules.

The entire query workflow is orchestrated by LangGraph, which handles conditional routing between the AI response path and the human escalation path using a typed StateGraph.

**Tech stack:** Python, LangChain, LangGraph, ChromaDB, HuggingFace Transformers, OpenAI API

**Three things I learned building this:**

1. The hardest part of RAG isn't retrieval — it's calibrating when the system should *not* answer. Getting the confidence threshold right required empirical tuning, not just picking a number.

2. LangGraph changes how you think about LLM workflows. Once you model your pipeline as a graph with conditional edges instead of a linear chain, adding new capabilities (logging, feedback, routing) becomes trivial.

3. Embedding model choice matters more than LLM choice for retrieval quality. The LLM only sees what the retriever gives it — if retrieval is wrong, the LLM produces confident-sounding nonsense.

Full demo walkthrough: [VIDEO_LINK]

Built during my internship at @Innomatics Research Labs under the guidance of @[Mentor Name].

Open to conversations about RAG architecture, LangGraph patterns, or production AI systems — feel free to connect.

#GenAI #RAG #LangGraph #AIProjects #MachineLearning #Innomatics #LearningJourney

---

*Word count: ~320*
