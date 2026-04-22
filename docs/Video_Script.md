# Video Presentation Script — RAG Customer Support Assistant
## Duration: 7–9 minutes | Teleprompter Format

---

### [00:00 – 00:45] SEGMENT 1: Introduction

Hi, I'm [Your Name], from the [Batch Name] batch at Innomatics Research Labs.

[PAUSE — look at camera]

Today I'm going to walk you through a system I built from scratch — a RAG-based Customer Support Assistant that uses LangGraph for workflow orchestration and includes a Human-in-the-Loop escalation path.

[SLOW DOWN]

The problem this solves is straightforward: traditional support chatbots are brittle. They match keywords to scripted responses, and the moment a customer asks something slightly outside the script, they fail. No fallback, no awareness of their own limitations — just a generic "I don't understand."

What I've built is fundamentally different. It retrieves real information from support documents, reasons over that information using an LLM, and — this is the critical part — knows when it *can't* answer, and routes those queries to a human operator.

Let me show you how it works.

---

### [00:45 – 02:45] SEGMENT 2: System Overview

[SHOW SCREEN — display the architecture diagram from the HLD]

Here's the architecture. There are two pipelines — let me walk through each one.

[PAUSE — point to the ingestion pipeline]

First, the **ingestion pipeline** — this runs offline. We take PDF documents — in our case, a customer support FAQ — and load them using PyPDFLoader, which extracts text page by page.

Those pages are then split into smaller chunks using a Recursive Character Text Splitter. I'm using 500-character chunks with a 50-character overlap. The 500 character size is deliberate — it's large enough to capture a complete FAQ answer, but small enough that retrieval stays precise. The overlap prevents us from cutting a sentence in half at chunk boundaries.

[SLOW DOWN]

Each chunk is then embedded — converted into a 384-dimensional vector — using the all-MiniLM-L6-v2 model from HuggingFace. This runs entirely locally, no API calls, no data leaving the machine.

These embeddings are stored in ChromaDB, which gives us persistent vector storage with built-in similarity search.

[PAUSE — point to the query pipeline]

Now, the **query pipeline** — this is the real-time path. When a customer submits a question, it enters a LangGraph StateGraph — a graph-based workflow engine that I chose specifically because this system needs conditional branching.

[Point to each node as you mention it]

Node one: **retrieve_node** — embeds the query using the same MiniLM model and searches ChromaDB for the top 3 most relevant chunks. It also computes a confidence score — the average similarity across those 3 results.

This feeds into node two: **generate_node** — which takes the query and the retrieved chunks, constructs a grounded prompt, and sends it to the LLM — GPT-3.5-turbo in my case, though the system also supports Google Gemini as an alternative.

After generation, we hit the **conditional router**. This is where it gets interesting.

[SLOW DOWN]

The router checks two things: Is the confidence score below 0.4? And did the LLM signal uncertainty in its response? If either condition is true, the query routes to node three — the **HITL node**, where a human operator provides the answer. If both conditions are clear, the response goes directly to the customer.

---

### [02:45 – 04:15] SEGMENT 3: End-to-End Workflow

Let me trace through two scenarios end to end.

[SHOW SCREEN — terminal or code]

**Scenario one — normal path.** A customer asks: "What is your return policy?"

The query is embedded and sent to ChromaDB. It finds three relevant chunks from the FAQ — all about the return policy. The confidence score comes back at, say, 0.72 — well above our 0.4 threshold.

These chunks are formatted into a prompt and sent to GPT-3.5-turbo. The LLM reads the context and generates a clear, concise answer: "Our return policy allows returns within 30 days, items must be in original packaging, refunds are processed in 5-7 business days."

The router checks — confidence 0.72, no "escalate" keyword in the response — and routes directly to END. The customer gets their answer.

[PAUSE]

**Scenario two — HITL path.** A customer asks: "I need help with a billing dispute for order number 98765."

The query is embedded and searched against the FAQ. It finds some loosely related chunks about billing, but the average confidence is only 0.31 — below our threshold.

The LLM receives the context, recognizes it can't resolve a specific order dispute from general FAQ content, and includes "ESCALATE" in its response — exactly as the system prompt instructed it to.

The router detects both signals — low confidence AND the escalate keyword — and routes to the HITL node. A human operator sees the original question, the LLM's attempted response, and the confidence score. They type an authoritative answer, which replaces the LLM's response and is returned to the customer with a "[Human Agent]" prefix.

That's the complete lifecycle.

---

### [04:15 – 06:30] SEGMENT 4: Live Demo

Let me show you this running.

[ACTION: Open terminal in the project directory]

First, let me ingest the sample support PDF.

[ACTION: Type `python -m src.app --ingest data/sample_support_docs.pdf`]

[SHOW SCREEN]

You can see the pipeline running — it loaded the PDF, extracted the pages, chunked them into approximately 40-50 smaller fragments, embedded each one, and stored everything in ChromaDB.

[PAUSE — wait for ingestion to complete]

Now let's run some queries.

[ACTION: Type `python -m src.app`]

I'm in interactive mode. Let me ask a normal question first.

[ACTION: Type `What is your return policy?`]

[SHOW SCREEN — point to output]

Look at the output. You can see the three chunks that were retrieved, each with its similarity score. The average confidence is above 0.4. The LLM generated a response using those chunks. And the router sent it straight to END. No escalation.

[PAUSE]

Now let me ask something that should trigger HITL.

[ACTION: Type `I need help with a billing dispute for order #98765`]

[SHOW SCREEN — point to output]

Watch what happens. The retrieval finds some billing-related chunks, but the confidence is low. The LLM includes "ESCALATE" in its response. The router catches both signals and drops us into the HITL node.

[SHOW SCREEN — point to the HITL prompt]

See — it shows me the original query, the LLM's attempted answer, and the confidence score. As the human operator, I have full context to craft a response.

[ACTION: Type a response like "I've escalated your billing dispute to our billing team. Case #12345 has been created. You'll hear back within 24 hours."]

[SHOW SCREEN]

And there it is — the human response replaces the LLM's answer and is delivered to the customer.

[PAUSE]

Let me also try a completely off-topic question to show the confidence mechanism.

[ACTION: Type `What is quantum computing?`]

[SHOW SCREEN]

The retrieved chunks are irrelevant — the confidence score drops below 0.4. Even without the LLM explicitly saying "escalate", the low confidence alone triggers the HITL path. That's the multi-signal routing I mentioned earlier.

---

### [06:30 – 07:30] SEGMENT 5: Technical Decisions

Let me quickly cover the key technical decisions behind this system.

[SLOW DOWN]

**Chunk size of 500** — this matches the typical length of a single FAQ answer. Smaller chunks would split answers mid-sentence. Larger chunks would dilute retrieval precision.

**All-MiniLM-L6-v2** for embeddings — it runs locally, produces compact 384-dimensional vectors, and ranks in the top tier on the MTEB benchmark for models its size. No API costs, no data leaving the machine.

**ChromaDB** over alternatives like FAISS or Pinecone — FAISS doesn't persist to disk natively and has no metadata filtering. Pinecone is cloud-only and costs money. ChromaDB gives us local persistence with a clean Python API.

**LangGraph** instead of LangChain chains — because this workflow branches. The HITL node is conditional. LangChain chains are linear by design; bolting conditional logic onto them obscures the control flow. LangGraph makes branching a first-class concept.

And the **confidence threshold of 0.4** — calibrated empirically. In-domain queries consistently score above 0.5. Out-of-domain queries score below 0.4. The threshold cleanly separates these populations.

---

### [07:30 – 08:15] SEGMENT 6: Challenges & Learnings

[PAUSE — shift to a reflective tone]

The biggest technical challenge I ran into was with ChromaDB's relevance score normalization. The `similarity_search_with_relevance_scores` method returns values between 0 and 1, but the actual scale depends on whether the underlying distance metric is L2, cosine, or inner product — and this behavior changed across ChromaDB versions.

My initial confidence thresholds were completely wrong because the scores weren't on the scale I expected. The fix was to explicitly L2-normalize all embeddings in the embedding model configuration — setting `normalize_embeddings=True` — which guarantees cosine similarity semantics regardless of ChromaDB's internal implementation.

[SLOW DOWN]

The key insight from building this system: the hardest part of RAG is not retrieval or generation — it's knowing when the system should *not* answer. Building the confidence estimation and escalation routing was more challenging and more important than any other component.

---

### [08:15 – 09:00] SEGMENT 7: Conclusion

[PAUSE — look at camera]

To summarize: I built a Retrieval-Augmented Generation system that ingests PDF documents, retrieves relevant context using semantic search, generates grounded responses using an LLM, and — crucially — escalates to a human operator when it can't confidently answer.

[SLOW DOWN]

Going forward, there are two enhancements I'd prioritize. First, **multi-turn conversation memory** using LangGraph's built-in state persistence — so customers can ask follow-up questions without repeating context. Second, **quantitative evaluation** using the RAGAS framework to measure faithfulness, relevance, and retrieval quality with actual metrics instead of manual inspection.

[PAUSE]

This project demonstrates that production-grade AI systems are not just about the LLM — they're about the engineering around the LLM: how you structure knowledge, how you route decisions, and how you handle the cases where AI falls short.

Thank you.

[END]

---

*Script length: ~2,100 words | Estimated delivery time: 8-9 minutes at presentation pace*
