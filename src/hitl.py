"""
hitl.py — Human-in-the-Loop (HITL) Escalation Module
======================================================
Implements the human escalation interface for queries that the LLM
cannot confidently answer. In this CLI implementation, the human
operator is prompted via stdin. In production, this would be replaced
by a webhook/queue system (e.g., Slack integration, ticketing API).

HITL triggers when:
  1. Average retrieval similarity score < CONFIDENCE_THRESHOLD (0.4)
  2. LLM response contains the keyword "escalate"
  3. LLM explicitly indicates uncertainty in its output

This ensures the system degrades gracefully — instead of hallucinating
an answer, it routes to a human who can provide an authoritative response.
"""

import os
from dotenv import load_dotenv

load_dotenv()

CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", "0.4"))


def should_escalate(confidence: float, response: str) -> bool:
    """
    Determine whether a query should be escalated to a human operator.

    Decision logic (OR-based — any condition triggers escalation):
      1. confidence < CONFIDENCE_THRESHOLD → retrieval quality is too low
         for the LLM to produce a reliable answer
      2. "escalate" in response → the LLM itself signals it cannot answer
      3. "i don't know" / "not sure" in response → uncertainty indicators

    Args:
        confidence: Float in [0, 1] representing average retrieval score.
        response: The LLM-generated response text.

    Returns:
        bool: True if the query should be escalated.
    """
    response_lower = response.lower()

    # Condition 1: Low retrieval confidence
    if confidence < CONFIDENCE_THRESHOLD:
        print(f"   ⚠️  Low confidence ({confidence:.3f} < {CONFIDENCE_THRESHOLD}) → Escalating")
        return True

    # Condition 2: LLM explicitly requests escalation
    if "escalate" in response_lower:
        print(f"   ⚠️  LLM flagged 'escalate' in response → Escalating")
        return True

    # Condition 3: LLM expresses uncertainty
    uncertainty_phrases = ["i don't know", "not sure", "cannot answer", "no relevant information"]
    for phrase in uncertainty_phrases:
        if phrase in response_lower:
            print(f"   ⚠️  LLM expressed uncertainty ('{phrase}') → Escalating")
            return True

    return False


def get_human_response(query: str, llm_response: str, confidence: float) -> str:
    """
    Present the escalated query to a human operator and collect their response.

    Displays the original query, the LLM's attempted response, and the
    confidence score to give the human operator full context before they
    craft their answer.

    In production, this function would:
    - POST to a webhook (Slack, Teams, PagerDuty)
    - Enqueue to a task queue (Celery, SQS) with a ticket ID
    - Block on an async callback or poll for resolution

    Args:
        query: The user's original question.
        llm_response: The LLM's generated (low-confidence) response.
        confidence: The retrieval confidence score.

    Returns:
        str: The human operator's response text.
    """
    print("\n" + "=" * 60)
    print("🧑‍💼 HUMAN-IN-THE-LOOP ESCALATION")
    print("=" * 60)
    print(f"\n📋 Customer Query:  {query}")
    print(f"🤖 LLM Attempted:  {llm_response[:200]}{'...' if len(llm_response) > 200 else ''}")
    print(f"📊 Confidence:      {confidence:.3f}")
    print(f"\n{'─' * 60}")
    print("The system could not confidently answer this query.")
    print("Please provide an authoritative response below.")
    print(f"{'─' * 60}")

    # Collect human input — this blocks until the operator responds
    human_response = input("\n✍️  Your response: ").strip()

    # Handle empty responses gracefully
    if not human_response:
        human_response = "A human agent has reviewed this query and will follow up shortly."
        print(f"   ℹ️  Empty input — using default response")

    print(f"\n   ✅ Human response recorded")
    print("=" * 60)

    return human_response
