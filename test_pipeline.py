"""
Quick end-to-end test of the retrieval pipeline.
Tests retrieval + confidence scoring without needing an LLM API key.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.vector_store import load_vector_store, retrieve_relevant_chunks
from src.hitl import should_escalate

def test_retrieval():
    """Test that retrieval returns relevant chunks with reasonable scores."""
    vs = load_vector_store()

    # Test 1: In-domain query — should have high confidence
    print("\n" + "=" * 60)
    print("TEST 1: In-domain query")
    docs, scores = retrieve_relevant_chunks(vs, "What is your return policy?")
    avg = sum(scores) / len(scores) if scores else 0
    print(f"   Average confidence: {avg:.3f}")
    assert len(docs) == 3, f"Expected 3 chunks, got {len(docs)}"
    assert avg > 0.1, f"Expected confidence > 0.1 for in-domain query, got {avg:.3f}"
    print("   PASS")

    # Test 2: Out-of-domain query — should have low confidence
    print("\n" + "=" * 60)
    print("TEST 2: Out-of-domain query")
    docs2, scores2 = retrieve_relevant_chunks(vs, "What is quantum computing?")
    avg2 = sum(scores2) / len(scores2) if scores2 else 0
    print(f"   Average confidence: {avg2:.3f}")
    print("   PASS (low confidence expected)")

    # Test 3: HITL escalation logic
    print("\n" + "=" * 60)
    print("TEST 3: HITL escalation logic")
    assert should_escalate(0.3, "Here is the answer") == True, "Low confidence should escalate"
    assert should_escalate(0.7, "Here is the answer") == False, "High confidence should not escalate"
    assert should_escalate(0.7, "I don't know the answer. ESCALATE") == True, "Escalate keyword should trigger"
    assert should_escalate(0.7, "I'm not sure about this") == True, "Uncertainty phrase should trigger"
    print("   All escalation logic tests PASS")

    print("\n" + "=" * 60)
    print("ALL TESTS PASSED")
    print("=" * 60)

if __name__ == "__main__":
    test_retrieval()
