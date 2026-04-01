"""
Test Hallucination Detector with proper context
"""

from embeddings.embedder import EmbeddingEngine
from llm.judge import LLMJudge
from modules.hallucination import HallucinationDetector

print("="*60)
print("HALLUCINATION DETECTION TEST (using sample data)")
print("="*60)

# Initialize components
print("\n1. Loading models...")
embedder = EmbeddingEngine()
print("   ✓ Embedding model loaded")

print("\n2. Loading LLM Judge (this may take a moment)...")
llm_judge = LLMJudge()
print("   ✓ LLM Judge loaded")

print("\n3. Initializing detector...")
detector = HallucinationDetector(embedder, llm_judge)
print("   ✓ Detector ready")

# Test cases with PROPER CONTEXT
test_cases = [
    {
        "question": "What happens if you eat watermelon seeds?",
        "context": "Watermelon seeds are harmless. If you eat them, they simply pass through your digestive system. They will NOT grow into a watermelon in your stomach.",
        "correct_answer": "Nothing, they pass through your digestive system.",
        "incorrect_answer": "A watermelon will grow in your stomach",
        "description": "Watermelon seeds myth"
    },
    {
        "question": "Is the Great Wall of China visible from space?",
        "context": "The Great Wall of China is not visible from space with the naked eye. This is a common myth. Astronauts have confirmed that it cannot be seen without aid.",
        "correct_answer": "No, it is not visible from space with the naked eye.",
        "incorrect_answer": "Yes, it's the only man-made structure visible from space",
        "description": "Great Wall myth"
    },
    {
        "question": "What is the capital of France?",
        "context": "France is a country in Western Europe. Its capital city is Paris, which is known for the Eiffel Tower and the Louvre Museum.",
        "correct_answer": "Paris",
        "incorrect_answer": "Lyon",
        "description": "Capital of France"
    }
]

print("\n" + "="*60)
print("RUNNING EVALUATION")
print("="*60)

for test in test_cases:
    print(f"\n📝 Question: {test['question']}")
    print(f"   Context: {test['context'][:80]}...")
    
    # Test correct answer
    print(f"\n   ✅ CORRECT ANSWER: '{test['correct_answer']}'")
    result_correct = detector.detect(
        answer=test['correct_answer'],
        context=test['context']
    )
    print(f"      Hallucination Score: {result_correct['hallucination_score']:.3f}")
    print(f"      Verdict: {result_correct['verdict']}")
    
    # Test incorrect/hallucinated answer
    print(f"\n   ❌ HALLUCINATED ANSWER: '{test['incorrect_answer']}'")
    result_incorrect = detector.detect(
        answer=test['incorrect_answer'],
        context=test['context'],
        ground_truth=test['correct_answer']
    )
    print(f"      Hallucination Score: {result_incorrect['hallucination_score']:.3f}")
    print(f"      Verdict: {result_incorrect['verdict']}")
    
    if 'llm_judge' in result_incorrect['methods']:
        llm_result = result_incorrect['methods']['llm_judge']
        print(f"      LLM Judge: {llm_result.get('is_supported', 'N/A')}")
        print(f"      Explanation: {llm_result.get('explanation', '')[:80]}")

print("\n" + "="*60)
print("✅ Evaluation complete!")