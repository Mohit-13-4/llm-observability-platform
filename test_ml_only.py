# test_ml_only.py
from embeddings.embedder import EmbeddingEngine
from llm.judge import LLMJudge
from modules.hallucination import HallucinationDetector

print("="*60)
print("TESTING ML-ONLY HALLUCINATION DETECTION")
print("="*60)

embedder = EmbeddingEngine()
llm_judge = LLMJudge()
detector = HallucinationDetector(embedder, llm_judge)

# Test with a new city the model has never seen in hardcoded rules
test_cases = [
    {
        "name": "Correct capital",
        "answer": "New Delhi is the capital of India",
        "context": "India is a country in South Asia. Its capital is New Delhi."
    },
    {
        "name": "Wrong capital (Mumbai)",
        "answer": "Mumbai is the capital of India",
        "context": "India is a country in South Asia. Its capital is New Delhi."
    },
    {
        "name": "Wrong capital (Chennai - new city!)",
        "answer": "Chennai is the capital of India",
        "context": "India is a country in South Asia. Its capital is New Delhi."
    }
]

for test in test_cases:
    print(f"\n📝 {test['name']}")
    result = detector.detect(
        answer=test['answer'],
        context=test['context']
    )
    
    print(f"   Hallucination Score: {result['hallucination_score']:.3f}")
    print(f"   Verdict: {result['verdict']}")
    
    # Show method contributions
    if 'llm_judge' in result['methods']:
        llm = result['methods']['llm_judge']
        print(f"   LLM Judge: {'❌ Not Supported' if not llm.get('is_supported') else '✅ Supported'}")