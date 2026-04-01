"""
Test Hallucination Detector with LLM Judge and Ground Truth
"""

from embeddings.embedder import EmbeddingEngine
from llm.judge import LLMJudge
from modules.hallucination import HallucinationDetector

print("Loading embedder...")
embedder = EmbeddingEngine()

print("\nLoading LLM Judge...")
llm_judge = LLMJudge()

print("\nInitializing Hallucination Detector with LLM Judge...")
detector = HallucinationDetector(embedder, llm_judge)

print("\n" + "="*60)
print("HALLUCINATION DETECTOR TEST (with LLM Judge & Ground Truth)")
print("="*60)

# Test cases with ground truth
test_cases = [
    {
        "answer": "The Eiffel Tower is in Paris.",
        "context": "The Eiffel Tower is located in Paris, France.",
        "ground_truth": "Paris, France",
        "description": "Correct answer"
    },
    {
        "answer": "The Eiffel Tower is in London.",
        "context": "The Eiffel Tower is located in Paris, France.",
        "ground_truth": "Paris, France",
        "description": "Wrong answer (hallucination)"
    },
    {
        "answer": "The Great Wall of China is visible from space.",
        "context": "The Great Wall of China is a series of fortifications. It is not visible from space with the naked eye.",
        "ground_truth": "The Great Wall of China is not visible from space",
        "description": "Common myth (hallucination)"
    }
]

for test in test_cases:
    print(f"\n📝 Test: {test['description']}")
    print(f"   Context: {test['context'][:60]}...")
    print(f"   Answer: {test['answer']}")
    print(f"   Ground Truth: {test['ground_truth'][:50]}...")
    
    result = detector.detect(
        test['answer'], 
        test['context'], 
        ground_truth=test['ground_truth']
    )
    
    print(f"\n   Hallucination Score: {result['hallucination_score']:.3f}")
    print(f"   Verdict: {result['verdict']}")
    print(f"   Methods Used: {list(result['methods'].keys())}")
    
    if 'llm_judge' in result['methods']:
        llm_result = result['methods']['llm_judge']
        print(f"   LLM Judge: {llm_result.get('is_supported', 'N/A')}")
    
    if 'ground_truth_similarity' in result['methods']:
        gt_sim = result['methods']['ground_truth_similarity']
        print(f"   Ground Truth Similarity: {gt_sim:.3f}")