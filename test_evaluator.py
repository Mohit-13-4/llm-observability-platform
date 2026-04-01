"""
Test the complete LLM Evaluator pipeline.
"""

from core.evaluator import LLMEvaluator

print("="*60)
print("TESTING LLM EVALUATOR")
print("="*60)

# Initialize evaluator
print("\nInitializing evaluator...")
evaluator = LLMEvaluator()

# Test cases
test_cases = [
    {
        "prompt": "Where is the Eiffel Tower?",
        "output": "The Eiffel Tower is in Paris.",
        "context": "The Eiffel Tower is located in Paris, France.",
        "ground_truth": "Paris, France",
        "description": "Correct answer"
    },
    {
        "prompt": "Where is the Eiffel Tower?",
        "output": "The Eiffel Tower is in London.",
        "context": "The Eiffel Tower is located in Paris, France.",
        "ground_truth": "Paris, France",
        "description": "Wrong answer (hallucination)"
    },
    {
        "prompt": "What is the capital of France?",
        "output": "The capital of France is Paris, the city of love and lights.",
        "context": None,
        "ground_truth": "Paris",
        "description": "Paraphrased answer"
    }
]

print("\n" + "-"*60)
for test in test_cases:
    print(f"\n📝 Test: {test['description']}")
    print(f"   Prompt: {test['prompt']}")
    print(f"   Output: {test['output']}")
    
    result = evaluator.evaluate_single(
        prompt=test['prompt'],
        model_output=test['output'],
        context=test.get('context'),
        ground_truth=test.get('ground_truth')
    )
    
    print(f"\n   📊 Results:")
    print(f"      Faithfulness: {result.faithfulness_score:.3f}")
    print(f"      Hallucination: {result.hallucination_score:.3f}")
    if result.similarity_score:
        print(f"      Similarity: {result.similarity_score:.3f}")
    print(f"      Overall Score: {result.overall_score:.3f}")
    print(f"      Verdict: ", end="")
    
    if result.overall_score >= 0.8:
        print("✅ EXCELLENT")
    elif result.overall_score >= 0.6:
        print("⚠️ GOOD")
    else:
        print("❌ POOR - Hallucination likely")