"""
Quick test for Faithfulness Module
"""

from embeddings.embedder import EmbeddingEngine
from modules.faithfulness import FaithfulnessModule

# Create embedder (this will download a small model first time)
print("Loading embedder...")
embedder = EmbeddingEngine()

# Create faithfulness module
faithfulness = FaithfulnessModule(embedder)

# Test cases
test_cases = [
    {
        "context": "The Eiffel Tower is located in Paris, France. It was completed in 1889.",
        "answer": "The Eiffel Tower is in Paris.",
        "description": "Correct, supported answer"
    },
    {
        "context": "The Eiffel Tower is located in Paris, France. It was completed in 1889.",
        "answer": "The Eiffel Tower is in London.",
        "description": "Contradictory answer"
    },
    {
        "context": "The Eiffel Tower is located in Paris, France. It was completed in 1889.",
        "answer": "It is a famous landmark in the city of love.",
        "description": "Paraphrased answer"
    }
]

print("\n" + "="*60)
print("FAITHFULNESS MODULE TEST")
print("="*60)

for test in test_cases:
    print(f"\n📝 Test: {test['description']}")
    print(f"   Context: {test['context'][:60]}...")
    print(f"   Answer: {test['answer']}")
    
    result = faithfulness.evaluate(test['answer'], test['context'])
    
    print(f"   Score: {result['score']:.3f}")
    print(f"   Level: {result['level']}")
    print(f"   Explanation: {result['explanation']}")