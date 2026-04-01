"""
Run Hallucination Detection on TruthfulQA Dataset
"""

import time
import pandas as pd
from tqdm import tqdm
from embeddings.embedder import EmbeddingEngine
from llm.judge import LLMJudge
from modules.hallucination import HallucinationDetector
from data_loaders.truthfulqa import TruthfulQALoader

print("="*60)
print("RUNNING HALLUCINATION DETECTION ON TRUTHFULQA")
print("="*60)

# Initialize components
print("\nLoading models...")
embedder = EmbeddingEngine()
llm_judge = LLMJudge()
detector = HallucinationDetector(embedder, llm_judge)
loader = TruthfulQALoader()

# Get evaluation pairs
print("\nCreating evaluation pairs...")
pairs = loader.create_evaluation_pairs(num_samples=20)  # Start with 20 samples
print(f"Evaluating {len(pairs)} samples...")

# Run evaluation
results = []
start_time = time.time()

for pair in tqdm(pairs, desc="Evaluating"):
    result = detector.detect(
        answer=pair["answer"],
        context=pair["question"],
        ground_truth=pair["ground_truth"]
    )
    
    results.append({
        "question": pair["question"],
        "answer": pair["answer"],
        "is_actually_hallucination": pair["is_hallucination"],
        "detected_hallucination_score": result["hallucination_score"],
        "verdict": result["verdict"],
        "category": pair["category"]
    })

elapsed = time.time() - start_time

# Create DataFrame
df = pd.DataFrame(results)

# Calculate metrics
print("\n" + "="*60)
print("📊 RESULTS")
print("="*60)

print(f"\nTime taken: {elapsed:.2f} seconds")
print(f"Samples evaluated: {len(df)}")

# Accuracy calculation
df['correct_detection'] = (
    (df['detected_hallucination_score'] > 0.5) == df['is_actually_hallucination']
)
accuracy = df['correct_detection'].mean()
print(f"\nDetection Accuracy: {accuracy:.2%}")

# By category
print("\n📊 By Category:")
category_stats = df.groupby('category').agg({
    'detected_hallucination_score': 'mean',
    'is_actually_hallucination': 'mean',
    'correct_detection': 'mean'
})
print(category_stats)

# Save results
df.to_csv("results/truthfulqa_evaluation.csv", index=False)
print("\n✅ Results saved to results/truthfulqa_evaluation.csv")