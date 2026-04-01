"""
Test TruthfulQA Dataset Loader
"""

from data_loaders.truthfulqa import TruthfulQALoader

print("="*60)
print("TESTING TRUTHFULQA DATASET LOADER")
print("="*60)

# Load dataset
loader = TruthfulQALoader()
data = loader.load()

print(f"\n📊 Dataset Statistics:")
print(f"   Total questions: {len(data['questions'])}")
print(f"   Categories: {set(data['categories'])}")

# Get sample
print("\n📝 Sample Questions:")
samples = loader.get_samples(3)
for i, sample in enumerate(samples):
    print(f"\n   {i+1}. Question: {sample['question'][:80]}...")
    print(f"      Correct Answer: {sample['correct_answer'][:60]}...")
    print(f"      Incorrect Answers: {sample['incorrect_answers'][:2]}")

# Create evaluation pairs
print("\n📊 Creating Evaluation Pairs...")
pairs = loader.create_evaluation_pairs(num_samples=5)
print(f"   Created {len(pairs)} evaluation pairs")
print(f"   Correct answers: {sum(1 for p in pairs if not p['is_hallucination'])}")
print(f"   Hallucinated answers: {sum(1 for p in pairs if p['is_hallucination'])}")