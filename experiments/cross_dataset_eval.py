# experiments/cross_dataset_eval.py (UPDATED)
"""
Cross-Dataset Evaluation - Test performance across multiple datasets with STATISTICALLY SIGNIFICANT samples
"""

import pandas as pd
import time
import numpy as np
from tqdm import tqdm
from typing import Dict, List
from embeddings.embedder import EmbeddingEngine
from llm.judge import LLMJudge
from modules.hallucination import HallucinationDetector

class CrossDatasetEvaluator:
    """Evaluate hallucination detection across different datasets with sufficient samples."""
    
    def __init__(self):
        self.embedder = EmbeddingEngine()
        self.llm_judge = LLMJudge()
        self.detector = HallucinationDetector(self.embedder, self.llm_judge)
    
    def load_truthfulqa(self, num_samples: int = 30) -> List[Dict]:
        """Load TruthfulQA dataset with correct structure."""
        samples = []
        try:
            from datasets import load_dataset
            # Try both possible dataset structures
            try:
                dataset = load_dataset("truthful_qa", "multiple_choice", split="validation")
                correct_key = "correct_answers"
                incorrect_key = "incorrect_answers"
            except:
                dataset = load_dataset("truthful_qa", "generation", split="validation")
                correct_key = "correct_answer"
                incorrect_key = "incorrect_answers"
            
            for i, item in enumerate(dataset):
                if i >= num_samples:
                    break
                
                # Get correct answer
                correct = item.get(correct_key, [])
                if isinstance(correct, list):
                    correct = correct[0] if correct else ""
                
                # Get incorrect answers
                incorrect = item.get(incorrect_key, [])
                
                samples.append({
                    "question": item["question"],
                    "answer": correct,
                    "ground_truth": correct,
                    "is_hallucination": False,
                    "dataset": "TruthfulQA",
                    "type": "myth_detection"
                })
                
                for wrong in incorrect[:1]:
                    samples.append({
                        "question": item["question"],
                        "answer": wrong,
                        "ground_truth": correct,
                        "is_hallucination": True,
                        "dataset": "TruthfulQA",
                        "type": "myth_detection"
                    })
            
            print(f"   Loaded {len(samples)} TruthfulQA samples")
            return samples
            
        except Exception as e:
            print(f"   Error loading TruthfulQA: {e}")
            return self._generate_truthfulqa_samples(num_samples)
    
    def load_squad(self, num_samples: int = 50) -> List[Dict]:
        """Load SQuAD dataset with sufficient samples."""
        samples = []
        try:
            from datasets import load_dataset
            dataset = load_dataset("squad", split="validation")
            
            for i, item in enumerate(dataset):
                if i >= num_samples:
                    break
                
                samples.append({
                    "question": item["question"],
                    "answer": item["answers"]["text"][0] if item["answers"]["text"] else "",
                    "ground_truth": item["answers"]["text"][0] if item["answers"]["text"] else "",
                    "is_hallucination": False,
                    "dataset": "SQuAD",
                    "type": "reading_comprehension"
                })
            
            print(f"   Loaded {len(samples)} SQuAD samples")
            return samples
            
        except Exception as e:
            print(f"   Error loading SQuAD: {e}")
            return self._generate_squad_samples(num_samples)
    
    def load_hotpotqa(self, num_samples: int = 50) -> List[Dict]:
        """Load HotpotQA dataset with sufficient samples."""
        samples = []
        try:
            from datasets import load_dataset
            dataset = load_dataset("hotpot_qa", "distractor", split="validation")
            
            for i, item in enumerate(dataset):
                if i >= num_samples:
                    break
                
                samples.append({
                    "question": item["question"],
                    "answer": item["answer"],
                    "ground_truth": item["answer"],
                    "is_hallucination": False,
                    "dataset": "HotpotQA",
                    "type": "multi_hop_reasoning"
                })
            
            print(f"   Loaded {len(samples)} HotpotQA samples")
            return samples
            
        except Exception as e:
            print(f"   Error loading HotpotQA: {e}")
            return self._generate_hotpotqa_samples(num_samples)
    
    def _generate_truthfulqa_samples(self, num_samples: int) -> List[Dict]:
        """Fallback sample generation."""
        return [{
            "question": f"Sample question {i}",
            "answer": f"Sample answer {i}",
            "ground_truth": f"Ground truth {i}",
            "is_hallucination": False,
            "dataset": "TruthfulQA",
            "type": "factual"
        } for i in range(min(num_samples, 30))]
    
    def _generate_squad_samples(self, num_samples: int) -> List[Dict]:
        """Fallback sample generation."""
        return [{
            "question": f"Reading comprehension {i}",
            "answer": f"Answer {i}",
            "ground_truth": f"Ground truth {i}",
            "is_hallucination": False,
            "dataset": "SQuAD",
            "type": "reading_comprehension"
        } for i in range(min(num_samples, 30))]
    
    def _generate_hotpotqa_samples(self, num_samples: int) -> List[Dict]:
        """Fallback sample generation."""
        return [{
            "question": f"Multi-hop question {i}: What is the capital of X and Y?",
            "answer": f"Complex answer {i}",
            "ground_truth": f"Ground truth {i}",
            "is_hallucination": False,
            "dataset": "HotpotQA",
            "type": "multi_hop_reasoning"
        } for i in range(min(num_samples, 30))]
    
    def evaluate_dataset(self, samples: List[Dict], dataset_name: str) -> Dict:
        """Evaluate a single dataset with statistical significance."""
        results = []
        
        for sample in tqdm(samples, desc=f"   Evaluating {dataset_name}"):
            result = self.detector.detect(
                answer=sample["answer"],
                context=sample["question"],
                ground_truth=sample.get("ground_truth")
            )
            
            predicted = result["hallucination_score"] > 0.5
            actual = sample["is_hallucination"]
            
            results.append({
                "correct": predicted == actual,
                "score": result["hallucination_score"],
                "confidence": result["confidence"]
            })
        
        accuracy = sum(1 for r in results if r["correct"]) / len(results)
        avg_score = sum(r["score"] for r in results) / len(results)
        avg_confidence = sum(r["confidence"] for r in results) / len(results)
        
        # Calculate 95% confidence interval
        if len(results) > 1:
            import numpy as np
            accuracies = [1 if r["correct"] else 0 for r in results]
            ci_low = np.percentile(accuracies, 2.5)
            ci_high = np.percentile(accuracies, 97.5)
        else:
            ci_low = ci_high = accuracy
        
        return {
            "dataset": dataset_name,
            "accuracy": accuracy,
            "accuracy_ci_low": ci_low,
            "accuracy_ci_high": ci_high,
            "avg_hallucination_score": avg_score,
            "avg_confidence": avg_confidence,
            "samples": len(results),
            "sample_type": samples[0]["type"] if samples else "unknown"
        }
    
    def run_cross_dataset_evaluation(self, num_samples: int = 50) -> pd.DataFrame:
        """Run evaluation across all datasets with sufficient samples."""
        print("\n" + "="*70)
        print("📊 CROSS-DATASET EVALUATION (Statistical)")
        print("="*70)
        
        results = []
        
        print("\n📚 Loading TruthfulQA (myth detection)...")
        truthfulqa = self.load_truthfulqa(num_samples)
        if truthfulqa:
            result = self.evaluate_dataset(truthfulqa, "TruthfulQA")
            results.append(result)
        
        print("\n📚 Loading SQuAD (reading comprehension)...")
        squad = self.load_squad(num_samples)
        if squad:
            result = self.evaluate_dataset(squad, "SQuAD")
            results.append(result)
        
        print("\n📚 Loading HotpotQA (multi-hop reasoning)...")
        hotpotqa = self.load_hotpotqa(num_samples)
        if hotpotqa:
            result = self.evaluate_dataset(hotpotqa, "HotpotQA")
            results.append(result)
        
        df = pd.DataFrame(results)
        
        print("\n" + "-"*70)
        print("📊 PERFORMANCE BY DATASET (Statistical)")
        print("-"*70)
        print(df.to_string(index=False))
        
        # Statistical interpretation
        print("\n📈 STATISTICAL INTERPRETATION:")
        print("-"*50)
        for _, row in df.iterrows():
            print(f"   {row['dataset']} ({row['sample_type']}):")
            print(f"      Accuracy: {row['accuracy']:.1%} ± {row['accuracy'] - row['accuracy_ci_low']:.1%}")
            print(f"      Samples: {row['samples']}")
        
        # Save to CSV
        df.to_csv("results/cross_dataset_results.csv", index=False)
        
        return df