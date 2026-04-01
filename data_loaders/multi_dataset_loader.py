# data_loaders/multi_dataset_loader.py
"""
Multi-Dataset Loader - Loads SQuAD, HotpotQA, Natural Questions
"""

import pandas as pd
from typing import List, Dict, Optional

class MultiDatasetLoader:
    """
    Loader for multiple QA datasets.
    """
    
    def __init__(self):
        self.datasets = {}
    
    def load_squad(self, split: str = "validation") -> List[Dict]:
        """Load SQuAD dataset."""
        try:
            from datasets import load_dataset
            print("Loading SQuAD dataset...")
            dataset = load_dataset("squad", split=split)
            
            samples = []
            for item in dataset:
                samples.append({
                    "question": item["question"],
                    "context": item["context"],
                    "answer": item["answers"]["text"][0] if item["answers"]["text"] else "",
                    "dataset": "squad"
                })
            
            print(f"✅ Loaded {len(samples)} SQuAD samples")
            return samples
        except Exception as e:
            print(f"Error loading SQuAD: {e}")
            return self._create_squad_samples()
    
    def load_hotpotqa(self, split: str = "validation") -> List[Dict]:
        """Load HotpotQA dataset."""
        try:
            from datasets import load_dataset
            print("Loading HotpotQA dataset...")
            dataset = load_dataset("hotpot_qa", "distractor", split=split)
            
            samples = []
            for item in dataset:
                samples.append({
                    "question": item["question"],
                    "context": item["context"],
                    "answer": item["answer"],
                    "dataset": "hotpotqa"
                })
            
            print(f"✅ Loaded {len(samples)} HotpotQA samples")
            return samples
        except Exception as e:
            print(f"Error loading HotpotQA: {e}")
            return self._create_hotpotqa_samples()
    
    def load_natural_questions(self, split: str = "validation") -> List[Dict]:
        """Load Natural Questions dataset."""
        try:
            from datasets import load_dataset
            print("Loading Natural Questions dataset...")
            dataset = load_dataset("natural_questions", split=split)
            
            samples = []
            for item in dataset:
                if "long_answer" in item and item["long_answer"]:
                    samples.append({
                        "question": item["question"]["text"],
                        "context": item["document"]["title"],
                        "answer": item["long_answer"][0]["text"] if item["long_answer"] else "",
                        "dataset": "natural_questions"
                    })
            
            print(f"✅ Loaded {len(samples)} Natural Questions samples")
            return samples
        except Exception as e:
            print(f"Error loading Natural Questions: {e}")
            return self._create_nq_samples()
    
    def load_all(self, max_samples: int = 50) -> pd.DataFrame:
        """Load all datasets and combine."""
        squad = self.load_squad()
        hotpotqa = self.load_hotpotqa()
        nq = self.load_natural_questions()
        
        all_samples = squad[:max_samples] + hotpotqa[:max_samples] + nq[:max_samples]
        df = pd.DataFrame(all_samples)
        print(f"\n📊 Combined dataset: {len(df)} samples from 3 datasets")
        return df
    
    def _create_squad_samples(self):
        """Fallback samples."""
        return [
            {"question": "What is the capital of France?", "context": "Paris is the capital of France.", "answer": "Paris", "dataset": "squad"},
            {"question": "Who wrote Romeo and Juliet?", "context": "William Shakespeare wrote Romeo and Juliet.", "answer": "William Shakespeare", "dataset": "squad"},
        ]
    
    def _create_hotpotqa_samples(self):
        """Fallback samples."""
        return [
            {"question": "Which city hosted the 2016 Olympics?", "context": "Rio de Janeiro hosted the 2016 Olympics.", "answer": "Rio de Janeiro", "dataset": "hotpotqa"},
        ]
    
    def _create_nq_samples(self):
        """Fallback samples."""
        return [
            {"question": "What is the highest mountain?", "context": "Mount Everest is the highest mountain.", "answer": "Mount Everest", "dataset": "natural_questions"},
        ]