"""
Batch Evaluation - Run evaluations on multiple samples
"""

import time
import pandas as pd
import numpy as np
from tqdm import tqdm
from typing import List, Dict, Optional
from datetime import datetime
import os

from embeddings.embedder import EmbeddingEngine
from llm.judge import LLMJudge
from modules.hallucination import HallucinationDetector
from data_loaders.truthfulqa_loader import TruthfulQALoader
from utils.config import config

class BatchEvaluator:
    """
    Run batch evaluations on datasets.
    """
    
    def __init__(self):
        self.embedder = EmbeddingEngine()
        self.llm_judge = LLMJudge()
        self.detector = HallucinationDetector(self.embedder, self.llm_judge)
        self.results = []
    
    def evaluate_pairs(self, pairs: List[Dict], verbose: bool = True) -> pd.DataFrame:
        """
        Evaluate a list of (question, answer, ground_truth) pairs.
        Now uses the context field properly.
        """
        results = []
        
        iterator = tqdm(pairs, desc="Evaluating") if verbose else pairs
        
        for pair in iterator:
            # Use the context from the pair if available
            context = pair.get("context", pair["question"])
            
            result = self.detector.detect(
                answer=pair["answer"],
                context=context,  # Use proper context
                ground_truth=pair.get("ground_truth")
            )
            
            results.append({
                "question": pair["question"],
                "answer": pair["answer"],
                "is_actual_hallucination": pair.get("is_hallucination", False),
                "detected_hallucination_score": result["hallucination_score"],
                "confidence": result.get("confidence", 0.5),
                "verdict": result["verdict"],
                "category": pair.get("category", "unknown"),
                "context_similarity": result["methods"].get("context_similarity", 0),
                "contradiction_detected": result.get("contradiction_detected", False),
                "llm_judge_supported": result["methods"].get("llm_judge", {}).get("is_supported", None) if "llm_judge" in result["methods"] else None
            })
        
        self.results = results
        return pd.DataFrame(results)
    
    def evaluate_truthfulqa(self, num_samples: int = 20, include_correct: bool = True, 
                           include_incorrect: bool = True) -> pd.DataFrame:
        """
        Evaluate on TruthfulQA dataset.
        """
        print(f"Loading TruthfulQA samples...")
        loader = TruthfulQALoader(use_real_data=False)  # Start with sample data
        pairs = loader.create_evaluation_pairs(
            include_correct=include_correct,
            include_incorrect=include_incorrect,
            num_samples=num_samples
        )
        
        print(f"Created {len(pairs)} evaluation pairs")
        return self.evaluate_pairs(pairs)
    
    def get_summary(self) -> Dict:
        """Get summary statistics from results."""
        if not self.results:
            return {}
        
        df = pd.DataFrame(self.results)
        
        # Calculate accuracy (if we have ground truth)
        if 'is_actual_hallucination' in df.columns:
            df['correct_detection'] = (
                (df['detected_hallucination_score'] > 0.5) == df['is_actual_hallucination']
            )
            accuracy = df['correct_detection'].mean()
        else:
            accuracy = None
        
        summary = {
            "total_samples": len(df),
            "avg_hallucination_score": df['detected_hallucination_score'].mean(),
            "avg_confidence": df['confidence'].mean(),
            "hallucination_rate": (df['detected_hallucination_score'] > 0.5).mean(),
            "high_risk_samples": (df['detected_hallucination_score'] > 0.7).sum(),
            "medium_risk_samples": ((df['detected_hallucination_score'] > 0.4) & (df['detected_hallucination_score'] <= 0.7)).sum(),
            "low_risk_samples": (df['detected_hallucination_score'] <= 0.4).sum(),
        }
        
        if accuracy:
            summary["detection_accuracy"] = accuracy
        
        # Category breakdown
        if 'category' in df.columns:
            summary["by_category"] = df.groupby('category').agg({
                'detected_hallucination_score': 'mean',
                'confidence': 'mean'
            }).to_dict()
        
        return summary
    
    def save_results(self, filename: str = None):
        """Save results to CSV."""
        if not self.results:
            print("No results to save")
            return
        
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"evaluation_results_{timestamp}.csv"
        
        filepath = os.path.join(config.results_dir, filename)
        df = pd.DataFrame(self.results)
        df.to_csv(filepath, index=False)
        print(f"Results saved to {filepath}")
        return filepath
    
    def print_summary(self):
        """Print summary statistics."""
        summary = self.get_summary()
        
        print("\n" + "="*60)
        print("📊 EVALUATION SUMMARY")
        print("="*60)
        print(f"Total Samples: {summary['total_samples']}")
        print(f"Average Hallucination Score: {summary['avg_hallucination_score']:.3f}")
        print(f"Average Confidence: {summary['avg_confidence']:.3f}")
        print(f"Hallucination Rate: {summary['hallucination_rate']:.1%}")
        print(f"Risk Distribution:")
        print(f"  🔴 High Risk: {summary['high_risk_samples']}")
        print(f"  🟡 Medium Risk: {summary['medium_risk_samples']}")
        print(f"  🟢 Low Risk: {summary['low_risk_samples']}")
        
        if 'detection_accuracy' in summary:
            print(f"Detection Accuracy: {summary['detection_accuracy']:.1%}")