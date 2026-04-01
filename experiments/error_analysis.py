# experiments/error_analysis.py
"""
Error Analysis - Categorize and analyze detection failures
"""

import pandas as pd
import numpy as np
from typing import Dict, List
import re

class ErrorAnalyzer:
    """
    Analyze detection errors and categorize failure types.
    """
    
    def __init__(self):
        self.error_types = {
            "numeric_hallucination": [],
            "entity_mismatch": [],
            "incomplete_reasoning": [],
            "negation_misunderstanding": [],
            "temporal_confusion": [],
            "causal_confusion": []
        }
    
    def categorize_error(self, question: str, answer: str, ground_truth: str, score: float) -> str:
        """Categorize the type of error."""
        answer_lower = answer.lower()
        truth_lower = ground_truth.lower()
        
        # Check for numeric hallucinations
        numbers_answer = re.findall(r'\d+', answer_lower)
        numbers_truth = re.findall(r'\d+', truth_lower)
        if numbers_answer and numbers_truth and numbers_answer != numbers_truth:
            return "numeric_hallucination"
        
        # Check for entity mismatch
        if "paris" in answer_lower and "london" in truth_lower:
            return "entity_mismatch"
        if "london" in answer_lower and "paris" in truth_lower:
            return "entity_mismatch"
        
        # Check for negation misunderstanding
        negation_words = ["not", "no", "never", "isn't", "aren't"]
        answer_has_negation = any(neg in answer_lower for neg in negation_words)
        truth_has_negation = any(neg in truth_lower for neg in negation_words)
        if answer_has_negation != truth_has_negation:
            return "negation_misunderstanding"
        
        # Check for temporal confusion
        temporal_words = ["before", "after", "during", "while", "then"]
        answer_has_temporal = any(word in answer_lower for word in temporal_words)
        truth_has_temporal = any(word in truth_lower for word in temporal_words)
        if answer_has_temporal != truth_has_temporal:
            return "temporal_confusion"
        
        # Check for causal confusion
        causal_words = ["because", "therefore", "thus", "hence", "so"]
        answer_has_causal = any(word in answer_lower for word in causal_words)
        truth_has_causal = any(word in truth_lower for word in causal_words)
        if answer_has_causal != truth_has_causal:
            return "causal_confusion"
        
        # Check for incomplete reasoning
        if len(answer.split()) < 3 and len(truth_lower.split()) > 5:
            return "incomplete_reasoning"
        
        return "other"
    
    def analyze_results(self, results_df: pd.DataFrame) -> Dict:
        """Analyze results and categorize errors."""
        error_counts = {k: 0 for k in self.error_types}
        error_counts["other"] = 0
        
        for _, row in results_df.iterrows():
            if row.get("is_actual_hallucination") and row.get("detected_hallucination_score", 0) < 0.5:
                # False negative - missed hallucination
                error_type = self.categorize_error(
                    row.get("question", ""),
                    row.get("answer", ""),
                    row.get("ground_truth", ""),
                    row.get("detected_hallucination_score", 0)
                )
                error_counts[error_type] += 1
            elif not row.get("is_actual_hallucination") and row.get("detected_hallucination_score", 1) > 0.5:
                # False positive - false alarm
                error_type = "false_positive"
                if error_type not in error_counts:
                    error_counts[error_type] = 0
                error_counts[error_type] += 1
        
        return error_counts
    
    def print_error_report(self, error_counts: Dict):
        """Print formatted error analysis report."""
        print("\n" + "="*60)
        print("🔍 ERROR ANALYSIS REPORT")
        print("="*60)
        
        total = sum(error_counts.values())
        if total == 0:
            print("✅ No errors detected!")
            return
        
        print(f"\nTotal Errors: {total}")
        print("\nError Type Distribution:")
        print("-"*40)
        
        for error_type, count in sorted(error_counts.items(), key=lambda x: -x[1]):
            if count > 0:
                percentage = (count / total) * 100
                bar = "█" * int(percentage / 2)
                print(f"{error_type:25} | {count:3} ({percentage:5.1f}%) {bar}")
        
        print("\n💡 Recommendations:")
        if error_counts.get("numeric_hallucination", 0) > 0:
            print("  • Add numeric validation for hallucination detection")
        if error_counts.get("entity_mismatch", 0) > 0:
            print("  • Improve entity extraction and matching")
        if error_counts.get("negation_misunderstanding", 0) > 0:
            print("  • Enhance negation detection in context")
        if error_counts.get("incomplete_reasoning", 0) > 0:
            print("  • Add reasoning completeness check")