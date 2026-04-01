"""
Ensemble Detector - Combines multiple hallucination detection methods
"""

import numpy as np
from typing import List, Dict, Optional

class EnsembleDetector:
    """
    Combines multiple detectors for better accuracy.
    """
    
    def __init__(self, detectors: List):
        """
        Args:
            detectors: List of detector objects
        """
        self.detectors = detectors
        self.weights = [1.0 / len(detectors)] * len(detectors)
    
    def set_weights(self, weights: List[float]):
        """Set custom weights for each detector."""
        self.weights = weights / np.sum(weights)
    
    def detect(self, answer: str, context: str, ground_truth: Optional[str] = None) -> Dict:
        """
        Combine predictions from all detectors.
        """
        results = []
        scores = []
        
        for detector in self.detectors:
            result = detector.detect(answer, context, ground_truth)
            results.append(result)
            scores.append(result["hallucination_score"])
        
        # Weighted average
        ensemble_score = np.average(scores, weights=self.weights)
        
        # Majority vote for verdict
        verdicts = [r["verdict"] for r in results]
        from collections import Counter
        most_common_verdict = Counter(verdicts).most_common(1)[0][0]
        
        # Calculate confidence (lower std = higher confidence)
        confidence = 1.0 - np.std(scores)
        
        return {
            "hallucination_score": ensemble_score,
            "confidence": confidence,
            "verdict": most_common_verdict,
            "individual_results": results,
            "method": "ensemble"
        }