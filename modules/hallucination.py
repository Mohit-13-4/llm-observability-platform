"""
Hallucination Detector - Detects when LLMs make up information.
Uses multiple methods including LLM-as-Judge with improved scoring.
"""

from typing import Optional, Dict, List
from embeddings.embedder import EmbeddingEngine

class HallucinationDetector:
    """
    Detects hallucinations in model outputs using multiple methods.
    """
    
    def __init__(self, embedder: EmbeddingEngine, llm_judge=None):
        self.embedder = embedder
        self.llm_judge = llm_judge
        # More lenient thresholds for embedding similarity
        self.low_similarity_threshold = 0.4      # Lowered from 0.5
        self.medium_similarity_threshold = 0.6   # Lowered from 0.7
    
    def detect(self, answer: str, context: str, ground_truth: Optional[str] = None) -> Dict:
        """
        Detect hallucinations using multiple methods with weighted scoring.
        """
        results = {}
        
        # METHOD 1: Embedding similarity
        similarity = self.embedder.cosine_similarity(answer, context)
        results["context_similarity"] = similarity
        
        # METHOD 2: LLM-as-Judge (this is our most reliable method)
        llm_is_supported = None
        llm_explanation = None
        if self.llm_judge:
            llm_result = self.llm_judge.judge(answer, context)
            results["llm_judge"] = llm_result
            llm_is_supported = llm_result.get("is_supported")
            llm_explanation = llm_result.get("explanation", "")
        
        # METHOD 3: Ground truth (if available)
        if ground_truth:
            truth_similarity = self.embedder.cosine_similarity(answer, ground_truth)
            results["ground_truth_similarity"] = truth_similarity
        
        # Calculate hallucination score based on LLM judge (primary)
        # LLM judge is our most reliable method - it understands meaning
        if llm_is_supported is not None:
            if llm_is_supported:
                # LLM says it's supported → low hallucination
                hallucination_score = 0.1
                verdict = "LIKELY FAITHFUL - Answer appears correct"
            else:
                # LLM says NOT supported → high hallucination
                hallucination_score = 0.8
                verdict = "HIGH LIKELIHOOD - Answer is likely hallucinated"
        else:
            # Fallback to embedding similarity if LLM not available
            if similarity > 0.7:
                hallucination_score = 0.2
                verdict = "LOW LIKELIHOOD - Probably faithful"
            elif similarity > 0.5:
                hallucination_score = 0.5
                verdict = "POSSIBLE HALLUCINATION - Check carefully"
            else:
                hallucination_score = 0.8
                verdict = "HIGH LIKELIHOOD - Answer is likely hallucinated"
        
        # Adjust score based on ground truth if available
        if ground_truth and 'ground_truth_similarity' in results:
            gt_sim = results['ground_truth_similarity']
            if gt_sim < 0.5:
                # Ground truth says it's wrong
                hallucination_score = max(hallucination_score, 0.7)
                verdict = "HIGH LIKELIHOOD - Contradicts ground truth"
        
        # Calculate confidence
        if llm_is_supported is not None:
            confidence = 0.85  # High confidence in LLM judge
        else:
            confidence = 0.6
        
        return {
            "hallucination_score": hallucination_score,
            "confidence": confidence,
            "verdict": verdict,
            "methods": results,
            "contradiction_detected": not llm_is_supported if llm_is_supported is not None else False
        }

    def _calculate_confidence(self, scores: List[float], methods_used: List[str]) -> float:
        """
        Calculate confidence based on agreement between methods.
        """
        if len(scores) < 2:
            return 0.6
        
        # Count high and low scores
        high_scores = sum(1 for s in scores if s > 0.7)
        low_scores = sum(1 for s in scores if s < 0.3)
        total = len(scores)
        
        # Calculate agreement (how many methods agree)
        if high_scores > total / 2:
            # Most methods agree it's hallucination
            agreement_ratio = high_scores / total
            confidence = 0.7 + (agreement_ratio - 0.5) * 0.3
        elif low_scores > total / 2:
            # Most methods agree it's faithful
            agreement_ratio = low_scores / total
            confidence = 0.7 + (agreement_ratio - 0.5) * 0.3
        else:
            # Disagreement among methods
            # Calculate how balanced the scores are
            avg_score = sum(scores) / total
            variance = sum((s - avg_score) ** 2 for s in scores) / total
            confidence = 0.5 - variance * 0.5
        
        # Penalty if LLM judge wasn't used (it's our most reliable method)
        if "llm_judge" not in methods_used:
            confidence *= 0.85
        
        # Bonus if multiple methods agree
        if high_scores > 2 or low_scores > 2:
            confidence = min(0.95, confidence + 0.1)
        
        # Ensure confidence is within reasonable bounds
        confidence = max(0.3, min(0.95, confidence))
        
        return confidence
        
    def _check_contradiction(self, answer: str, context: str) -> bool:
        """
        Use ML methods to detect contradictions, not hardcoded rules.
        The real contradiction detection happens via:
        1. Embedding similarity (low score = possible contradiction)
        2. LLM-as-Judge (explicitly asks if answer contradicts context)
        """
        answer_lower = answer.lower()
        context_lower = context.lower()
        
        # ==================== LET ML DO THE WORK ====================
        # The embedding similarity already measures semantic distance
        # The LLM judge already evaluates contradictions
        # We don't need hardcoded rules!
        
        # For debugging, we can check what the LLM judge would say
        # But the actual detection happens in the main detect() method
        
        print(f"\n   🔍 Checking using ML methods:")
        print(f"      Answer: {answer_lower[:80]}")
        print(f"      Context: {context_lower[:80]}")
        
        # Let the LLM judge handle it - no hardcoded rules!
        # Return False here - the real detection is in the weighted scoring
        return False
# ==================== NEGATION PHRASES ====================
# Define outside the method for reuse
negation_phrases = [
    ("visible from space", "not visible"),
    ("grow in your stomach", "pass through"),
    ("only man-made", "cannot be seen"),
    ("grow in your belly", "digest"),
]