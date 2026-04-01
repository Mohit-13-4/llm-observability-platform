"""
Main Evaluation Pipeline - Orchestrates all evaluation modules.
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any
from embeddings.embedder import EmbeddingEngine
from modules.hallucination import HallucinationDetector
from llm.judge import LLMJudge
from utils.config import config

@dataclass
class EvaluationResult:
    """Container for evaluation results."""
    faithfulness_score: float
    hallucination_score: float
    overall_score: float = 0.0
    similarity_score: Optional[float] = None
    consistency_score: Optional[float] = None
    details: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        hallucination_inv = 1 - self.hallucination_score
        self.overall_score = (
            0.5 * self.faithfulness_score +
            0.5 * hallucination_inv
        )
        
        if self.similarity_score is not None:
            self.overall_score = (self.overall_score + self.similarity_score) / 2
        
        self.overall_score = max(0.0, min(1.0, self.overall_score))

class LLMEvaluator:
    """Main evaluation pipeline."""
    
    def __init__(self):
        print("Initializing LLM Evaluator...")
        self.embedder = EmbeddingEngine()
        self.llm_judge = LLMJudge()  # Add LLM judge!
        self.hallucination = HallucinationDetector(self.embedder, self.llm_judge)
        print("✅ All modules initialized!")
    
    def evaluate_single(self, 
                        prompt: str,
                        model_output: str,
                        context: Optional[str] = None,
                        ground_truth: Optional[str] = None) -> EvaluationResult:
        """Run full evaluation on a single sample."""
        context = context or prompt
        
        hallucination_result = self.hallucination.detect(
            model_output, context, ground_truth
        )
        
        similarity = hallucination_result["methods"].get("context_similarity", 0.5)
        faithfulness_score = similarity
        hallucination_score = hallucination_result["hallucination_score"]
        
        similarity_score = None
        if ground_truth:
            similarity_score = hallucination_result["methods"].get("ground_truth_similarity", None)
        
        return EvaluationResult(
            faithfulness_score=faithfulness_score,
            hallucination_score=hallucination_score,
            similarity_score=similarity_score,
            consistency_score=None,
            details={
                "hallucination": hallucination_result,
                "prompt": prompt,
                "model_output": model_output,
                "context": context
            }
        )