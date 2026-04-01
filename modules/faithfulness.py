"""
Faithfulness Module - Checks if answer is faithful to the provided context.

How it works:
1. Convert answer and context to vector embeddings
2. Calculate cosine similarity between them
3. Higher similarity = more faithful
"""

from embeddings.embedder import EmbeddingEngine
from typing import Dict

class FaithfulnessModule:
    """
    Evaluates whether the answer is faithful to the provided context.
    
    Score range: 0 (completely unfaithful) to 1 (perfectly faithful)
    """
    
    def __init__(self, embedder: EmbeddingEngine):
        """
        Initialize with an embedding engine.
        
        The embedder is passed in (dependency injection) so we can reuse
        the same embedder across multiple modules, saving memory.
        """
        self.embedder = embedder
    
    def evaluate(self, answer: str, context: str) -> Dict:
        """
        Evaluate faithfulness of answer to context.
        
        Args:
            answer: The model's generated answer
            context: The context the model was supposed to use
        
        Returns:
            Dictionary with:
            - score: Faithfulness score (0-1)
            - similarity: Raw cosine similarity
            - level: 'high', 'moderate', or 'low'
            - explanation: Human-readable explanation
        """
        # Step 1: Calculate cosine similarity between answer and context
        # This gives us a number between -1 and 1, but for text it's usually 0-1
        similarity = self.embedder.cosine_similarity(answer, context)
        
        # Step 2: Ensure score is between 0 and 1
        faithfulness_score = max(0.0, min(1.0, similarity))
        
        # Step 3: Determine faithfulness level based on score
        if faithfulness_score >= 0.8:
            level = "high"
            explanation = (
                f"The answer is well supported by the context. "
                f"(similarity: {faithfulness_score:.3f})"
            )
        elif faithfulness_score >= 0.6:
            level = "moderate"
            explanation = (
                f"The answer is partially supported by the context. "
                f"(similarity: {faithfulness_score:.3f})"
            )
        else:
            level = "low"
            explanation = (
                f"The answer may not be supported by the context. "
                f"(similarity: {faithfulness_score:.3f})"
            )
        
        return {
            "score": faithfulness_score,
            "similarity": similarity,
            "level": level,
            "explanation": explanation
        }
    
    def batch_evaluate(self, answers: list, contexts: list) -> list:
        """
        Evaluate faithfulness for multiple pairs at once.
        
        This is more efficient than calling evaluate() in a loop
        because we can batch the embedding computation.
        """
        # Step 1: Get all embeddings at once
        similarities = self.embedder.batch_similarity(answers, contexts)
        
        # Step 2: Process each result
        results = []
        for i, sim in enumerate(similarities):
            sim = float(sim)
            if sim >= 0.8:
                level = "high"
            elif sim >= 0.6:
                level = "moderate"
            else:
                level = "low"
            
            results.append({
                "score": sim,
                "similarity": sim,
                "level": level,
                "explanation": f"Similarity: {sim:.3f}"
            })
        
        return results