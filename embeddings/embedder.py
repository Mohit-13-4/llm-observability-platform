"""
Embedding Engine - Converts text to vector representations.
"""

import torch
import numpy as np
from sentence_transformers import SentenceTransformer
from typing import List, Union
from utils.config import config

class EmbeddingEngine:
    """Handles text embedding generation and similarity computations."""
    
    def __init__(self, model_name: str = None):
        self.model_name = model_name or config.embedding_model
        self.device = torch.device(config.llm_device)
        
        print(f"Loading embedding model: {self.model_name}")
        self.model = SentenceTransformer(self.model_name)
        self.model.to(self.device)
        print(f"✅ Embedding model loaded on {self.device}")
        
        self._cache = {}
    
    def embed(self, texts: Union[str, List[str]]) -> np.ndarray:
        if isinstance(texts, str):
            texts = [texts]
        
        results = []
        texts_to_embed = []
        indices = []
        
        for i, t in enumerate(texts):
            if t in self._cache:
                results.append(self._cache[t])
            else:
                texts_to_embed.append(t)
                indices.append(i)
        
        if texts_to_embed:
            embeddings = self.model.encode(
                texts_to_embed,
                convert_to_numpy=True,
                show_progress_bar=False
            )
            for idx, emb in zip(indices, embeddings):
                self._cache[texts[idx]] = emb
                results.append(emb)
        
        final_results = [None] * len(texts)
        for i, idx in enumerate(indices):
            final_results[idx] = results[i]
        
        cache_idx = 0
        for i in range(len(texts)):
            if final_results[i] is None:
                final_results[i] = results[cache_idx]
                cache_idx += 1
        
        return np.array(final_results)
    
    def cosine_similarity(self, text1: str, text2: str) -> float:
        emb1 = self.embed(text1)
        emb2 = self.embed(text2)
        
        norm1 = np.linalg.norm(emb1)
        norm2 = np.linalg.norm(emb2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        similarity = np.dot(emb1, emb2.T) / (norm1 * norm2)
        return float(similarity[0] if similarity.size > 1 else similarity)
    
    def batch_similarity(self, texts1: List[str], texts2: List[str]) -> np.ndarray:
        emb1 = self.embed(texts1)
        emb2 = self.embed(texts2)
        
        emb1 = emb1 / np.linalg.norm(emb1, axis=1, keepdims=True)
        emb2 = emb2 / np.linalg.norm(emb2, axis=1, keepdims=True)
        
        return np.dot(emb1, emb2.T)
    
    def clear_cache(self):
        self._cache.clear()