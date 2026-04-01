"""
LLM-as-a-Judge with caching for speed
"""

import torch
import hashlib
import pickle
import os
from typing import Dict, Optional

class LLMJudge:
    """Local LLM judge with caching for faster repeated evaluations."""
    
    def __init__(self, model_name: str = "microsoft/phi-2", use_cache: bool = True):
        self.model_name = model_name
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.tokenizer = None
        self.use_cache = use_cache
        self.cache = {}
        self.cache_file = "llm_cache.pkl"
        self._load_cache()
        
        print(f"Loading LLM judge: {model_name}")
        self._load_model()
    
    def _load_cache(self):
        """Load cache from disk."""
        if os.path.exists(self.cache_file):
            try:
                with open(self.cache_file, 'rb') as f:
                    self.cache = pickle.load(f)
                print(f"✅ Loaded {len(self.cache)} cached evaluations")
            except:
                print("⚠️ Could not load cache, starting fresh")
    
    def _save_cache(self):
        """Save cache to disk."""
        try:
            with open(self.cache_file, 'wb') as f:
                pickle.dump(self.cache, f)
        except:
            pass
    
    def _get_cache_key(self, answer: str, context: str) -> str:
        """Generate unique cache key."""
        key_str = f"{answer[:200]}|{context[:200]}"
        return hashlib.md5(key_str.encode()).hexdigest()
    
    def _load_model(self):
        """Load the local LLM model."""
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                trust_remote_code=True,
                padding_side="left"
            )
            
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                trust_remote_code=True
            )
            
            if self.device == "cuda":
                self.model = self.model.cuda()
            
            self.model.eval()
            print(f"✅ LLM judge loaded on {self.device}")
            
        except Exception as e:
            print(f"⚠️ Failed to load LLM judge: {e}")
            self.model = None
    
    def judge(self, answer: str, context: str) -> Dict:
        """Judge with caching."""
        # Check cache first
        if self.use_cache:
            cache_key = self._get_cache_key(answer, context)
            if cache_key in self.cache:
                return self.cache[cache_key]
        
        # Run actual judgment
        result = self._judge_impl(answer, context)
        
        # Save to cache
        if self.use_cache:
            self.cache[cache_key] = result
            self._save_cache()
        
        return result
    
    def _judge_impl(self, answer: str, context: str) -> Dict:
        """Actual judgment implementation."""
        if self.model is None:
            return self._fallback_judge(answer, context)
        
        # More strict prompt with clear instructions
        prompt = f"""You are a strict truthfulness evaluator. Your ONLY job is to check if the answer is directly supported by the context.

    RULES:
    1. Answer YES only if the exact information is in the context
    2. Answer NO if the answer adds information not in the context
    3. Answer NO if the answer contradicts the context
    4. IGNORE your own knowledge - ONLY use the context provided

    CONTEXT: {context}

    ANSWER: {answer}

    Based ONLY on the context above, is the answer FULLY supported?
    Answer with YES or NO only.

    Answer:"""
        
        try:
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=512
            )
            
            if self.device == "cuda":
                inputs = {k: v.cuda() for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=10,  # Shorter response for YES/NO
                    temperature=0.1,     # Very low temperature for consistency
                    do_sample=True,
                    pad_token_id=self.tokenizer.pad_token_id
                )
            
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            response = response.replace(prompt, "").strip().upper()
            
            # Extract YES/NO
            is_supported = "YES" in response[:10]
            
            return {
                "is_supported": is_supported,
                "confidence": 0.9 if is_supported else 0.85,
                "explanation": response[:100],
                "method": "llm-as-judge"
            }
            
        except Exception as e:
            print(f"LLM judge error: {e}")
            return self._fallback_judge(answer, context)
    
    def _fallback_judge(self, answer: str, context: str) -> Dict:
        """Fallback when LLM is not available."""
        return {
            "is_supported": None,
            "confidence": 0.5,
            "explanation": "LLM judge not available. Using embedding-based evaluation.",
            "method": "fallback"
        }
    
    def clear_cache(self):
        """Clear the cache."""
        self.cache = {}
        if os.path.exists(self.cache_file):
            os.remove(self.cache_file)
        print("✅ Cache cleared")