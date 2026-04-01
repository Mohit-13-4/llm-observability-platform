# experiments/model_comparison.py
"""
Multi-Model Benchmark - Compare hallucination rates across different LLMs
"""

import time
import pandas as pd
import numpy as np
from tqdm import tqdm
from typing import Dict, List
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

from embeddings.embedder import EmbeddingEngine
from llm.judge import LLMJudge
from modules.hallucination import HallucinationDetector
from data_loaders.truthfulqa_loader import TruthfulQALoader

class MultiModelBenchmark:
    """
    Compare hallucination rates across different LLM models.
    """
    
    def __init__(self):
        self.embedder = EmbeddingEngine()
        self.llm_judge = LLMJudge()
        self.detector = HallucinationDetector(self.embedder, self.llm_judge)
        self.results = []
    
    def load_model(self, model_name: str):
        """Load a model for evaluation."""
        print(f"Loading {model_name}...")
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                trust_remote_code=True
            )
            if torch.cuda.is_available():
                model = model.cuda()
            model.eval()
            return model, tokenizer
        except Exception as e:
            print(f"Error loading {model_name}: {e}")
            return None, None
    
    def generate_response(self, model, tokenizer, prompt: str, max_tokens: int = 50) -> str:
        """Generate a response from the model."""
        try:
            inputs = tokenizer(prompt, return_tensors="pt")
            if torch.cuda.is_available():
                inputs = {k: v.cuda() for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=tokenizer.eos_token_id
                )
            
            response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            # Remove the prompt from the response
            response = response[len(prompt):].strip()
            return response
        except Exception as e:
            print(f"Generation error: {e}")
            return ""
    
    def evaluate_model(self, model_name: str, samples: List[Dict], num_samples: int = 20) -> Dict:
        """Evaluate a single model on hallucination detection."""
        print(f"\n{'='*60}")
        print(f"Evaluating {model_name}")
        print(f"{'='*60}")
        
        # Load model
        model, tokenizer = self.load_model(model_name)
        if model is None:
            return {"model": model_name, "error": "Failed to load"}
        
        results = []
        start_time = time.time()
        
        for sample in tqdm(samples[:num_samples], desc=f"Testing {model_name}"):
            # Generate response
            response = self.generate_response(model, tokenizer, sample["question"])
            
            if not response:
                continue
            
            # Evaluate the response
            eval_result = self.detector.detect(
                answer=response,
                context=sample["question"],
                ground_truth=sample["correct_answer"]
            )
            
            results.append({
                "question": sample["question"],
                "response": response,
                "hallucination_score": eval_result["hallucination_score"],
                "is_hallucination": eval_result["hallucination_score"] > 0.5,
                "latency": time.time() - start_time
            })
        
        elapsed = time.time() - start_time
        
        # Calculate metrics
        df = pd.DataFrame(results)
        hallucination_rate = df["is_hallucination"].mean()
        avg_score = df["hallucination_score"].mean()
        avg_latency = df["latency"].mean() if len(df) > 0 else 0
        
        return {
            "model": model_name,
            "hallucination_rate": hallucination_rate,
            "avg_hallucination_score": avg_score,
            "avg_latency": avg_latency,
            "num_samples": len(results),
            "total_time": elapsed,
            "results": df
        }
    
    # experiments/model_comparison.py - Update the run_comparison method

    def run_comparison(self, models: List[str], num_samples: int = 5) -> pd.DataFrame:  # Reduced from 20 to 5
        """Run comparison across multiple models."""
        print("Loading TruthfulQA samples...")
        loader = TruthfulQALoader(use_real_data=False)
        samples = loader.get_samples(num_samples)  # Now using fewer samples
        print(f"Using {len(samples)} samples for evaluation")
        
        all_results = []
        
        for model_name in models:
            result = self.evaluate_model(model_name, samples, num_samples)
            if "error" not in result:
                all_results.append({
                    "Model": result["model"],
                    "Hallucination Rate": f"{result['hallucination_rate']:.1%}",
                    "Avg Score": f"{result['avg_hallucination_score']:.3f}",
                    "Avg Latency": f"{result['avg_latency']:.2f}s",
                    "Samples": result["num_samples"]
                })
        
        df = pd.DataFrame(all_results)
        return df
    
    def print_comparison_table(self, df: pd.DataFrame):
        """Print a formatted comparison table."""
        print("\n" + "="*70)
        print("📊 MODEL COMPARISON BENCHMARK")
        print("="*70)
        print(df.to_string(index=False))
        print("="*70)
        
        # Find best model
        best = df.iloc[0] if len(df) > 0 else None
        if best is not None:
            print(f"\n🏆 Best Model: {best['Model']} with {best['Hallucination Rate']} hallucination rate")