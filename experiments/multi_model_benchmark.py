# experiments/multi_model_benchmark.py (UPDATED)
"""
Multi-Model Benchmark - Compare hallucination rates across different LLMs
WITH MULTIPLE WORKING MODELS
"""

import time
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
from typing import Dict, List
from embeddings.embedder import EmbeddingEngine
from llm.judge import LLMJudge
from modules.hallucination import HallucinationDetector
from data_loaders.truthfulqa_loader import TruthfulQALoader

class MultiModelBenchmark:
    """Compare hallucination rates across different LLM models."""
    
    def __init__(self):
        self.embedder = EmbeddingEngine()
        self.llm_judge = LLMJudge()
        self.detector = HallucinationDetector(self.embedder, self.llm_judge)
        self.results = []
    
    def load_model(self, model_name: str):
        """Load a model for evaluation with memory optimization."""
        print(f"   Loading {model_name}...")
        try:
            # Use 8-bit quantization for larger models
            from transformers import BitsAndBytesConfig
            
            quantization_config = BitsAndBytesConfig(
                load_in_8bit=True,
                llm_int8_threshold=6.0
            )
            
            tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                quantization_config=quantization_config,
                device_map="auto",
                trust_remote_code=True
            )
            model.eval()
            return model, tokenizer
        except Exception as e:
            print(f"   ⚠️ Could not load {model_name}: {e}")
            return None, None
    
    def generate_response(self, model, tokenizer, prompt: str, max_tokens: int = 50) -> str:
        """Generate a response from the model."""
        try:
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
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
            response = response[len(prompt):].strip()
            return response[:200]
        except Exception as e:
            print(f"   Generation error: {e}")
            return ""
    
    def evaluate_model(self, model_name: str, samples: List[Dict], num_samples: int = 20) -> Dict:
        """Evaluate a single model using ML-based hallucination detection."""
        print(f"\n   📊 Evaluating {model_name}...")
        
        model, tokenizer = self.load_model(model_name)
        if model is None:
            return {"model": model_name, "error": "Failed to load", "hallucination_rate": 0.5}
        
        results = []
        latencies = []
        
        for sample in tqdm(samples[:num_samples], desc=f"   Testing {model_name}", leave=False):
            start = time.time()
            response = self.generate_response(model, tokenizer, sample["question"])
            latency = time.time() - start
            
            if not response:
                continue
            
            eval_result = self.detector.detect(
                answer=response,
                context=sample["question"],
                ground_truth=sample["correct_answer"]
            )
            
            results.append({
                "is_hallucination": eval_result["hallucination_score"] > 0.5,
                "score": eval_result["hallucination_score"],
                "confidence": eval_result["confidence"]
            })
            latencies.append(latency)
        
        if not results:
            return {"model": model_name, "error": "No valid responses", "hallucination_rate": 0.5}
        
        hallucination_rate = sum(1 for r in results if r["is_hallucination"]) / len(results)
        avg_score = sum(r["score"] for r in results) / len(results)
        avg_confidence = sum(r["confidence"] for r in results) / len(results)
        avg_latency = sum(latencies) / len(latencies)
        
        return {
            "model": model_name,
            "hallucination_rate": hallucination_rate,
            "avg_hallucination_score": avg_score,
            "avg_confidence": avg_confidence,
            "avg_latency": avg_latency,
            "samples": len(results)
        }
    
    def run_benchmark(self, models: List[str] = None, num_samples: int = 20) -> pd.DataFrame:
        """Run benchmark on multiple models."""
        if models is None:
            # Models that can run on RTX 3050 4GB
            models = [
                "microsoft/phi-1_5",           # Already working
                "TinyLlama/TinyLlama-1.1B-Chat-v1.0"  # 1.1B parameters
            ]
        
        print("\n" + "="*70)
        print("📊 MULTI-MODEL BENCHMARK (Statistical)")
        print("="*70)
        
        loader = TruthfulQALoader(use_real_data=False)
        samples = loader.get_samples(num_samples)
        
        all_results = []
        for model_name in models:
            result = self.evaluate_model(model_name, samples, num_samples)
            if "error" not in result:
                all_results.append(result)
        
        df = pd.DataFrame(all_results)
        
        print("\n" + "-"*70)
        print("📊 MODEL COMPARISON RESULTS")
        print("-"*70)
        print(df.to_string(index=False))
        
        # Statistical interpretation
        print("\n📈 STATISTICAL INTERPRETATION:")
        print("-"*50)
        for _, row in df.iterrows():
            print(f"   {row['model']}:")
            print(f"      Hallucination Rate: {row['hallucination_rate']:.1%}")
            print(f"      Avg Score: {row['avg_hallucination_score']:.3f}")
            print(f"      Avg Latency: {row['avg_latency']:.1f}s")
        
        df.to_csv("results/multi_model_benchmark.csv", index=False)
        
        return df