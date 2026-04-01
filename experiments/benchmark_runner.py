"""
Benchmark Runner - Run performance benchmarks on the system
"""

import time
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict
import os

from experiments.batch_evaluator import BatchEvaluator
from utils.config import config

class BenchmarkRunner:
    """
    Run benchmarks to evaluate system performance.
    """
    
    def __init__(self):
        self.evaluator = BatchEvaluator()
        self.benchmark_results = []
    
    def run_latency_benchmark(self, num_runs: int = 10) -> pd.DataFrame:
        """Measure latency per evaluation."""
        test_question = "What is the capital of France?"
        test_answer = "Paris"
        
        latencies = []
        
        print(f"Running latency benchmark ({num_runs} runs)...")
        
        for i in range(num_runs):
            start = time.time()
            result = self.evaluator.detector.detect(
                answer=test_answer,
                context=test_question
            )
            elapsed = time.time() - start
            latencies.append(elapsed)
            
            if (i + 1) % 5 == 0:
                print(f"  Run {i+1}/{num_runs}: {elapsed:.3f}s")
        
        df = pd.DataFrame({
            "run": range(1, num_runs + 1),
            "latency_seconds": latencies
        })
        
        print(f"\n📊 Latency Stats:")
        print(f"  Mean: {df['latency_seconds'].mean():.3f}s")
        print(f"  Std: {df['latency_seconds'].std():.3f}s")
        print(f"  Min: {df['latency_seconds'].min():.3f}s")
        print(f"  Max: {df['latency_seconds'].max():.3f}s")
        
        return df
    
    def run_accuracy_benchmark(self, num_samples: int = 20) -> pd.DataFrame:
        """Run accuracy benchmark on TruthfulQA."""
        print(f"Running accuracy benchmark on {num_samples} samples...")
        
        results_df = self.evaluator.evaluate_truthfulqa(num_samples=num_samples)
        self.evaluator.print_summary()
        
        return results_df
    
    def compare_thresholds(self) -> pd.DataFrame:
        """Compare performance at different hallucination thresholds."""
        thresholds = [0.3, 0.4, 0.5, 0.6, 0.7]
        results = []
        
        print("Comparing thresholds...")
        
        # Run evaluation once
        results_df = self.evaluator.evaluate_truthfulqa(num_samples=15)
        
        for threshold in thresholds:
            correct = 0
            total = 0
            
            for _, row in results_df.iterrows():
                predicted = row['detected_hallucination_score'] > threshold
                actual = row['is_actual_hallucination']
                total += 1
                if predicted == actual:
                    correct += 1
            
            accuracy = correct / total if total > 0 else 0
            
            results.append({
                "threshold": threshold,
                "accuracy": accuracy,
                "true_positives": ((results_df['detected_hallucination_score'] > threshold) & results_df['is_actual_hallucination']).sum(),
                "false_positives": ((results_df['detected_hallucination_score'] > threshold) & ~results_df['is_actual_hallucination']).sum(),
                "true_negatives": ((results_df['detected_hallucination_score'] <= threshold) & ~results_df['is_actual_hallucination']).sum(),
                "false_negatives": ((results_df['detected_hallucination_score'] <= threshold) & results_df['is_actual_hallucination']).sum()
            })
        
        df = pd.DataFrame(results)
        print("\n📊 Threshold Comparison:")
        print(df.to_string(index=False))
        
        return df
    
    def plot_results(self, results_df: pd.DataFrame = None):
        """Plot evaluation results."""
        if results_df is None:
            results_df = self.evaluator.evaluate_truthfulqa(num_samples=20)
        
        # Create plots directory
        os.makedirs(config.plots_dir, exist_ok=True)
        
        # 1. Distribution of hallucination scores
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 3, 1)
        plt.hist(results_df['detected_hallucination_score'], bins=20, edgecolor='black')
        plt.axvline(x=0.5, color='red', linestyle='--', label='Threshold (0.5)')
        plt.xlabel('Hallucination Score')
        plt.ylabel('Frequency')
        plt.title('Distribution of Hallucination Scores')
        plt.legend()
        
        # 2. Score by category
        plt.subplot(1, 3, 2)
        if 'category' in results_df.columns:
            category_means = results_df.groupby('category')['detected_hallucination_score'].mean()
            category_means.plot(kind='bar')
            plt.xlabel('Category')
            plt.ylabel('Average Hallucination Score')
            plt.title('Hallucination Score by Category')
            plt.xticks(rotation=45)
        
        # 3. Confidence vs Score
        plt.subplot(1, 3, 3)
        plt.scatter(results_df['detected_hallucination_score'], results_df['confidence'], alpha=0.6)
        plt.xlabel('Hallucination Score')
        plt.ylabel('Confidence')
        plt.title('Confidence vs Hallucination Score')
        
        plt.tight_layout()
        plt.savefig(os.path.join(config.plots_dir, 'evaluation_results.png'), dpi=150)
        plt.show()
        
        print(f"Plot saved to {config.plots_dir}/evaluation_results.png")
    
    def run_full_benchmark(self) -> Dict:
        """Run complete benchmark suite."""
        print("\n" + "="*60)
        print("🚀 RUNNING FULL BENCHMARK SUITE")
        print("="*60)
        
        # 1. Latency benchmark
        print("\n📈 1. LATENCY BENCHMARK")
        print("-"*40)
        latency_df = self.run_latency_benchmark(num_runs=10)
        
        # 2. Accuracy benchmark
        print("\n📈 2. ACCURACY BENCHMARK")
        print("-"*40)
        accuracy_df = self.run_accuracy_benchmark(num_samples=20)
        
        # 3. Threshold comparison
        print("\n📈 3. THRESHOLD COMPARISON")
        print("-"*40)
        threshold_df = self.compare_thresholds()
        
        # 4. Generate plots
        print("\n📈 4. GENERATING PLOTS")
        print("-"*40)
        self.plot_results(accuracy_df)
        
        # Summary
        summary = {
            "latency_mean": latency_df['latency_seconds'].mean(),
            "latency_std": latency_df['latency_seconds'].std(),
            "accuracy": self.evaluator.get_summary().get('detection_accuracy', 0),
            "avg_hallucination_score": self.evaluator.get_summary()['avg_hallucination_score'],
            "best_threshold": threshold_df.loc[threshold_df['accuracy'].idxmax(), 'threshold']
        }
        
        print("\n" + "="*60)
        print("📊 BENCHMARK SUMMARY")
        print("="*60)
        print(f"Average Latency: {summary['latency_mean']:.3f}s (±{summary['latency_std']:.3f})")
        print(f"Detection Accuracy: {summary['accuracy']:.1%}")
        print(f"Average Hallucination Score: {summary['avg_hallucination_score']:.3f}")
        print(f"Optimal Threshold: {summary['best_threshold']}")
        
        return summary