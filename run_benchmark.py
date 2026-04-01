"""
Main script to run evaluations and benchmarks
"""

import sys
import argparse
from experiments.benchmark_runner import BenchmarkRunner
from experiments.batch_evaluator import BatchEvaluator

def main():
    parser = argparse.ArgumentParser(description="LLM Observability Platform")
    parser.add_argument("--benchmark", action="store_true", help="Run full benchmark suite")
    parser.add_argument("--evaluate", action="store_true", help="Run single evaluation")
    parser.add_argument("--question", type=str, help="Question to evaluate")
    parser.add_argument("--answer", type=str, help="Answer to evaluate")
    parser.add_argument("--context", type=str, help="Context (optional)")
    parser.add_argument("--ground_truth", type=str, help="Ground truth (optional)")
    
    args = parser.parse_args()
    
    if args.benchmark:
        print("Running full benchmark...")
        runner = BenchmarkRunner()
        results = runner.run_full_benchmark()
        
    elif args.evaluate and args.question and args.answer:
        print("Running single evaluation...")
        evaluator = BatchEvaluator()
        
        pair = [{
            "question": args.question,
            "answer": args.answer,
            "context": args.context,
            "ground_truth": args.ground_truth
        }]
        
        df = evaluator.evaluate_pairs(pair)
        print(df.to_string())
        
        # Print detailed results
        result = evaluator.results[0]
        print(f"\n📊 Results:")
        print(f"  Hallucination Score: {result['detected_hallucination_score']:.3f}")
        print(f"  Confidence: {result['confidence']:.3f}")
        print(f"  Verdict: {result['verdict']}")
        
    else:
        print("Usage:")
        print("  python run_benchmark.py --benchmark")
        print("  python run_benchmark.py --evaluate --question 'What is...' --answer 'Paris'")
        print("  python run_benchmark.py --evaluate --question 'What is...' --answer 'London' --ground_truth 'Paris'")

if __name__ == "__main__":
    main()