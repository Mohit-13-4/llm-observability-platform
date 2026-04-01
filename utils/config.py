"""
Configuration management for LLM Observability Platform.
"""

import os
from dataclasses import dataclass
from typing import Optional

@dataclass
class Config:
    # ==================== Embedding Settings ====================
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_dim: int = 384
    
    # ==================== Local LLM Settings ====================
    use_local_llm: bool = True
    local_llm_model: str = "microsoft/phi-2"  # Small, efficient, runs on laptop
    llm_device: str = "cuda" if __import__('torch').cuda.is_available() else "cpu"
    
    # ==================== Evaluation Weights ====================
    faithfulness_weight: float = 0.30
    hallucination_weight: float = 0.30
    similarity_weight: float = 0.20
    consistency_weight: float = 0.20
    
    # ==================== Thresholds ====================
    hallucination_threshold: float = 0.65
    faithfulness_threshold: float = 0.60
    similarity_threshold: float = 0.70
    
    # ==================== Dataset Settings ====================
    truthfulqa_path: str = "data/truthfulqa"
    truthfulqa_split: str = "validation"
    
    # ==================== Paths ====================
    data_dir: str = "data"
    results_dir: str = "results"
    plots_dir: str = "results/plots"
    
    # ==================== Experiment Settings ====================
    num_consistency_runs: int = 3
    random_seed: int = 42
    
    # ==================== Dashboard Settings ====================
    dashboard_port: int = 8501
    api_port: int = 8000

# Create directories
os.makedirs(Config.data_dir, exist_ok=True)
os.makedirs(Config.results_dir, exist_ok=True)
os.makedirs(Config.plots_dir, exist_ok=True)

config = Config()