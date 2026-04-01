# experiments/advanced_metrics.py
"""
Advanced Metrics - F1 Score, ROC Curve, Threshold Analysis
"""

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, roc_curve, auc, confusion_matrix
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from typing import Dict, Tuple

class AdvancedMetrics:
    """
    Compute advanced evaluation metrics.
    """
    
    def __init__(self):
        self.results = None
    
    def compute_f1(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
        """Compute F1 score and related metrics."""
        f1 = f1_score(y_true, y_pred, average='binary')
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        
        return {
            "f1_score": f1,
            "precision": precision,
            "recall": recall,
            "specificity": specificity,
            "true_positives": tp,
            "false_positives": fp,
            "true_negatives": tn,
            "false_negatives": fn
        }
    
    def compute_roc(self, y_true: np.ndarray, y_scores: np.ndarray) -> Dict:
        """Compute ROC curve and AUC."""
        fpr, tpr, thresholds = roc_curve(y_true, y_scores)
        roc_auc = auc(fpr, tpr)
        
        # Find optimal threshold (Youden's J statistic)
        j_scores = tpr - fpr
        optimal_idx = np.argmax(j_scores)
        optimal_threshold = thresholds[optimal_idx]
        
        return {
            "fpr": fpr,
            "tpr": tpr,
            "auc": roc_auc,
            "optimal_threshold": optimal_threshold,
            "thresholds": thresholds
        }
    
    def threshold_analysis(self, y_true: np.ndarray, y_scores: np.ndarray, thresholds: np.ndarray) -> pd.DataFrame:
        """Analyze performance at different thresholds."""
        results = []
        
        for threshold in thresholds:
            y_pred = (y_scores > threshold).astype(int)
            metrics = self.compute_f1(y_true, y_pred)
            results.append({
                "threshold": threshold,
                "f1_score": metrics["f1_score"],
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "specificity": metrics["specificity"]
            })
        
        return pd.DataFrame(results)
    
    def plot_roc_curve(self, fpr, tpr, auc_score: float, save_path: str = None):
        """Plot ROC curve."""
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=fpr, y=tpr,
            mode='lines',
            name=f'ROC Curve (AUC = {auc_score:.3f})',
            line=dict(color='blue', width=2)
        ))
        
        fig.add_trace(go.Scatter(
            x=[0, 1], y=[0, 1],
            mode='lines',
            name='Random Classifier',
            line=dict(color='red', width=1, dash='dash')
        ))
        
        fig.update_layout(
            title='ROC Curve',
            xaxis_title='False Positive Rate',
            yaxis_title='True Positive Rate',
            width=500,
            height=500
        )
        
        if save_path:
            fig.write_image(save_path)
        
        return fig
    
    def print_metrics_report(self, metrics: Dict):
        """Print formatted metrics report."""
        print("\n" + "="*60)
        print("📊 ADVANCED METRICS REPORT")
        print("="*60)
        print(f"\nF1 Score: {metrics['f1_score']:.3f}")
        print(f"Precision: {metrics['precision']:.3f}")
        print(f"Recall: {metrics['recall']:.3f}")
        print(f"Specificity: {metrics['specificity']:.3f}")
        print(f"\nConfusion Matrix:")
        print(f"  True Positives: {metrics['true_positives']}")
        print(f"  False Positives: {metrics['false_positives']}")
        print(f"  True Negatives: {metrics['true_negatives']}")
        print(f"  False Negatives: {metrics['false_negatives']}")