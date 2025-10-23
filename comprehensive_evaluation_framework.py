import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, 
                           roc_auc_score, confusion_matrix, classification_report,
                           precision_recall_curve, roc_curve, average_precision_score)
import matplotlib.pyplot as plt
import seaborn as sns
import math
import warnings
from torch_geometric.nn import GCNConv, GATConv
from typing import Dict, List, Tuple, Optional, Any
import logging
import json
import os
from datetime import datetime
import time
from collections import defaultdict
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
warnings.filterwarnings('ignore')

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ComprehensiveEvaluator:
    """Comprehensive evaluation framework for Graph Informer models."""
    
    def __init__(self, device='cpu'):
        self.device = device
        self.evaluation_results = {}
        self.model_comparisons = {}
        self.metric_history = defaultdict(list)
        
    def evaluate_model(self, model, test_loader, model_name="Model", 
                      return_detailed_output=False, threshold_strategy='optimal'):
        """Comprehensive evaluation of a single model."""
        logger.info(f"Evaluating {model_name}...")
        
        model.eval()
        all_outputs = []
        all_labels = []
        all_predictions = []
        all_anomaly_scores = []
        all_reconstruction_errors = []
        
        inference_times = []
        
        with torch.no_grad():
            for batch_x, batch_y in test_loader:
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                
                # Measure inference time
                start_time = time.time()
                
                if hasattr(model, 'forward') and 'return_detailed_output' in model.forward.__code__.co_varnames:
                    outputs = model(batch_x, return_detailed_output=True)
                else:
                    # Fallback for models without detailed output
                    outputs = model(batch_x)
                    if isinstance(outputs, tuple):
                        outputs = {
                            'reconstructed_global': outputs[0],
                            'reconstructed_local': outputs[1] if len(outputs) > 1 else outputs[0],
                            'ensemble_score': outputs[2] if len(outputs) > 2 else outputs[0],
                            'attention_weights': outputs[3] if len(outputs) > 3 else None,
                            'global_features': outputs[4] if len(outputs) > 4 else outputs[0].mean(dim=1),
                            'local_features': outputs[5] if len(outputs) > 5 else outputs[0].max(dim=1)[0]
                        }
                
                inference_time = time.time() - start_time
                inference_times.append(inference_time)
                
                all_outputs.append(outputs)
                all_labels.extend(batch_y.cpu().numpy())
        
        all_labels = np.array(all_labels)
        
        # Extract outputs
        if 'ensemble_score' in all_outputs[0]:
            all_anomaly_scores = torch.cat([out['ensemble_score'] for out in all_outputs]).cpu().numpy().flatten()
        else:
            # Fallback: use reconstruction error as anomaly score
            test_data = test_loader.dataset.tensors[0].numpy()
            reconstructed_global = torch.cat([out['reconstructed_global'] for out in all_outputs]).cpu().numpy()
            all_anomaly_scores = np.mean((reconstructed_global - test_data.mean(axis=1))**2, axis=1)
        
        # Calculate reconstruction errors
        test_data = test_loader.dataset.tensors[0].numpy()
        if 'reconstructed_global' in all_outputs[0]:
            reconstructed_global = torch.cat([out['reconstructed_global'] for out in all_outputs]).cpu().numpy()
            reconstructed_local = torch.cat([out['reconstructed_local'] for out in all_outputs]).cpu().numpy()
            
            reconstruction_error_global = np.mean((reconstructed_global - test_data.mean(axis=1))**2, axis=1)
            reconstruction_error_local = np.mean((reconstructed_local - test_data.max(axis=1))**2, axis=1)
            all_reconstruction_errors = [reconstruction_error_global, reconstruction_error_local]
        
        # Determine optimal threshold
        threshold = self._determine_threshold(all_labels, all_anomaly_scores, strategy=threshold_strategy)
        
        # Make predictions
        all_predictions = (all_anomaly_scores > threshold).astype(int)
        
        # Calculate comprehensive metrics
        metrics = self._calculate_comprehensive_metrics(all_labels, all_predictions, all_anomaly_scores)
        
        # Add performance metrics
        metrics['inference_time'] = {
            'mean': np.mean(inference_times),
            'std': np.std(inference_times),
            'min': np.min(inference_times),
            'max': np.max(inference_times)
        }
        
        metrics['threshold'] = threshold
        metrics['threshold_strategy'] = threshold_strategy
        
        # Store results
        self.evaluation_results[model_name] = {
            'metrics': metrics,
            'predictions': all_predictions,
            'labels': all_labels,
            'anomaly_scores': all_anomaly_scores,
            'reconstruction_errors': all_reconstruction_errors,
            'outputs': all_outputs if return_detailed_output else None
        }
        
        logger.info(f"Evaluation of {model_name} completed")
        return metrics
    
    def _determine_threshold(self, labels, scores, strategy='optimal'):
        """Determine optimal threshold using various strategies."""
        if strategy == 'optimal':
            # Youden's J statistic
            fpr, tpr, thresholds = roc_curve(labels, scores)
            youden_j = tpr - fpr
            optimal_idx = np.argmax(youden_j)
            return thresholds[optimal_idx]
        
        elif strategy == 'percentile':
            # 95th percentile of normal data
            normal_scores = scores[labels == 0]
            return np.percentile(normal_scores, 95)
        
        elif strategy == 'mean_std':
            # Mean + 2*std of normal data
            normal_scores = scores[labels == 0]
            return np.mean(normal_scores) + 2 * np.std(normal_scores)
        
        elif strategy == 'median':
            # Median of all scores
            return np.median(scores)
        
        else:
            return 0.5  # Default threshold
    
    def _calculate_comprehensive_metrics(self, labels, predictions, scores):
        """Calculate comprehensive evaluation metrics."""
        metrics = {}
        
        # Basic classification metrics
        metrics['accuracy'] = accuracy_score(labels, predictions)
        metrics['precision'] = precision_score(labels, predictions, average='weighted', zero_division=0)
        metrics['recall'] = recall_score(labels, predictions, average='weighted', zero_division=0)
        metrics['f1_score'] = f1_score(labels, predictions, average='weighted', zero_division=0)
        
        # ROC and PR metrics
        metrics['roc_auc'] = roc_auc_score(labels, scores)
        metrics['average_precision'] = average_precision_score(labels, scores)
        
        # Confusion matrix
        cm = confusion_matrix(labels, predictions)
        metrics['confusion_matrix'] = cm.tolist()
        
        # Additional metrics
        tn, fp, fn, tp = cm.ravel()
        metrics['true_positive_rate'] = tp / (tp + fn) if (tp + fn) > 0 else 0
        metrics['false_positive_rate'] = fp / (fp + tn) if (fp + tn) > 0 else 0
        metrics['true_negative_rate'] = tn / (tn + fp) if (tn + fp) > 0 else 0
        metrics['false_negative_rate'] = fn / (fn + tp) if (fn + tp) > 0 else 0
        
        # Precision and recall for each class
        precision_per_class = precision_score(labels, predictions, average=None, zero_division=0)
        recall_per_class = recall_score(labels, predictions, average=None, zero_division=0)
        f1_per_class = f1_score(labels, predictions, average=None, zero_division=0)
        
        metrics['precision_per_class'] = precision_per_class.tolist()
        metrics['recall_per_class'] = recall_per_class.tolist()
        metrics['f1_per_class'] = f1_per_class.tolist()
        
        # Class distribution
        metrics['class_distribution'] = {
            'total': len(labels),
            'normal': np.sum(labels == 0),
            'anomaly': np.sum(labels == 1),
            'normal_ratio': np.mean(labels == 0),
            'anomaly_ratio': np.mean(labels == 1)
        }
        
        return metrics
    
    def compare_models(self, model_names=None):
        """Compare multiple models."""
        if model_names is None:
            model_names = list(self.evaluation_results.keys())
        
        comparison = {}
        
        # Extract key metrics for comparison
        key_metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc', 'average_precision']
        
        for metric in key_metrics:
            comparison[metric] = {}
            for model_name in model_names:
                if model_name in self.evaluation_results:
                    comparison[metric][model_name] = self.evaluation_results[model_name]['metrics'][metric]
        
        # Calculate rankings
        rankings = {}
        for metric in key_metrics:
            if metric in comparison:
                sorted_models = sorted(comparison[metric].items(), key=lambda x: x[1], reverse=True)
                rankings[metric] = [model for model, _ in sorted_models]
        
        # Overall ranking (average rank across all metrics)
        overall_ranks = defaultdict(int)
        for metric, ranking in rankings.items():
            for i, model in enumerate(ranking):
                overall_ranks[model] += i + 1
        
        # Normalize by number of metrics
        num_metrics = len(rankings)
        for model in overall_ranks:
            overall_ranks[model] /= num_metrics
        
        overall_ranking = sorted(overall_ranks.items(), key=lambda x: x[1])
        
        comparison['rankings'] = rankings
        comparison['overall_ranking'] = overall_ranking
        
        self.model_comparisons = comparison
        return comparison
    
    def generate_visualizations(self, save_path='evaluation_plots'):
        """Generate comprehensive visualizations."""
        os.makedirs(save_path, exist_ok=True)
        
        # 1. Model comparison bar chart
        self._plot_model_comparison(save_path)
        
        # 2. ROC curves
        self._plot_roc_curves(save_path)
        
        # 3. Precision-Recall curves
        self._plot_precision_recall_curves(save_path)
        
        # 4. Confusion matrices
        self._plot_confusion_matrices(save_path)
        
        # 5. Anomaly score distributions
        self._plot_anomaly_score_distributions(save_path)
        
        # 6. Performance over time (if available)
        self._plot_performance_over_time(save_path)
        
        logger.info(f"Visualizations saved to {save_path}")
    
    def _plot_model_comparison(self, save_path):
        """Plot model comparison bar chart."""
        if not self.model_comparisons:
            self.compare_models()
        
        key_metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc']
        models = list(self.evaluation_results.keys())
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()
        
        for i, metric in enumerate(key_metrics):
            if i < len(axes):
                values = [self.evaluation_results[model]['metrics'][metric] for model in models]
                bars = axes[i].bar(models, values, color=plt.cm.viridis(np.linspace(0, 1, len(models))))
                axes[i].set_title(f'{metric.replace("_", " ").title()}')
                axes[i].set_ylabel('Score')
                axes[i].set_ylim(0, 1)
                
                # Add value labels on bars
                for bar, value in zip(bars, values):
                    axes[i].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                               f'{value:.3f}', ha='center', va='bottom')
                
                axes[i].tick_params(axis='x', rotation=45)
        
        # Overall ranking
        if 'overall_ranking' in self.model_comparisons:
            ranking_data = self.model_comparisons['overall_ranking']
            models_rank = [item[0] for item in ranking_data]
            ranks = [item[1] for item in ranking_data]
            
            bars = axes[-1].bar(models_rank, ranks, color=plt.cm.RdYlGn_r(np.linspace(0, 1, len(models_rank))))
            axes[-1].set_title('Overall Ranking (Lower is Better)')
            axes[-1].set_ylabel('Average Rank')
            
            for bar, rank in zip(bars, ranks):
                axes[-1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                            f'{rank:.2f}', ha='center', va='bottom')
            
            axes[-1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(f'{save_path}/model_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_roc_curves(self, save_path):
        """Plot ROC curves for all models."""
        plt.figure(figsize=(10, 8))
        
        for model_name, results in self.evaluation_results.items():
            labels = results['labels']
            scores = results['anomaly_scores']
            
            fpr, tpr, _ = roc_curve(labels, scores)
            auc = results['metrics']['roc_auc']
            
            plt.plot(fpr, tpr, label=f'{model_name} (AUC = {auc:.3f})', linewidth=2)
        
        plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curves Comparison')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(f'{save_path}/roc_curves.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_precision_recall_curves(self, save_path):
        """Plot Precision-Recall curves for all models."""
        plt.figure(figsize=(10, 8))
        
        for model_name, results in self.evaluation_results.items():
            labels = results['labels']
            scores = results['anomaly_scores']
            
            precision, recall, _ = precision_recall_curve(labels, scores)
            ap = results['metrics']['average_precision']
            
            plt.plot(recall, precision, label=f'{model_name} (AP = {ap:.3f})', linewidth=2)
        
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curves Comparison')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(f'{save_path}/precision_recall_curves.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_confusion_matrices(self, save_path):
        """Plot confusion matrices for all models."""
        n_models = len(self.evaluation_results)
        cols = min(3, n_models)
        rows = (n_models + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 4*rows))
        if n_models == 1:
            axes = [axes]
        elif rows == 1:
            axes = axes.reshape(1, -1)
        
        for i, (model_name, results) in enumerate(self.evaluation_results.items()):
            row = i // cols
            col = i % cols
            
            cm = np.array(results['metrics']['confusion_matrix'])
            
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                       xticklabels=['Normal', 'Anomaly'],
                       yticklabels=['Normal', 'Anomaly'],
                       ax=axes[row, col] if rows > 1 else axes[col])
            
            axes[row, col].set_title(f'{model_name} Confusion Matrix') if rows > 1 else axes[col].set_title(f'{model_name} Confusion Matrix')
            axes[row, col].set_xlabel('Predicted') if rows > 1 else axes[col].set_xlabel('Predicted')
            axes[row, col].set_ylabel('Actual') if rows > 1 else axes[col].set_ylabel('Actual')
        
        # Hide empty subplots
        for i in range(n_models, rows * cols):
            row = i // cols
            col = i % cols
            axes[row, col].set_visible(False)
        
        plt.tight_layout()
        plt.savefig(f'{save_path}/confusion_matrices.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_anomaly_score_distributions(self, save_path):
        """Plot anomaly score distributions for all models."""
        n_models = len(self.evaluation_results)
        cols = min(2, n_models)
        rows = (n_models + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(8*cols, 6*rows))
        if n_models == 1:
            axes = [axes]
        elif rows == 1:
            axes = axes.reshape(1, -1)
        
        for i, (model_name, results) in enumerate(self.evaluation_results.items()):
            row = i // cols
            col = i % cols
            
            labels = results['labels']
            scores = results['anomaly_scores']
            threshold = results['metrics']['threshold']
            
            normal_scores = scores[labels == 0]
            anomaly_scores = scores[labels == 1]
            
            ax = axes[row, col] if rows > 1 else axes[col]
            ax.hist(normal_scores, bins=50, alpha=0.7, label='Normal', color='blue', density=True)
            ax.hist(anomaly_scores, bins=50, alpha=0.7, label='Anomaly', color='red', density=True)
            ax.axvline(threshold, color='green', linestyle='--', label=f'Threshold: {threshold:.3f}')
            
            ax.set_title(f'{model_name} Anomaly Score Distribution')
            ax.set_xlabel('Anomaly Score')
            ax.set_ylabel('Density')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        # Hide empty subplots
        for i in range(n_models, rows * cols):
            row = i // cols
            col = i % cols
            axes[row, col].set_visible(False)
        
        plt.tight_layout()
        plt.savefig(f'{save_path}/anomaly_score_distributions.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_performance_over_time(self, save_path):
        """Plot performance over time if available."""
        if not self.metric_history:
            return
        
        plt.figure(figsize=(12, 8))
        
        for model_name, history in self.metric_history.items():
            if history:
                epochs = range(len(history))
                plt.plot(epochs, history, label=model_name, linewidth=2)
        
        plt.xlabel('Epoch')
        plt.ylabel('Performance Metric')
        plt.title('Performance Over Time')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(f'{save_path}/performance_over_time.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def generate_report(self, save_path='evaluation_report.json'):
        """Generate comprehensive evaluation report."""
        report = {
            'timestamp': datetime.now().isoformat(),
            'evaluation_results': {},
            'model_comparisons': self.model_comparisons,
            'summary': {}
        }
        
        # Add evaluation results
        for model_name, results in self.evaluation_results.items():
            report['evaluation_results'][model_name] = {
                'metrics': results['metrics'],
                'threshold': results['metrics']['threshold'],
                'threshold_strategy': results['metrics']['threshold_strategy']
            }
        
        # Add summary statistics
        if self.evaluation_results:
            key_metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc']
            summary = {}
            
            for metric in key_metrics:
                values = [results['metrics'][metric] for results in self.evaluation_results.values()]
                summary[metric] = {
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'min': np.min(values),
                    'max': np.max(values),
                    'best_model': max(self.evaluation_results.keys(), 
                                    key=lambda x: self.evaluation_results[x]['metrics'][metric])
                }
            
            report['summary'] = summary
        
        # Save report
        with open(save_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"Evaluation report saved to {save_path}")
        return report
    
    def print_summary(self):
        """Print evaluation summary."""
        if not self.evaluation_results:
            print("No evaluation results available.")
            return
        
        print("\n" + "="*80)
        print("COMPREHENSIVE EVALUATION SUMMARY")
        print("="*80)
        
        # Model comparison table
        key_metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc']
        
        print(f"\n{'Model':<30} {'Accuracy':<10} {'Precision':<10} {'Recall':<10} {'F1-Score':<10} {'ROC-AUC':<10}")
        print("-" * 80)
        
        for model_name, results in self.evaluation_results.items():
            metrics = results['metrics']
            print(f"{model_name:<30} {metrics['accuracy']:<10.4f} {metrics['precision']:<10.4f} "
                  f"{metrics['recall']:<10.4f} {metrics['f1_score']:<10.4f} {metrics['roc_auc']:<10.4f}")
        
        # Best performing model for each metric
        print(f"\n{'Metric':<15} {'Best Model':<30} {'Score':<10}")
        print("-" * 55)
        
        for metric in key_metrics:
            best_model = max(self.evaluation_results.keys(), 
                           key=lambda x: self.evaluation_results[x]['metrics'][metric])
            best_score = self.evaluation_results[best_model]['metrics'][metric]
            print(f"{metric:<15} {best_model:<30} {best_score:<10.4f}")
        
        # Overall ranking
        if self.model_comparisons and 'overall_ranking' in self.model_comparisons:
            print(f"\nOverall Ranking (Average Rank):")
            print("-" * 40)
            for i, (model, rank) in enumerate(self.model_comparisons['overall_ranking']):
                print(f"{i+1}. {model:<30} {rank:.2f}")

def create_power_system_graph(num_nodes=14):
    """Create IEEE 14-bus system topology."""
    edges = [
        (0, 1), (0, 2), (0, 3), (1, 2), (1, 4), (2, 3), (2, 4), (2, 5),
        (3, 5), (4, 5), (4, 6), (4, 7), (5, 6), (6, 7), (6, 8), (6, 9),
        (6, 10), (7, 8), (8, 9), (9, 10), (9, 11), (9, 12), (10, 11),
        (10, 12), (11, 12), (12, 13)
    ]
    
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)
    
    return edge_index

def load_test_data(benign_file, seq_len=24, num_nodes=14):
    """Load test data for evaluation."""
    print("Loading test data for evaluation...")
    
    # Load benign data
    benign_df = pd.read_excel(benign_file)
    feature_columns = ['Pd_new', 'Qd_new', 'Vm', 'Va']
    X_benign = benign_df[feature_columns].values
    
    # Generate test malicious data
    n_samples = len(X_benign)
    X_malicious = np.random.normal(0, 0.1, (n_samples, len(feature_columns))) + X_benign
    
    # Create labels
    y_benign = np.zeros(len(X_benign))
    y_malicious = np.ones(len(X_malicious))
    
    # Combine data
    X = np.vstack([X_benign, X_malicious])
    y = np.concatenate([y_benign, y_malicious])
    
    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Create sequence data
    sequences = []
    sequence_labels = []
    
    for i in range(len(X_scaled) - seq_len + 1):
        seq = X_scaled[i:i + seq_len]
        label = y[i + seq_len - 1]
        sequences.append(seq)
        sequence_labels.append(label)
    
    X_seq = np.array(sequences)
    y_seq = np.array(sequence_labels)
    
    print(f"Test data created: {X_seq.shape}")
    print(f"Class distribution: Normal={np.sum(y_seq==0)}, Anomaly={np.sum(y_seq==1)}")
    
    return X_seq, y_seq, scaler, feature_columns

def main():
    """Main function for comprehensive evaluation framework."""
    print("🔍 Comprehensive Evaluation Framework for Graph Informer Models")
    print("=" * 80)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load test data
    X_seq, y_seq, scaler, feature_names = load_test_data('benign_bus14.xlsx')
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X_seq, y_seq, test_size=0.3, random_state=42, stratify=y_seq)
    
    # Convert to PyTorch tensors
    X_test = torch.FloatTensor(X_test)
    y_test = torch.LongTensor(y_test)
    
    # Create test data loader
    from torch.utils.data import TensorDataset, DataLoader
    test_dataset = TensorDataset(X_test, y_test)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    # Initialize evaluator
    evaluator = ComprehensiveEvaluator(device=device)
    
    # Example: Evaluate different models (you would load your actual models here)
    print("\nNote: This is a demonstration of the evaluation framework.")
    print("In practice, you would load your trained models and evaluate them.")
    
    # Create a simple mock model for demonstration
    class MockModel(nn.Module):
        def __init__(self, input_dim):
            super().__init__()
            self.linear = nn.Linear(input_dim, 1)
            self.sigmoid = nn.Sigmoid()
        
        def forward(self, x, return_detailed_output=False):
            # Simple mock: use mean of input as features
            features = x.mean(dim=1)
            score = self.sigmoid(self.linear(features))
            
            if return_detailed_output:
                return {
                    'reconstructed_global': x.mean(dim=1),
                    'reconstructed_local': x.max(dim=1)[0],
                    'ensemble_score': score,
                    'attention_weights': None,
                    'global_features': features,
                    'local_features': x.max(dim=1)[0]
                }
            else:
                return x.mean(dim=1), x.max(dim=1)[0], score, None, features, x.max(dim=1)[0]
    
    # Create mock models for demonstration
    models = {
        'Mock Model 1': MockModel(len(feature_names)),
        'Mock Model 2': MockModel(len(feature_names)),
        'Mock Model 3': MockModel(len(feature_names))
    }
    
    # Evaluate models
    for model_name, model in models.items():
        model = model.to(device)
        metrics = evaluator.evaluate_model(model, test_loader, model_name)
        print(f"\n{model_name} Results:")
        print(f"Accuracy: {metrics['accuracy']:.4f}")
        print(f"F1-Score: {metrics['f1_score']:.4f}")
        print(f"ROC-AUC: {metrics['roc_auc']:.4f}")
    
    # Compare models
    comparison = evaluator.compare_models()
    
    # Generate visualizations
    evaluator.generate_visualizations()
    
    # Generate report
    report = evaluator.generate_report()
    
    # Print summary
    evaluator.print_summary()
    
    print(f"\n✅ Comprehensive evaluation framework demonstration complete!")
    print(f"📊 Results saved to evaluation_report.json")
    print(f"📈 Visualizations saved to evaluation_plots/")
    
    return evaluator, report

if __name__ == "__main__":
    main()
