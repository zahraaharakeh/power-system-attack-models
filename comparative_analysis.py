import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns
import math
import warnings
from torch_geometric.nn import GCNConv, GATConv
from typing import Dict, List, Tuple, Optional
import logging
import json
import os
from datetime import datetime
import time
from collections import defaultdict
import sys
warnings.filterwarnings('ignore')

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import all the graph informer variants
try:
    from unsupervised_graph_informer import UnsupervisedGraphInformer, load_and_preprocess_unsupervised_data, train_unsupervised_model, evaluate_unsupervised_model
    from graph_informer_transformer import GraphInformerTransformer, load_and_preprocess_graph_data, train_graph_informer_unsupervised, evaluate_graph_informer_unsupervised
    from enhanced_graph_informer import EnhancedGraphInformerTransformer, load_and_preprocess_enhanced_data, train_enhanced_model, evaluate_enhanced_model
    from optimized_graph_informer import OptimizedGraphInformer, load_and_preprocess_optimized_data, train_optimized_model, evaluate_optimized_model
    from adaptive_graph_informer import AdaptiveGraphInformer, load_and_preprocess_adaptive_data, train_adaptive_model, evaluate_adaptive_model
    from advanced_graph_informer import AdvancedGraphInformer, load_and_preprocess_advanced_data, train_advanced_model, evaluate_advanced_model
    from enhanced_unsupervised_graph_informer import EnhancedUnsupervisedGraphInformer, load_and_preprocess_enhanced_unsupervised_data, train_enhanced_unsupervised_model, evaluate_enhanced_unsupervised_model
    from online_learning_graph_informer import OnlineLearningGraphInformer, load_and_preprocess_online_data, train_online_learning_model, evaluate_online_learning_model
    from comprehensive_evaluation_framework import ComprehensiveEvaluator
except ImportError as e:
    logger.warning(f"Some modules could not be imported: {e}")
    logger.info("Continuing with available modules...")

class ComparativeAnalyzer:
    """Comparative analysis of different Graph Informer variants."""
    
    def __init__(self, device='cpu'):
        self.device = device
        self.results = {}
        self.training_times = {}
        self.inference_times = {}
        self.model_sizes = {}
        self.evaluator = ComprehensiveEvaluator(device=device)
        
    def run_comparative_analysis(self, benign_file='benign_bus14.xlsx', 
                                seq_len=24, num_nodes=14, 
                                num_epochs=50, batch_size=32):
        """Run comprehensive comparative analysis."""
        logger.info("Starting comparative analysis of Graph Informer variants...")
        
        # Define models to compare
        models_to_compare = {
            'UnsupervisedGraphInformer': {
                'class': UnsupervisedGraphInformer,
                'load_func': load_and_preprocess_unsupervised_data,
                'train_func': train_unsupervised_model,
                'eval_func': evaluate_unsupervised_model,
                'params': {'input_dim': 4, 'd_model': 256, 'n_heads': 8, 'n_layers': 3, 'seq_len': seq_len, 'num_nodes': num_nodes}
            },
            'GraphInformerTransformer': {
                'class': GraphInformerTransformer,
                'load_func': load_and_preprocess_graph_data,
                'train_func': train_graph_informer_unsupervised,
                'eval_func': evaluate_graph_informer_unsupervised,
                'params': {'input_dim': 4, 'd_model': 256, 'n_heads': 8, 'n_layers': 3, 'seq_len': seq_len, 'num_nodes': num_nodes}
            },
            'EnhancedGraphInformer': {
                'class': EnhancedGraphInformerTransformer,
                'load_func': load_and_preprocess_enhanced_data,
                'train_func': train_enhanced_model,
                'eval_func': evaluate_enhanced_model,
                'params': {'input_dim': 4, 'd_model': 256, 'n_heads': 8, 'n_layers': 3, 'seq_len': seq_len, 'num_nodes': num_nodes}
            },
            'OptimizedGraphInformer': {
                'class': OptimizedGraphInformer,
                'load_func': load_and_preprocess_optimized_data,
                'train_func': train_optimized_model,
                'eval_func': evaluate_optimized_model,
                'params': {'input_dim': 4, 'd_model': 256, 'n_heads': 8, 'n_layers': 3, 'seq_len': seq_len, 'num_nodes': num_nodes}
            },
            'AdaptiveGraphInformer': {
                'class': AdaptiveGraphInformer,
                'load_func': load_and_preprocess_adaptive_data,
                'train_func': train_adaptive_model,
                'eval_func': evaluate_adaptive_model,
                'params': {'input_dim': 4, 'd_model': 256, 'n_heads': 8, 'n_layers': 3, 'seq_len': seq_len, 'num_nodes': num_nodes}
            },
            'AdvancedGraphInformer': {
                'class': AdvancedGraphInformer,
                'load_func': load_and_preprocess_advanced_data,
                'train_func': train_advanced_model,
                'eval_func': evaluate_advanced_model,
                'params': {'input_dim': 4, 'd_model': 256, 'n_heads': 8, 'n_layers': 3, 'seq_len': seq_len, 'num_nodes': num_nodes}
            },
            'EnhancedUnsupervisedGraphInformer': {
                'class': EnhancedUnsupervisedGraphInformer,
                'load_func': load_and_preprocess_enhanced_unsupervised_data,
                'train_func': train_enhanced_unsupervised_model,
                'eval_func': evaluate_enhanced_unsupervised_model,
                'params': {'input_dim': 4, 'd_model': 256, 'n_heads': 8, 'n_layers': 3, 'seq_len': seq_len, 'num_nodes': num_nodes}
            },
            'OnlineLearningGraphInformer': {
                'class': OnlineLearningGraphInformer,
                'load_func': load_and_preprocess_online_data,
                'train_func': train_online_learning_model,
                'eval_func': evaluate_online_learning_model,
                'params': {'input_dim': 4, 'd_model': 256, 'n_heads': 8, 'n_layers': 3, 'seq_len': seq_len, 'num_nodes': num_nodes}
            }
        }
        
        # Filter available models
        available_models = {}
        for name, config in models_to_compare.items():
            try:
                # Test if all required functions are available
                config['class']
                config['load_func']
                config['train_func']
                config['eval_func']
                available_models[name] = config
                logger.info(f"✓ {name} is available")
            except (NameError, AttributeError) as e:
                logger.warning(f"✗ {name} is not available: {e}")
        
        if not available_models:
            logger.error("No models are available for comparison!")
            return {}
        
        logger.info(f"Running comparison on {len(available_models)} models...")
        
        # Run comparison for each available model
        for model_name, config in available_models.items():
            try:
                logger.info(f"\n{'='*60}")
                logger.info(f"Evaluating {model_name}")
                logger.info(f"{'='*60}")
                
                # Load and preprocess data
                start_time = time.time()
                X_seq, y_seq, edge_index, scaler, feature_names = config['load_func'](
                    benign_file, seq_len, num_nodes
                )
                data_load_time = time.time() - start_time
                
                # Split data
                X_train, X_test, y_train, y_test = train_test_split(
                    X_seq, y_seq, test_size=0.2, random_state=42, stratify=y_seq
                )
                X_train, X_val, y_train, y_val = train_test_split(
                    X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
                )
                
                # Convert to PyTorch tensors
                X_train = torch.FloatTensor(X_train)
                y_train = torch.LongTensor(y_train)
                X_val = torch.FloatTensor(X_val)
                y_val = torch.LongTensor(y_val)
                X_test = torch.FloatTensor(X_test)
                y_test = torch.LongTensor(y_test)
                
                # Create data loaders
                from torch.utils.data import TensorDataset, DataLoader
                train_dataset = TensorDataset(X_train, y_train)
                val_dataset = TensorDataset(X_val, y_val)
                test_dataset = TensorDataset(X_test, y_test)
                
                train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
                val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
                test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
                
                # Create model
                model = config['class'](**config['params'])
                model = model.to(self.device)
                
                # Calculate model size
                model_size = sum(p.numel() for p in model.parameters())
                self.model_sizes[model_name] = model_size
                
                logger.info(f"Model parameters: {model_size:,}")
                logger.info(f"Data load time: {data_load_time:.2f}s")
                
                # Train model
                start_time = time.time()
                train_losses, val_losses = config['train_func'](
                    model, train_loader, val_loader, num_epochs=num_epochs, device=self.device
                )
                training_time = time.time() - start_time
                self.training_times[model_name] = training_time
                
                logger.info(f"Training time: {training_time:.2f}s")
                
                # Evaluate model
                start_time = time.time()
                metrics = config['eval_func'](model, test_loader, device=self.device)
                inference_time = time.time() - start_time
                self.inference_times[model_name] = inference_time
                
                logger.info(f"Inference time: {inference_time:.2f}s")
                
                # Store results
                self.results[model_name] = {
                    'metrics': metrics,
                    'training_time': training_time,
                    'inference_time': inference_time,
                    'model_size': model_size,
                    'data_load_time': data_load_time,
                    'train_losses': train_losses,
                    'val_losses': val_losses
                }
                
                # Log key metrics
                logger.info(f"Results for {model_name}:")
                logger.info(f"  Accuracy: {metrics.get('accuracy', 0):.4f}")
                logger.info(f"  Precision: {metrics.get('precision', 0):.4f}")
                logger.info(f"  Recall: {metrics.get('recall', 0):.4f}")
                logger.info(f"  F1-Score: {metrics.get('f1_score', 0):.4f}")
                logger.info(f"  ROC-AUC: {metrics.get('roc_auc', 0):.4f}")
                
            except Exception as e:
                logger.error(f"Error evaluating {model_name}: {e}")
                continue
        
        return self.results
    
    def generate_comparison_report(self, save_path='comparative_analysis_report.json'):
        """Generate comprehensive comparison report."""
        if not self.results:
            logger.error("No results available for comparison report")
            return {}
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'summary': {},
            'detailed_results': {},
            'rankings': {},
            'performance_analysis': {}
        }
        
        # Extract key metrics for comparison
        key_metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc']
        
        # Calculate summary statistics
        summary = {}
        for metric in key_metrics:
            values = []
            for model_name, result in self.results.items():
                if metric in result['metrics']:
                    values.append(result['metrics'][metric])
            
            if values:
                summary[metric] = {
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'min': np.min(values),
                    'max': np.max(values),
                    'best_model': max(self.results.keys(), 
                                    key=lambda x: self.results[x]['metrics'].get(metric, 0))
                }
        
        report['summary'] = summary
        
        # Detailed results
        for model_name, result in self.results.items():
            report['detailed_results'][model_name] = {
                'metrics': result['metrics'],
                'training_time': result['training_time'],
                'inference_time': result['inference_time'],
                'model_size': result['model_size'],
                'data_load_time': result['data_load_time']
            }
        
        # Rankings
        rankings = {}
        for metric in key_metrics:
            if metric in summary:
                model_scores = {}
                for model_name, result in self.results.items():
                    if metric in result['metrics']:
                        model_scores[model_name] = result['metrics'][metric]
                
                sorted_models = sorted(model_scores.items(), key=lambda x: x[1], reverse=True)
                rankings[metric] = [model for model, _ in sorted_models]
        
        report['rankings'] = rankings
        
        # Overall ranking
        overall_ranks = defaultdict(int)
        for metric, ranking in rankings.items():
            for i, model in enumerate(ranking):
                overall_ranks[model] += i + 1
        
        num_metrics = len(rankings)
        for model in overall_ranks:
            overall_ranks[model] /= num_metrics
        
        overall_ranking = sorted(overall_ranks.items(), key=lambda x: x[1])
        report['rankings']['overall'] = overall_ranking
        
        # Performance analysis
        performance_analysis = {
            'fastest_training': min(self.training_times.items(), key=lambda x: x[1]),
            'fastest_inference': min(self.inference_times.items(), key=lambda x: x[1]),
            'smallest_model': min(self.model_sizes.items(), key=lambda x: x[1]),
            'largest_model': max(self.model_sizes.items(), key=lambda x: x[1])
        }
        
        report['performance_analysis'] = performance_analysis
        
        # Save report
        with open(save_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"Comparison report saved to {save_path}")
        return report
    
    def plot_comparison_results(self, save_path='comparison_plots'):
        """Generate comparison plots."""
        if not self.results:
            logger.error("No results available for plotting")
            return
        
        os.makedirs(save_path, exist_ok=True)
        
        # 1. Performance metrics comparison
        self._plot_metrics_comparison(save_path)
        
        # 2. Training curves
        self._plot_training_curves(save_path)
        
        # 3. Performance vs efficiency
        self._plot_performance_efficiency(save_path)
        
        # 4. Model size comparison
        self._plot_model_sizes(save_path)
        
        logger.info(f"Comparison plots saved to {save_path}")
    
    def _plot_metrics_comparison(self, save_path):
        """Plot metrics comparison."""
        key_metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc']
        models = list(self.results.keys())
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()
        
        for i, metric in enumerate(key_metrics):
            if i < len(axes):
                values = []
                model_names = []
                for model_name, result in self.results.items():
                    if metric in result['metrics']:
                        values.append(result['metrics'][metric])
                        model_names.append(model_name)
                
                if values:
                    bars = axes[i].bar(model_names, values, color=plt.cm.viridis(np.linspace(0, 1, len(values))))
                    axes[i].set_title(f'{metric.replace("_", " ").title()}')
                    axes[i].set_ylabel('Score')
                    axes[i].set_ylim(0, 1)
                    
                    # Add value labels on bars
                    for bar, value in zip(bars, values):
                        axes[i].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                                   f'{value:.3f}', ha='center', va='bottom')
                    
                    axes[i].tick_params(axis='x', rotation=45)
        
        # Overall ranking
        if 'rankings' in self.results and 'overall' in self.results['rankings']:
            ranking_data = self.results['rankings']['overall']
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
        plt.savefig(f'{save_path}/metrics_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_training_curves(self, save_path):
        """Plot training curves."""
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # Training losses
        for model_name, result in self.results.items():
            if 'train_losses' in result and result['train_losses']:
                axes[0].plot(result['train_losses'], label=f'{model_name} (Train)', linewidth=2)
            if 'val_losses' in result and result['val_losses']:
                axes[0].plot(result['val_losses'], label=f'{model_name} (Val)', linewidth=2, linestyle='--')
        
        axes[0].set_title('Training Curves')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Training times
        models = list(self.training_times.keys())
        times = list(self.training_times.values())
        
        bars = axes[1].bar(models, times, color=plt.cm.plasma(np.linspace(0, 1, len(models))))
        axes[1].set_title('Training Times')
        axes[1].set_xlabel('Model')
        axes[1].set_ylabel('Time (seconds)')
        
        for bar, time in zip(bars, times):
            axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(times)*0.01, 
                       f'{time:.1f}s', ha='center', va='bottom')
        
        axes[1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(f'{save_path}/training_curves.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_performance_efficiency(self, save_path):
        """Plot performance vs efficiency."""
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # Performance vs Training Time
        accuracies = []
        training_times = []
        model_names = []
        
        for model_name, result in self.results.items():
            if 'accuracy' in result['metrics']:
                accuracies.append(result['metrics']['accuracy'])
                training_times.append(result['training_time'])
                model_names.append(model_name)
        
        scatter = axes[0].scatter(training_times, accuracies, s=100, alpha=0.7, 
                                c=range(len(model_names)), cmap='viridis')
        axes[0].set_xlabel('Training Time (seconds)')
        axes[0].set_ylabel('Accuracy')
        axes[0].set_title('Performance vs Training Time')
        axes[0].grid(True, alpha=0.3)
        
        # Add model labels
        for i, name in enumerate(model_names):
            axes[0].annotate(name, (training_times[i], accuracies[i]), 
                           xytext=(5, 5), textcoords='offset points', fontsize=8)
        
        # Performance vs Model Size
        model_sizes = []
        for model_name in model_names:
            if model_name in self.model_sizes:
                model_sizes.append(self.model_sizes[model_name])
            else:
                model_sizes.append(0)
        
        scatter = axes[1].scatter(model_sizes, accuracies, s=100, alpha=0.7, 
                                c=range(len(model_names)), cmap='viridis')
        axes[1].set_xlabel('Model Size (parameters)')
        axes[1].set_ylabel('Accuracy')
        axes[1].set_title('Performance vs Model Size')
        axes[1].grid(True, alpha=0.3)
        
        # Add model labels
        for i, name in enumerate(model_names):
            axes[1].annotate(name, (model_sizes[i], accuracies[i]), 
                           xytext=(5, 5), textcoords='offset points', fontsize=8)
        
        plt.tight_layout()
        plt.savefig(f'{save_path}/performance_efficiency.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_model_sizes(self, save_path):
        """Plot model sizes comparison."""
        models = list(self.model_sizes.keys())
        sizes = list(self.model_sizes.values())
        
        plt.figure(figsize=(12, 8))
        bars = plt.bar(models, sizes, color=plt.cm.Set3(np.linspace(0, 1, len(models))))
        plt.title('Model Sizes Comparison')
        plt.xlabel('Model')
        plt.ylabel('Number of Parameters')
        plt.yscale('log')
        
        for bar, size in zip(bars, sizes):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() * 1.1, 
                    f'{size:,}', ha='center', va='bottom')
        
        plt.tick_params(axis='x', rotation=45)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f'{save_path}/model_sizes.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def print_summary(self):
        """Print comparison summary."""
        if not self.results:
            print("No comparison results available.")
            return
        
        print("\n" + "="*100)
        print("COMPREHENSIVE GRAPH INFORMER COMPARATIVE ANALYSIS")
        print("="*100)
        
        # Performance table
        key_metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc']
        
        print(f"\n{'Model':<35} {'Accuracy':<10} {'Precision':<10} {'Recall':<10} {'F1-Score':<10} {'ROC-AUC':<10} {'Train Time':<12} {'Model Size':<12}")
        print("-" * 120)
        
        for model_name, result in self.results.items():
            metrics = result['metrics']
            train_time = result['training_time']
            model_size = result['model_size']
            
            print(f"{model_name:<35} {metrics.get('accuracy', 0):<10.4f} {metrics.get('precision', 0):<10.4f} "
                  f"{metrics.get('recall', 0):<10.4f} {metrics.get('f1_score', 0):<10.4f} "
                  f"{metrics.get('roc_auc', 0):<10.4f} {train_time:<12.1f} {model_size:<12,}")
        
        # Best performers
        print(f"\n{'Metric':<15} {'Best Model':<35} {'Score':<10}")
        print("-" * 60)
        
        for metric in key_metrics:
            best_model = max(self.results.keys(), 
                           key=lambda x: self.results[x]['metrics'].get(metric, 0))
            best_score = self.results[best_model]['metrics'].get(metric, 0)
            print(f"{metric:<15} {best_model:<35} {best_score:<10.4f}")
        
        # Efficiency metrics
        print(f"\n{'Efficiency Metric':<20} {'Best Model':<35} {'Value':<15}")
        print("-" * 70)
        
        fastest_training = min(self.training_times.items(), key=lambda x: x[1])
        fastest_inference = min(self.inference_times.items(), key=lambda x: x[1])
        smallest_model = min(self.model_sizes.items(), key=lambda x: x[1])
        
        print(f"{'Fastest Training':<20} {fastest_training[0]:<35} {fastest_training[1]:<15.1f}s")
        print(f"{'Fastest Inference':<20} {fastest_inference[0]:<35} {fastest_inference[1]:<15.1f}s")
        print(f"{'Smallest Model':<20} {smallest_model[0]:<35} {smallest_model[1]:<15,} params")

def main():
    """Main function for comparative analysis."""
    print("🔍 Comprehensive Comparative Analysis of Graph Informer Variants")
    print("=" * 80)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Initialize analyzer
    analyzer = ComparativeAnalyzer(device=device)
    
    # Run comparative analysis
    results = analyzer.run_comparative_analysis(
        benign_file='benign_bus14.xlsx',
        seq_len=24,
        num_nodes=14,
        num_epochs=30,  # Reduced for faster comparison
        batch_size=32
    )
    
    if results:
        # Generate comparison report
        report = analyzer.generate_comparison_report()
        
        # Generate plots
        analyzer.plot_comparison_results()
        
        # Print summary
        analyzer.print_summary()
        
        print(f"\n✅ Comparative analysis complete!")
        print(f"📊 Results saved to comparative_analysis_report.json")
        print(f"📈 Plots saved to comparison_plots/")
        
        return analyzer, report
    else:
        print("❌ No results generated from comparative analysis")
        return None, None

if __name__ == "__main__":
    main()
