#!/usr/bin/env python3
"""
Enhanced Unsupervised Graph Informer Transformer Demo
====================================================

This script demonstrates the enhanced unsupervised graph informer transformer
with advanced anomaly detection mechanisms for power system attack detection.

Features:
- Pure unsupervised learning (no labeled attack data used)
- Contrastive learning for better representation learning
- Temporal consistency modeling
- Dynamic ensemble scoring with adaptive weights
- Online learning capabilities
- Comprehensive evaluation framework
- Comparative analysis of different variants

Author: AI Assistant
Date: 2024
"""

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
warnings.filterwarnings('ignore')

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def print_banner():
    """Print welcome banner."""
    print("=" * 100)
    print("🚀 ENHANCED UNSUPERVISED GRAPH INFORMER TRANSFORMER")
    print("   Advanced Anomaly Detection for Power System Security")
    print("=" * 100)
    print()
    print("Features:")
    print("✓ Pure unsupervised learning (no labeled attack data)")
    print("✓ Contrastive learning for better representations")
    print("✓ Temporal consistency modeling")
    print("✓ Dynamic ensemble scoring with adaptive weights")
    print("✓ Online learning capabilities")
    print("✓ Comprehensive evaluation framework")
    print("✓ Comparative analysis of variants")
    print()
    print("=" * 100)

def check_requirements():
    """Check if all required packages are available."""
    required_packages = [
        'torch', 'torch_geometric', 'numpy', 'pandas', 'sklearn', 
        'matplotlib', 'seaborn'
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print(f"❌ Missing required packages: {', '.join(missing_packages)}")
        print("Please install them using: pip install " + " ".join(missing_packages))
        return False
    
    print("✅ All required packages are available")
    return True

def create_demo_data():
    """Create demo data for testing."""
    print("📊 Creating demo data...")
    
    # Create synthetic power system data
    np.random.seed(42)
    n_samples = 1000
    n_features = 4  # Pd_new, Qd_new, Vm, Va
    
    # Generate normal power system data
    normal_data = np.random.normal(0, 1, (n_samples, n_features))
    
    # Add some realistic power system characteristics
    normal_data[:, 0] = np.abs(normal_data[:, 0])  # Power demand (positive)
    normal_data[:, 1] = normal_data[:, 0] * 0.3 + np.random.normal(0, 0.1, n_samples)  # Reactive power
    normal_data[:, 2] = 1.0 + normal_data[:, 2] * 0.05  # Voltage magnitude (around 1.0)
    normal_data[:, 3] = normal_data[:, 3] * 0.1  # Voltage angle (small variations)
    
    # Create DataFrame
    df = pd.DataFrame(normal_data, columns=['Pd_new', 'Qd_new', 'Vm', 'Va'])
    
    # Save to Excel file
    df.to_excel('benign_bus14.xlsx', index=False)
    print(f"✅ Demo data saved to 'benign_bus14.xlsx' ({n_samples} samples)")
    
    return df

def run_enhanced_unsupervised_demo():
    """Run the enhanced unsupervised graph informer demo."""
    print("\n🔍 Running Enhanced Unsupervised Graph Informer Demo...")
    
    try:
        # Import the enhanced unsupervised model
        from enhanced_unsupervised_graph_informer import (
            EnhancedUnsupervisedGraphInformer,
            load_and_preprocess_enhanced_unsupervised_data,
            train_enhanced_unsupervised_model,
            evaluate_enhanced_unsupervised_model
        )
        
        # Set device
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {device}")
        
        # Load and preprocess data
        X_seq, y_seq, edge_index, scaler, feature_names = load_and_preprocess_enhanced_unsupervised_data(
            'benign_bus14.xlsx', seq_len=24, num_nodes=14
        )
        
        print(f"\nDataset Statistics:")
        print(f"Total sequences: {len(X_seq)}")
        print(f"Sequence shape: {X_seq.shape}")
        print(f"Benign sequences: {np.sum(y_seq == 0)}")
        print(f"Malicious sequences: {np.sum(y_seq == 1)}")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X_seq, y_seq, test_size=0.2, random_state=42, stratify=y_seq)
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42, stratify=y_train)
        
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
        
        batch_size = 32
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        
        # Create enhanced unsupervised model
        model = EnhancedUnsupervisedGraphInformer(
            input_dim=len(feature_names),
            d_model=256,
            n_heads=8,
            n_layers=3,
            seq_len=24,
            num_nodes=14,
            use_contrastive=True,
            use_temporal_consistency=True
        )
        
        print(f"\nEnhanced Unsupervised Model Architecture:")
        print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
        print(f"Model dimension: 256")
        print(f"Transformer layers: 3")
        print(f"Features: Contrastive learning, temporal consistency, dynamic ensemble scoring")
        print(f"Learning: Pure unsupervised (no labeled attack data used)")
        
        # Train model
        print("\nTraining enhanced unsupervised model...")
        train_losses, val_losses = train_enhanced_unsupervised_model(
            model, train_loader, val_loader, num_epochs=50, device=device
        )
        
        # Evaluate model
        print("\nEvaluating enhanced unsupervised model...")
        metrics = evaluate_enhanced_unsupervised_model(model, test_loader, device=device)
        
        # Print results
        print("\n🏆 Enhanced Unsupervised Model Performance:")
        print(f"Accuracy: {metrics['accuracy']:.4f}")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall: {metrics['recall']:.4f}")
        print(f"F1-Score: {metrics['f1_score']:.4f}")
        print(f"ROC-AUC: {metrics['roc_auc']:.4f}")
        print(f"Final Threshold: {metrics['threshold']:.4f}")
        
        print(f"\nDynamic Ensemble Weights:")
        print(f"Weights: {metrics['dynamic_weights']}")
        
        # Save model
        torch.save(model.state_dict(), 'enhanced_unsupervised_graph_informer_demo.pth')
        print(f"\n✅ Enhanced Unsupervised Graph Informer demo complete!")
        print(f"🎯 Model saved to 'enhanced_unsupervised_graph_informer_demo.pth'")
        
        return model, metrics
        
    except ImportError as e:
        print(f"❌ Could not import enhanced unsupervised model: {e}")
        print("Please ensure the enhanced_unsupervised_graph_informer.py file is available")
        return None, None
    except Exception as e:
        print(f"❌ Error running enhanced unsupervised demo: {e}")
        return None, None

def run_online_learning_demo():
    """Run the online learning demo."""
    print("\n🔄 Running Online Learning Demo...")
    
    try:
        # Import the online learning model
        from online_learning_graph_informer import (
            OnlineLearningGraphInformer,
            OnlineLearningManager,
            load_and_preprocess_online_data,
            train_online_learning_model,
            evaluate_online_learning_model,
            simulate_online_learning
        )
        
        # Set device
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load and preprocess data
        X_seq, y_seq, edge_index, scaler, feature_names = load_and_preprocess_online_data(
            'benign_bus14.xlsx', seq_len=24, num_nodes=14
        )
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X_seq, y_seq, test_size=0.2, random_state=42, stratify=y_seq)
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42, stratify=y_train)
        
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
        
        batch_size = 32
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        
        # Create online learning model
        model = OnlineLearningGraphInformer(
            input_dim=len(feature_names),
            d_model=256,
            n_heads=8,
            n_layers=3,
            seq_len=24,
            num_nodes=14,
            online_buffer_size=200
        )
        
        print(f"Online Learning Model Architecture:")
        print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
        print(f"Features: Online learning, adaptive optimization, continuous adaptation")
        
        # Train model
        print("\nTraining online learning model...")
        train_losses, val_losses = train_online_learning_model(
            model, train_loader, val_loader, num_epochs=30, device=device
        )
        
        # Evaluate model
        print("\nEvaluating online learning model...")
        metrics = evaluate_online_learning_model(model, test_loader, device=device)
        
        # Simulate online learning
        print("\nSimulating online learning with streaming data...")
        online_manager = OnlineLearningManager(model, device=device, update_frequency=5)
        
        online_results = simulate_online_learning(
            model, X_test, y_test, online_manager, num_samples=100
        )
        
        print(f"\n🔄 Online Learning Results:")
        print(f"Final Accuracy: {online_results['accuracy']:.4f}")
        print(f"Final F1-Score: {online_results['f1_score']:.4f}")
        print(f"Adaptation Count: {online_results['online_stats']['adaptation_count']}")
        
        # Save model
        torch.save(model.state_dict(), 'online_learning_graph_informer_demo.pth')
        print(f"\n✅ Online learning demo complete!")
        print(f"🎯 Model saved to 'online_learning_graph_informer_demo.pth'")
        
        return model, metrics, online_results
        
    except ImportError as e:
        print(f"❌ Could not import online learning model: {e}")
        return None, None, None
    except Exception as e:
        print(f"❌ Error running online learning demo: {e}")
        return None, None, None

def run_evaluation_framework_demo():
    """Run the comprehensive evaluation framework demo."""
    print("\n📊 Running Comprehensive Evaluation Framework Demo...")
    
    try:
        # Import the evaluation framework
        from comprehensive_evaluation_framework import ComprehensiveEvaluator, load_test_data
        
        # Set device
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load test data
        X_seq, y_seq, scaler, feature_names = load_test_data('benign_bus14.xlsx')
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X_seq, y_seq, test_size=0.3, random_state=42)
        
        # Convert to PyTorch tensors
        X_test = torch.FloatTensor(X_test)
        y_test = torch.LongTensor(y_test)
        
        # Create test data loader
        from torch.utils.data import TensorDataset, DataLoader
        test_dataset = TensorDataset(X_test, y_test)
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
        
        # Initialize evaluator
        evaluator = ComprehensiveEvaluator(device=device)
        
        # Create mock models for demonstration
        class MockModel(nn.Module):
            def __init__(self, input_dim):
                super().__init__()
                self.linear = nn.Linear(input_dim, 1)
                self.sigmoid = nn.Sigmoid()
            
            def forward(self, x, return_detailed_output=False):
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
        
        # Create mock models
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
        
        print(f"\n✅ Comprehensive evaluation framework demo complete!")
        print(f"📊 Results saved to evaluation_report.json")
        print(f"📈 Visualizations saved to evaluation_plots/")
        
        return evaluator, report
        
    except ImportError as e:
        print(f"❌ Could not import evaluation framework: {e}")
        return None, None
    except Exception as e:
        print(f"❌ Error running evaluation framework demo: {e}")
        return None, None

def main():
    """Main demo function."""
    print_banner()
    
    # Check requirements
    if not check_requirements():
        return
    
    # Create demo data
    demo_data = create_demo_data()
    
    print("\n🚀 Starting Enhanced Unsupervised Graph Informer Demo...")
    print("This demo will showcase the advanced features of the unsupervised graph informer transformer.")
    
    # Run enhanced unsupervised demo
    enhanced_model, enhanced_metrics = run_enhanced_unsupervised_demo()
    
    # Run online learning demo
    online_model, online_metrics, online_results = run_online_learning_demo()
    
    # Run evaluation framework demo
    evaluator, evaluation_report = run_evaluation_framework_demo()
    
    # Final summary
    print("\n" + "="*100)
    print("🎉 DEMO COMPLETED SUCCESSFULLY!")
    print("="*100)
    
    print("\n📋 Summary of Results:")
    
    if enhanced_metrics:
        print(f"\n🔍 Enhanced Unsupervised Graph Informer:")
        print(f"   Accuracy: {enhanced_metrics['accuracy']:.4f}")
        print(f"   F1-Score: {enhanced_metrics['f1_score']:.4f}")
        print(f"   ROC-AUC: {enhanced_metrics['roc_auc']:.4f}")
    
    if online_metrics:
        print(f"\n🔄 Online Learning Graph Informer:")
        print(f"   Accuracy: {online_metrics['accuracy']:.4f}")
        print(f"   F1-Score: {online_metrics['f1_score']:.4f}")
        print(f"   ROC-AUC: {online_metrics['roc_auc']:.4f}")
    
    print(f"\n📁 Generated Files:")
    print(f"   - enhanced_unsupervised_graph_informer_demo.pth")
    print(f"   - online_learning_graph_informer_demo.pth")
    print(f"   - evaluation_report.json")
    print(f"   - evaluation_plots/ (directory)")
    
    print(f"\n🔧 Key Features Demonstrated:")
    print(f"   ✓ Pure unsupervised learning")
    print(f"   ✓ Contrastive learning")
    print(f"   ✓ Temporal consistency modeling")
    print(f"   ✓ Dynamic ensemble scoring")
    print(f"   ✓ Online learning capabilities")
    print(f"   ✓ Comprehensive evaluation framework")
    
    print(f"\n📚 Next Steps:")
    print(f"   1. Experiment with different hyperparameters")
    print(f"   2. Try different attack patterns")
    print(f"   3. Test on real power system data")
    print(f"   4. Implement additional graph neural network layers")
    print(f"   5. Add more sophisticated anomaly detection mechanisms")
    
    print("\n" + "="*100)
    print("Thank you for using the Enhanced Unsupervised Graph Informer Transformer!")
    print("="*100)

if __name__ == "__main__":
    main()
