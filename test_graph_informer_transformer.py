#!/usr/bin/env python3
"""
Test script for Graph Informer Transformer
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import warnings
warnings.filterwarnings('ignore')

def test_graph_informer_transformer():
    """Test the Graph Informer Transformer model."""
    print("Testing Graph Informer Transformer...")
    
    try:
        # Import the graph informer transformer model
        from graph_informer_transformer import (
            GraphInformerTransformer,
            load_and_preprocess_graph_data,
            train_graph_informer_unsupervised,
            evaluate_graph_informer_unsupervised
        )
        
        # Set device
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {device}")
        
        # Load and preprocess data
        X_seq, y_seq, edge_index, scaler, feature_names = load_and_preprocess_graph_data(
            'benign_bus14.xlsx', seq_len=24, num_nodes=14
        )
        
        print(f"Dataset Statistics:")
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
        
        # Create Graph Informer Transformer model
        model = GraphInformerTransformer(
            input_dim=len(feature_names),
            d_model=256,
            n_heads=8,
            n_layers=3,
            seq_len=24,
            num_nodes=14
        )
        
        print(f"Graph Informer Transformer Model Architecture:")
        print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
        print(f"Model dimension: 256")
        print(f"Transformer layers: 3")
        print(f"Features: Graph-aware attention, temporal modeling, unsupervised learning")
        print(f"Learning: Unsupervised (reconstruction + anomaly detection)")
        
        # Train model with fewer epochs for testing
        print("Training Graph Informer Transformer model...")
        train_losses, val_losses = train_graph_informer_unsupervised(
            model, train_loader, val_loader, num_epochs=20, device=device
        )
        
        # Evaluate model
        print("Evaluating Graph Informer Transformer model...")
        metrics = evaluate_graph_informer_unsupervised(model, test_loader, device=device)
        
        # Print results
        print("Graph Informer Transformer Model Performance:")
        print(f"Accuracy: {metrics['accuracy']:.4f}")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall: {metrics['recall']:.4f}")
        print(f"F1-Score: {metrics['f1_score']:.4f}")
        print(f"ROC-AUC: {metrics['roc_auc']:.4f}")
        print(f"Anomaly Threshold: {metrics['threshold']:.4f}")
        
        # Save model
        torch.save(model.state_dict(), 'test_graph_informer_transformer.pth')
        print("Graph Informer Transformer test complete!")
        print("Model saved to 'test_graph_informer_transformer.pth'")
        
        return model, metrics
        
    except ImportError as e:
        print(f"Could not import Graph Informer Transformer model: {e}")
        return None, None
    except Exception as e:
        print(f"Error running Graph Informer Transformer test: {e}")
        return None, None

if __name__ == "__main__":
    test_graph_informer_transformer()
