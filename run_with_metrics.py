#!/usr/bin/env python3
"""
Run Enhanced Unsupervised Graph Informer with Performance Metrics
================================================================
"""

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

def run_enhanced_model_with_metrics():
    """Run the enhanced model and show detailed performance metrics."""
    print("Enhanced Unsupervised Graph Informer - Performance Test")
    print("=" * 60)
    
    try:
        from enhanced_unsupervised_graph_informer import (
            EnhancedUnsupervisedGraphInformer,
            load_and_preprocess_enhanced_unsupervised_data,
            create_power_system_graph,
            train_enhanced_unsupervised_model,
            evaluate_enhanced_unsupervised_model
        )
        
        # Load data
        print("Loading and preprocessing data...")
        X_seq, y_seq, edge_index, scaler, feature_names = load_and_preprocess_enhanced_unsupervised_data(
            'benign_bus14.xlsx', seq_len=24, num_nodes=14
        )
        
        print(f"Data loaded: {X_seq.shape[0]} sequences, {X_seq.shape[2]} features")
        print(f"Class distribution: Benign={np.sum(y_seq==0)}, Malicious={np.sum(y_seq==1)}")
        
        # Create model with improved architecture
        print("\nCreating enhanced unsupervised model...")
        model = EnhancedUnsupervisedGraphInformer(
            input_dim=X_seq.shape[2],
            d_model=256,  # Increased for better capacity
            n_heads=8,
            n_layers=4,   # Increased layers
            seq_len=24,
            num_nodes=14,
            use_contrastive=True,
            use_temporal_consistency=True
        )
        
        total_params = sum(p.numel() for p in model.parameters())
        print(f"Model parameters: {total_params:,}")
        
        # Create data loaders
        from torch.utils.data import TensorDataset, DataLoader
        
        # Split data
        from sklearn.model_selection import train_test_split
        X_train, X_val, y_train, y_val = train_test_split(
            X_seq, y_seq, test_size=0.2, random_state=42, stratify=y_seq
        )
        
        # Create datasets
        train_dataset = TensorDataset(torch.FloatTensor(X_train), torch.LongTensor(y_train))
        val_dataset = TensorDataset(torch.FloatTensor(X_val), torch.LongTensor(y_val))
        
        # Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
        
        # Train model with more epochs for better accuracy
        print("\nTraining model (20 epochs for better accuracy)...")
        train_losses, val_losses = train_enhanced_unsupervised_model(
            model, train_loader, val_loader, 
            num_epochs=20,  # More epochs for better learning
            device='cpu'
        )
        
        print(f"Training completed. Final train loss: {train_losses[-1]:.4f}")
        print(f"Final validation loss: {val_losses[-1]:.4f}")
        
        # Create test data loader
        test_dataset = TensorDataset(torch.FloatTensor(X_seq), torch.LongTensor(y_seq))
        test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)
        
        # Evaluate model
        print("\nEvaluating model performance...")
        results = evaluate_enhanced_unsupervised_model(
            model, test_loader, device='cpu'
        )
        
        # Display results
        print("\n" + "=" * 60)
        print("PERFORMANCE METRICS")
        print("=" * 60)
        
        print(f"Accuracy:           {results['accuracy']:.4f} ({results['accuracy']*100:.2f}%)")
        print(f"Precision:          {results['precision']:.4f}")
        print(f"Recall:             {results['recall']:.4f}")
        print(f"F1-Score:           {results['f1_score']:.4f}")
        print(f"ROC-AUC:            {results['roc_auc']:.4f}")
        if 'average_precision' in results:
            print(f"Average Precision:  {results['average_precision']:.4f}")
        else:
            print(f"Average Precision:  N/A")
        
        # Print available keys for debugging
        print(f"\nAvailable result keys: {list(results.keys())}")
        
        if 'confusion_matrix' in results:
            print(f"\nConfusion Matrix:")
            print(f"True Negatives:  {results['confusion_matrix'][0,0]}")
            print(f"False Positives: {results['confusion_matrix'][0,1]}")
            print(f"False Negatives: {results['confusion_matrix'][1,0]}")
            print(f"True Positives:  {results['confusion_matrix'][1,1]}")
        
        if 'detection_rate' in results:
            print(f"\nDetection Rate:     {results['detection_rate']:.4f} ({results['detection_rate']*100:.2f}%)")
        if 'false_alarm_rate' in results:
            print(f"False Alarm Rate:   {results['false_alarm_rate']:.4f} ({results['false_alarm_rate']*100:.2f}%)")
        
        # Show some predictions
        print(f"\nSample Predictions:")
        print(f"Threshold used: {results['threshold']:.4f}")
        
        # Get some sample predictions
        model.eval()
        with torch.no_grad():
            sample_indices = np.random.choice(len(X_seq), 10, replace=False)
            sample_x = torch.FloatTensor(X_seq[sample_indices])
            sample_y = y_seq[sample_indices]
            
            outputs = model(sample_x, edge_index, return_detailed_output=True)
            predictions = (outputs['ensemble_score'].numpy().flatten() > results['threshold']).astype(int)
            
            print(f"\nSample Results (10 random samples):")
            print(f"{'Index':<6} {'True':<6} {'Pred':<6} {'Score':<8} {'Correct':<8}")
            print("-" * 40)
            
            correct = 0
            for i, (true_label, pred_label, score) in enumerate(zip(sample_y, predictions, outputs['ensemble_score'].numpy().flatten())):
                is_correct = "✓" if true_label == pred_label else "✗"
                if true_label == pred_label:
                    correct += 1
                print(f"{sample_indices[i]:<6} {true_label:<6} {pred_label:<6} {score:<8.4f} {is_correct:<8}")
            
            print(f"\nSample Accuracy: {correct/10:.2f} ({correct}/10)")
        
        # Plot training curves
        try:
            plt.figure(figsize=(12, 4))
            
            plt.subplot(1, 2, 1)
            plt.plot(train_losses, label='Training Loss', color='blue')
            plt.plot(val_losses, label='Validation Loss', color='red')
            plt.title('Training Progress')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.legend()
            plt.grid(True)
            
            plt.subplot(1, 2, 2)
            # Use ensemble_scores for ROC curve
            if 'ensemble_scores' in results:
                from sklearn.metrics import roc_curve
                fpr, tpr, _ = roc_curve(y_seq, results['ensemble_scores'])
                plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {results["roc_auc"]:.3f})')
                plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
                plt.xlim([0.0, 1.0])
                plt.ylim([0.0, 1.05])
                plt.xlabel('False Positive Rate')
                plt.ylabel('True Positive Rate')
                plt.title('ROC Curve')
                plt.legend(loc="lower right")
                plt.grid(True)
            
            plt.tight_layout()
            plt.savefig('enhanced_model_performance.png', dpi=150, bbox_inches='tight')
            plt.show()
            
            print(f"\nPerformance plot saved as 'enhanced_model_performance.png'")
        except Exception as e:
            print(f"Could not create plot: {e}")
        
        return results
        
    except Exception as e:
        print(f"Error running model: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    results = run_enhanced_model_with_metrics()
    
    if results:
        print(f"\n🎉 Enhanced Unsupervised Graph Informer completed successfully!")
        print(f"Final Accuracy: {results['accuracy']*100:.2f}%")
    else:
        print(f"\n❌ Model execution failed.")
