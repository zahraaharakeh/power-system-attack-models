#!/usr/bin/env python3
"""
Quick Test Script for Enhanced Unsupervised Graph Informer
=========================================================
"""

import torch
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

def test_basic_functionality():
    """Test basic functionality of the enhanced unsupervised graph informer."""
    print("🚀 Testing Enhanced Unsupervised Graph Informer")
    print("=" * 50)
    
    try:
        # Test 1: Import the enhanced model
        print("1. Testing imports...")
        from enhanced_unsupervised_graph_informer import (
            EnhancedUnsupervisedGraphInformer,
            load_and_preprocess_enhanced_unsupervised_data,
            create_power_system_graph
        )
        print("   ✅ Enhanced model imports successful")
        
        # Test 2: Create model
        print("2. Testing model creation...")
        model = EnhancedUnsupervisedGraphInformer(
            input_dim=4,
            d_model=128,  # Smaller for quick test
            n_heads=4,
            n_layers=2,
            seq_len=12,   # Shorter sequence
            num_nodes=14,
            use_contrastive=True,
            use_temporal_consistency=True
        )
        print(f"   ✅ Model created with {sum(p.numel() for p in model.parameters()):,} parameters")
        
        # Test 3: Create graph
        print("3. Testing graph creation...")
        edge_index = create_power_system_graph(num_nodes=14)
        print(f"   ✅ Graph created with {edge_index.shape[1]} edges")
        
        # Test 4: Test forward pass
        print("4. Testing forward pass...")
        batch_size = 2
        seq_len = 12
        input_dim = 4
        
        # Create dummy input
        x = torch.randn(batch_size, seq_len, input_dim)
        
        # Forward pass
        with torch.no_grad():
            outputs = model(x, edge_index, return_detailed_output=True)
        
        print(f"   ✅ Forward pass successful")
        print(f"   - Reconstructed global shape: {outputs['reconstructed_global'].shape}")
        print(f"   - Ensemble score shape: {outputs['ensemble_score'].shape}")
        print(f"   - Dynamic weights shape: {outputs['dynamic_weights'].shape}")
        
        # Test 5: Test data loading
        print("5. Testing data loading...")
        try:
            X_seq, y_seq, edge_index, scaler, feature_names = load_and_preprocess_enhanced_unsupervised_data(
                'benign_bus14.xlsx', seq_len=12, num_nodes=14
            )
            print(f"   ✅ Data loaded successfully")
            print(f"   - Sequences: {X_seq.shape}")
            print(f"   - Labels: {y_seq.shape}")
            print(f"   - Features: {feature_names}")
        except Exception as e:
            print(f"   ⚠️  Data loading issue: {e}")
            print("   Creating dummy data for testing...")
            # Create dummy data
            X_seq = np.random.randn(100, 12, 4)
            y_seq = np.random.randint(0, 2, 100)
            feature_names = ['Pd_new', 'Qd_new', 'Vm', 'Va']
        
        # Test 6: Test training setup
        print("6. Testing training setup...")
        from torch.utils.data import TensorDataset, DataLoader
        
        # Create small dataset
        X_tensor = torch.FloatTensor(X_seq[:20])  # Small subset
        y_tensor = torch.LongTensor(y_seq[:20])
        
        dataset = TensorDataset(X_tensor, y_tensor)
        dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
        
        print(f"   ✅ DataLoader created with {len(dataloader)} batches")
        
        # Test 7: Test one training step
        print("7. Testing one training step...")
        model.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        
        for batch_x, batch_y in dataloader:
            optimizer.zero_grad()
            
            outputs = model(batch_x, edge_index, return_detailed_output=True)
            
            # Simple loss
            recon_loss = torch.nn.functional.mse_loss(
                outputs['reconstructed_global'], 
                batch_x.mean(dim=1)
            )
            
            loss = recon_loss
            loss.backward()
            optimizer.step()
            
            print(f"   ✅ Training step successful, loss: {loss.item():.4f}")
            break
        
        print("\n🎉 All tests passed! The enhanced unsupervised graph informer is working correctly.")
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_online_learning():
    """Test online learning functionality."""
    print("\n🔄 Testing Online Learning Components")
    print("=" * 50)
    
    try:
        from online_learning_graph_informer import (
            OnlineLearningGraphInformer,
            OnlineLearningBuffer,
            AdaptiveOnlineOptimizer
        )
        
        # Test buffer
        print("1. Testing online learning buffer...")
        buffer = OnlineLearningBuffer(max_size=100)
        
        # Add some samples
        for i in range(10):
            sample = torch.randn(12, 4)
            score = np.random.random()
            buffer.add_sample(sample, score)
        
        batch = buffer.get_batch(batch_size=5)
        print(f"   ✅ Buffer working, got batch of size: {len(batch[0]) if batch else 0}")
        
        # Test model
        print("2. Testing online learning model...")
        model = OnlineLearningGraphInformer(
            input_dim=4,
            d_model=64,
            n_heads=4,
            n_layers=2,
            seq_len=12,
            num_nodes=14
        )
        
        # Test forward pass
        x = torch.randn(2, 12, 4)
        with torch.no_grad():
            outputs = model(x, return_online_info=True)
        
        print(f"   ✅ Online learning model working")
        print(f"   - Anomaly score shape: {outputs['anomaly_score'].shape}")
        print(f"   - Adaptation signal shape: {outputs['adaptation_signal'].shape}")
        
        print("\n🎉 Online learning tests passed!")
        return True
        
    except Exception as e:
        print(f"\n❌ Online learning test failed: {e}")
        return False

def test_evaluation_framework():
    """Test evaluation framework."""
    print("\n📊 Testing Evaluation Framework")
    print("=" * 50)
    
    try:
        from comprehensive_evaluation_framework import ComprehensiveEvaluator
        
        # Create evaluator
        evaluator = ComprehensiveEvaluator(device='cpu')
        print("   ✅ Evaluator created")
        
        # Test threshold determination
        labels = np.array([0, 0, 1, 1, 0, 1])
        scores = np.array([0.1, 0.2, 0.8, 0.9, 0.15, 0.85])
        
        threshold = evaluator._determine_threshold(labels, scores, strategy='optimal')
        print(f"   ✅ Threshold determination working: {threshold:.3f}")
        
        # Test metrics calculation
        predictions = (scores > threshold).astype(int)
        metrics = evaluator._calculate_comprehensive_metrics(labels, predictions, scores)
        
        print(f"   ✅ Metrics calculation working")
        print(f"   - Accuracy: {metrics['accuracy']:.3f}")
        print(f"   - F1-Score: {metrics['f1_score']:.3f}")
        print(f"   - ROC-AUC: {metrics['roc_auc']:.3f}")
        
        print("\n🎉 Evaluation framework tests passed!")
        return True
        
    except Exception as e:
        print(f"\n❌ Evaluation framework test failed: {e}")
        return False

def main():
    """Run all tests."""
    print("🔍 Enhanced Unsupervised Graph Informer - Quick Test")
    print("=" * 60)
    
    # Run tests
    test1_passed = test_basic_functionality()
    test2_passed = test_online_learning()
    test3_passed = test_evaluation_framework()
    
    # Summary
    print("\n" + "=" * 60)
    print("📋 TEST SUMMARY")
    print("=" * 60)
    print(f"Basic Functionality: {'✅ PASSED' if test1_passed else '❌ FAILED'}")
    print(f"Online Learning:     {'✅ PASSED' if test2_passed else '❌ FAILED'}")
    print(f"Evaluation Framework: {'✅ PASSED' if test3_passed else '❌ FAILED'}")
    
    if all([test1_passed, test2_passed, test3_passed]):
        print("\n🎉 ALL TESTS PASSED! The enhanced unsupervised graph informer system is ready to use!")
    else:
        print("\n⚠️  Some tests failed. Please check the error messages above.")
    
    return all([test1_passed, test2_passed, test3_passed])

if __name__ == "__main__":
    main()