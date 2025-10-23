import torch
import numpy as np
import warnings
warnings.filterwarnings('ignore')

def test_enhanced_model():
    print("1. Testing Enhanced Unsupervised Graph Informer...")
    try:
        from enhanced_unsupervised_graph_informer import (
            EnhancedUnsupervisedGraphInformer,
            create_power_system_graph
        )
        
        model = EnhancedUnsupervisedGraphInformer(
            input_dim=4, d_model=64, n_heads=4, n_layers=2, seq_len=12, num_nodes=14
        )
        
        edge_index = create_power_system_graph(num_nodes=14)
        x = torch.randn(2, 12, 4)
        
        with torch.no_grad():
            outputs = model(x, edge_index, return_detailed_output=True)
        
        print("   ✓ Enhanced model working")
        return True
    except Exception as e:
        print(f"   ✗ Enhanced model failed: {e}")
        return False

def test_online_learning():
    print("2. Testing Online Learning Components...")
    try:
        from online_learning_graph_informer import (
            OnlineLearningGraphInformer,
            OnlineLearningBuffer
        )
        
        model = OnlineLearningGraphInformer(
            input_dim=4, d_model=64, n_heads=4, n_layers=2, seq_len=12, num_nodes=14
        )
        
        buffer = OnlineLearningBuffer(max_size=100)
        buffer.add_sample(torch.randn(12, 4), 0.5)
        
        x = torch.randn(2, 12, 4)
        with torch.no_grad():
            outputs = model(x, return_online_info=True)
        
        print("   ✓ Online learning working")
        return True
    except Exception as e:
        print(f"   ✗ Online learning failed: {e}")
        return False

def test_evaluation_framework():
    print("3. Testing Evaluation Framework...")
    try:
        from comprehensive_evaluation_framework import ComprehensiveEvaluator
        
        evaluator = ComprehensiveEvaluator(device='cpu')
        
        labels = np.array([0, 0, 1, 1, 0, 1])
        scores = np.array([0.1, 0.2, 0.8, 0.9, 0.15, 0.85])
        
        threshold = evaluator._determine_threshold(labels, scores, strategy='optimal')
        predictions = (scores > threshold).astype(int)
        metrics = evaluator._calculate_comprehensive_metrics(labels, predictions, scores)
        
        print("   ✓ Evaluation framework working")
        return True
    except Exception as e:
        print(f"   ✗ Evaluation framework failed: {e}")
        return False

def test_comparative_analysis():
    print("4. Testing Comparative Analysis...")
    try:
        import comparative_analysis
        print("   ✓ Comparative analysis module loaded")
        return True
    except Exception as e:
        print(f"   ✗ Comparative analysis failed: {e}")
        return False

def main():
    print("Enhanced Unsupervised Graph Informer - System Test")
    print("=" * 50)
    
    tests = [
        test_enhanced_model,
        test_online_learning,
        test_evaluation_framework,
        test_comparative_analysis
    ]
    
    results = []
    for test in tests:
        results.append(test())
        print()
    
    print("=" * 50)
    print("SYSTEM TEST SUMMARY")
    print("=" * 50)
    
    passed = sum(results)
    total = len(results)
    
    print(f"Tests passed: {passed}/{total}")
    
    if passed == total:
        print("🎉 ALL SYSTEMS WORKING! The enhanced unsupervised graph informer is ready!")
    else:
        print("⚠️  Some components need attention.")
    
    return passed == total

if __name__ == "__main__":
    main()
