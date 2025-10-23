import torch
import numpy as np
import warnings
warnings.filterwarnings('ignore')

def test_model():
    print("Testing Enhanced Unsupervised Graph Informer...")
    
    try:
        from enhanced_unsupervised_graph_informer import (
            EnhancedUnsupervisedGraphInformer,
            create_power_system_graph
        )
        
        # Create model
        model = EnhancedUnsupervisedGraphInformer(
            input_dim=4,
            d_model=128,
            n_heads=4,
            n_layers=2,
            seq_len=12,
            num_nodes=14
        )
        
        # Create graph
        edge_index = create_power_system_graph(num_nodes=14)
        
        # Test forward pass
        x = torch.randn(2, 12, 4)
        
        with torch.no_grad():
            outputs = model(x, edge_index, return_detailed_output=True)
        
        print("Model test successful!")
        print(f"Reconstructed global shape: {outputs['reconstructed_global'].shape}")
        print(f"Ensemble score shape: {outputs['ensemble_score'].shape}")
        
        return True
        
    except Exception as e:
        print(f"Model test failed: {e}")
        return False

if __name__ == "__main__":
    test_model()