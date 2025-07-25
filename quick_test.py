#!/usr/bin/env python3
"""
Quick test for the simple advanced attention autoencoder
"""

import sys
import traceback

try:
    print("Testing simple advanced attention autoencoder...")
    
    # Import the module
    import simple_advanced_attention
    
    # Test model creation
    model = simple_advanced_attention.SimpleAdvancedAttentionAutoencoder(
        input_dim=4,
        hidden_dims=[256, 128, 64],
        latent_dim=32,
        attention_dim=64
    )
    print("✓ Model creation successful")
    
    # Test forward pass
    import torch
    x = torch.randn(10, 4)
    reconstructed, latent, attention_weights = model(x)
    print("✓ Forward pass successful")
    print(f"Input shape: {x.shape}")
    print(f"Reconstructed shape: {reconstructed.shape}")
    print(f"Latent shape: {latent.shape}")
    print(f"Attention weights shape: {attention_weights.shape}")
    
    print("✓ All tests passed!")
    
except Exception as e:
    print(f"✗ Error: {e}")
    print("Traceback:")
    traceback.print_exc()
    sys.exit(1) 