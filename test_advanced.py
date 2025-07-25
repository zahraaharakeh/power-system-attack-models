#!/usr/bin/env python3
"""
Simple test script for the advanced attention autoencoder
"""

import sys
import traceback

try:
    print("Testing import...")
    import attention_autoencoder_advanced
    print("✓ Import successful")
    
    print("Testing model creation...")
    model = attention_autoencoder_advanced.AdvancedAttentionAutoencoder(
        input_dim=4,
        hidden_dims=[256, 128, 64],
        latent_dim=32,
        attention_dim=64,
        num_heads=4
    )
    print("✓ Model creation successful")
    
    print("Testing forward pass...")
    import torch
    x = torch.randn(10, 4)  # 10 samples, 4 features
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