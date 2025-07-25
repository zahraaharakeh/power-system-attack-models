import torch
import torch.nn as nn
import numpy as np
from attention_autoencoder import AttentionAutoencoder

print("Testing AttentionAutoencoder model...")

try:
    # Create a simple test model
    input_dim = 4
    hidden_dims = [128, 64]
    latent_dim = 32
    attention_dim = 64
    
    model = AttentionAutoencoder(
        input_dim=input_dim,
        hidden_dims=hidden_dims,
        latent_dim=latent_dim,
        attention_dim=attention_dim
    )
    
    print(f"Model created successfully!")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Test forward pass
    batch_size = 16
    x = torch.randn(batch_size, input_dim)
    
    print(f"Input shape: {x.shape}")
    
    # Forward pass
    reconstructed, latent, attention_weights = model(x)
    
    print(f"Reconstructed shape: {reconstructed.shape}")
    print(f"Latent shape: {latent.shape}")
    print(f"Attention weights shape: {attention_weights.shape}")
    
    print("Forward pass successful!")
    print("Model test passed!")
    
except Exception as e:
    print(f"Error occurred: {e}")
    import traceback
    traceback.print_exc() 