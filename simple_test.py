import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, TensorDataset

# Simple attention layer
class SimpleAttentionLayer(nn.Module):
    def __init__(self, input_dim, attention_dim):
        super(SimpleAttentionLayer, self).__init__()
        self.attention_dim = attention_dim
        self.query = nn.Linear(input_dim, attention_dim)
        self.key = nn.Linear(input_dim, attention_dim)
        self.value = nn.Linear(input_dim, attention_dim)
        self.softmax = nn.Softmax(dim=-1)
        
    def forward(self, x):
        batch_size, seq_len, input_dim = x.size()
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)
        attention_scores = torch.bmm(Q, K.transpose(1, 2)) / np.sqrt(self.attention_dim)
        attention_weights = self.softmax(attention_scores)
        attended_output = torch.bmm(attention_weights, V)
        return attended_output, attention_weights

# Simple autoencoder
class SimpleAutoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim, attention_dim=32):
        super(SimpleAutoencoder, self).__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        # Attention
        self.attention = SimpleAttentionLayer(hidden_dim, attention_dim)
        
        # Latent
        self.latent_layer = nn.Sequential(
            nn.Linear(attention_dim, latent_dim),
            nn.ReLU()
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, input_dim)
        )
        
    def encode(self, x):
        encoded = self.encoder(x)
        batch_size = encoded.size(0)
        encoded = encoded.unsqueeze(1)  # Add sequence dimension
        attended, attention_weights = self.attention(encoded)
        attended = attended.squeeze(1)  # Remove sequence dimension
        latent = self.latent_layer(attended)
        return latent, attention_weights
    
    def decode(self, latent):
        return self.decoder(latent)
    
    def forward(self, x):
        latent, attention_weights = self.encode(x)
        reconstructed = self.decode(latent)
        return reconstructed, latent, attention_weights

print("Testing simple autoencoder...")

try:
    # Create model
    model = SimpleAutoencoder(input_dim=4, hidden_dim=64, latent_dim=16, attention_dim=32)
    print("Model created successfully!")
    
    # Create dummy data
    X = np.random.randn(100, 4)
    y = np.random.randint(0, 2, 100)
    
    # Create data loader
    dataset = TensorDataset(torch.FloatTensor(X), torch.FloatTensor(y))
    loader = DataLoader(dataset, batch_size=16, shuffle=True)
    
    print("Data loader created successfully!")
    
    # Test forward pass
    device = torch.device('cpu')
    model.to(device)
    
    for batch_x, batch_y in loader:
        batch_x = batch_x.to(device)
        reconstructed, latent, attention_weights = model(batch_x)
        print(f"Input shape: {batch_x.shape}")
        print(f"Reconstructed shape: {reconstructed.shape}")
        print(f"Latent shape: {latent.shape}")
        print(f"Attention weights shape: {attention_weights.shape}")
        break
    
    print("Forward pass successful!")
    
    # Test training step
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    model.train()
    for batch_x, batch_y in loader:
        batch_x = batch_x.to(device)
        
        optimizer.zero_grad()
        reconstructed, latent, attention_weights = model(batch_x)
        loss = criterion(reconstructed, batch_x)
        loss.backward()
        optimizer.step()
        
        print(f"Training loss: {loss.item():.6f}")
        break
    
    print("Training step successful!")
    print("All tests passed!")
    
except Exception as e:
    print(f"Error occurred: {e}")
    import traceback
    traceback.print_exc() 