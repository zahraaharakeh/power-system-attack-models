import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

print("Starting minimal test...")

# Test data loading
try:
    df = pd.read_excel('benign_bus14.xlsx')
    print(f"Data loaded: {df.shape}")
    
    # Extract features
    feature_columns = ['Pd_new', 'Qd_new', 'Vm', 'Va']
    X_benign = df[feature_columns].values
    print(f"Benign samples: {len(X_benign)}")
    
    # Generate simple malicious data
    X_malicious = X_benign + np.random.normal(0, 0.2, X_benign.shape)
    
    # Combine data
    X = np.vstack([X_benign, X_malicious])
    y = np.concatenate([np.zeros(len(X_benign)), np.ones(len(X_malicious))])
    
    print(f"Total samples: {len(X)}")
    print(f"Class distribution: {np.bincount(y.astype(int))}")
    
    # Standardize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Simple split
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"Training set: {len(X_train)}")
    print(f"Test set: {len(X_test)}")
    
    # Create simple model
    class SimpleAutoencoder(nn.Module):
        def __init__(self, input_dim):
            super(SimpleAutoencoder, self).__init__()
            self.encoder = nn.Sequential(
                nn.Linear(input_dim, 64),
                nn.ReLU(),
                nn.Linear(64, 32),
                nn.ReLU()
            )
            self.decoder = nn.Sequential(
                nn.Linear(32, 64),
                nn.ReLU(),
                nn.Linear(64, input_dim)
            )
        
        def forward(self, x):
            encoded = self.encoder(x)
            decoded = self.decoder(encoded)
            return decoded
    
    # Create model and train
    model = SimpleAutoencoder(X.shape[1])
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Create data loader
    train_dataset = TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train))
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    
    print("Training simple model...")
    model.train()
    for epoch in range(5):  # Just 5 epochs for testing
        total_loss = 0
        for batch_x, _ in train_loader:
            optimizer.zero_grad()
            reconstructed = model(batch_x)
            loss = criterion(reconstructed, batch_x)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        if epoch % 2 == 0:
            print(f"Epoch {epoch}, Loss: {total_loss/len(train_loader):.6f}")
    
    print("Minimal test completed successfully!")
    
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc() 