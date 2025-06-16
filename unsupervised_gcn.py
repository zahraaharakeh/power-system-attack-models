import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.data import Data, DataLoader
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from torch.optim.lr_scheduler import ReduceLROnPlateau

class UnsupervisedGCN(nn.Module):
    def __init__(self, num_features, hidden_channels=64):
        super(UnsupervisedGCN, self).__init__()
        
        # Encoder
        self.encoder_conv1 = GCNConv(num_features, hidden_channels)
        self.encoder_conv2 = GCNConv(hidden_channels, hidden_channels // 2)
        self.encoder_conv3 = GCNConv(hidden_channels // 2, hidden_channels // 4)
        
        # Decoder
        self.decoder_conv1 = GCNConv(hidden_channels // 4, hidden_channels // 2)
        self.decoder_conv2 = GCNConv(hidden_channels // 2, hidden_channels)
        self.decoder_conv3 = GCNConv(hidden_channels, num_features)
        
        # Final reconstruction layer
        self.final_layer = nn.Linear(num_features, num_features)
    
    def forward(self, x, edge_index, batch):
        # Encoder
        x = self.encoder_conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.2, training=self.training)
        
        x = self.encoder_conv2(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.2, training=self.training)
        
        x = self.encoder_conv3(x, edge_index)
        x = F.relu(x)
        
        # Decoder
        x = self.decoder_conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.2, training=self.training)
        
        x = self.decoder_conv2(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.2, training=self.training)
        
        x = self.decoder_conv3(x, edge_index)
        x = F.relu(x)
        
        # Final reconstruction
        x = self.final_layer(x)
        return x

def create_graph_data(features):
    """Create PyTorch Geometric Data objects from features."""
    data_list = []
    for i in range(len(features)):
        x = torch.FloatTensor(features[i]).view(1, -1)
        edge_idx = torch.LongTensor([[0], [0]])  # Self-loop
        data = Data(x=x, edge_index=edge_idx)
        data_list.append(data)
    return data_list

def load_and_preprocess_data(file_path):
    """Load and preprocess the data."""
    df = pd.read_excel(file_path)
    feature_columns = ['Pd_new', 'Qd_new', 'Vm', 'Va']
    X = df[feature_columns].values
    
    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    return X_scaled, scaler

def train_model(model, train_loader, val_loader, num_epochs=10000, device='cuda'):
    """Train the unsupervised GCN model."""
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)
    
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    patience = 50  # Increased patience for better training
    patience_counter = 0
    min_epochs = 1000  # Minimum number of epochs to train
    
    print("\nStarting training...")
    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            out = model(batch.x, batch.edge_index, batch.batch)
            loss = criterion(out, batch.x)  # Reconstruction loss
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                out = model(batch.x, batch.edge_index, batch.batch)
                loss = criterion(out, batch.x)
                val_loss += loss.item()
        
        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        
        # Learning rate scheduling
        scheduler.step(val_loss)
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), 'best_unsupervised_gcn.pth')
        else:
            patience_counter += 1
        
        if (epoch + 1) % 100 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}')
        
        # Only consider early stopping after minimum epochs
        if epoch >= min_epochs and patience_counter >= patience:
            print(f'Early stopping at epoch {epoch+1}')
            break
    
    return train_losses, val_losses

def evaluate_model(model, test_loader, device='cuda'):
    """Evaluate the model and calculate reconstruction error."""
    model.eval()
    reconstruction_errors = []
    
    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            out = model(batch.x, batch.edge_index, batch.batch)
            error = F.mse_loss(out, batch.x, reduction='none').mean(dim=1)
            reconstruction_errors.extend(error.cpu().numpy())
    
    return np.array(reconstruction_errors)

def plot_results(train_losses, val_losses, reconstruction_errors):
    """Plot training history and reconstruction error distribution."""
    plt.figure(figsize=(15, 5))
    
    # Plot training history
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training History')
    plt.legend()
    
    # Plot reconstruction error distribution
    plt.subplot(1, 2, 2)
    sns.histplot(reconstruction_errors, bins=50)
    plt.xlabel('Reconstruction Error')
    plt.ylabel('Count')
    plt.title('Reconstruction Error Distribution')
    
    plt.tight_layout()
    plt.savefig('unsupervised_gcn_results.png')
    plt.close()

def main():
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load and preprocess data
    X, scaler = load_and_preprocess_data('benign_bus14.xlsx')
    
    # Print dataset statistics
    print(f"\nDataset Statistics:")
    print(f"Total samples: {len(X)}")
    print(f"Feature shape: {X.shape}")
    
    # Split data
    X_train, X_test = train_test_split(X, test_size=0.2, random_state=42)
    
    # Create graph data
    train_data = create_graph_data(X_train)
    test_data = create_graph_data(X_test)
    
    # Create data loaders
    train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=32, shuffle=False)
    
    # Create and train model
    model = UnsupervisedGCN(num_features=4).to(device)
    train_losses, val_losses = train_model(model, train_loader, test_loader, device=device)
    
    # Evaluate model
    reconstruction_errors = evaluate_model(model, test_loader, device=device)
    
    # Calculate anomaly threshold (using 95th percentile)
    threshold = np.percentile(reconstruction_errors, 95)
    print(f"\nAnomaly Detection Results:")
    print(f"Reconstruction Error Threshold: {threshold:.6f}")
    print(f"  - This is the error value above which samples are considered anomalous")
    print(f"  - Calculated as the 95th percentile of reconstruction errors")
    print(f"  - Lower values indicate better reconstruction of normal data")
    
    num_anomalies = np.sum(reconstruction_errors > threshold)
    print(f"\nNumber of anomalies detected: {num_anomalies}")
    print(f"  - Number of samples with reconstruction error above the threshold")
    print(f"  - These samples show significant deviation from normal patterns")
    
    detection_rate = np.mean(reconstruction_errors > threshold)
    print(f"\nAnomaly detection rate: {detection_rate:.4f}")
    print(f"  - Proportion of samples classified as anomalous")
    print(f"  - Expected to be around 0.05 (5%) when using 95th percentile threshold")
    
    print(f"\nDetailed Error Statistics:")
    print(f"Mean reconstruction error: {np.mean(reconstruction_errors):.6f}")
    print(f"  - Average reconstruction error across all samples")
    print(f"  - Lower values indicate better overall reconstruction")
    
    print(f"Std reconstruction error: {np.std(reconstruction_errors):.6f}")
    print(f"  - Standard deviation of reconstruction errors")
    print(f"  - Higher values indicate more variability in reconstruction quality")
    
    # Plot results
    plot_results(train_losses, val_losses, reconstruction_errors)
    
    # Save results with detailed explanations
    with open('unsupervised_gcn_results.txt', 'w', encoding='utf-8') as f:
        f.write("Unsupervised GCN Model Results\n")
        f.write("=============================\n\n")
        f.write("Dataset Information:\n")
        f.write(f"Total samples: {len(X)}\n")
        f.write(f"Training samples: {len(X_train)}\n")
        f.write(f"Test samples: {len(X_test)}\n\n")
        
        f.write("Model Configuration:\n")
        f.write("- Architecture: GCN Autoencoder\n")
        f.write("- Hidden dimensions: 64 -> 32 -> 16 -> 32 -> 64\n")
        f.write("- Activation: ReLU\n")
        f.write("- Dropout: 0.2\n\n")
        
        f.write("Training Details:\n")
        f.write(f"Final training loss: {train_losses[-1]:.6f}\n")
        f.write(f"Final validation loss: {val_losses[-1]:.6f}\n")
        f.write(f"Number of epochs trained: {len(train_losses)}\n\n")
        
        f.write("Anomaly Detection Results:\n")
        f.write(f"Reconstruction Error Threshold: {threshold:.6f}\n")
        f.write("  - 95th percentile of reconstruction errors\n")
        f.write("  - Samples with errors above this are considered anomalous\n\n")
        
        f.write(f"Number of anomalies detected: {num_anomalies}\n")
        f.write("  - Count of samples classified as anomalous\n\n")
        
        f.write(f"Anomaly detection rate: {detection_rate:.4f}\n")
        f.write("  - Proportion of samples classified as anomalous\n")
        f.write("  - Expected to be around 0.05 (5%) with 95th percentile threshold\n\n")
        
        f.write("Error Statistics:\n")
        f.write(f"Mean reconstruction error: {np.mean(reconstruction_errors):.6f}\n")
        f.write("  - Average reconstruction error across all samples\n\n")
        
        f.write(f"Std reconstruction error: {np.std(reconstruction_errors):.6f}\n")
        f.write("  - Standard deviation of reconstruction errors\n\n")
        
        f.write("Interpretation:\n")
        f.write("1. The model learns to reconstruct normal patterns in the data\n")
        f.write("2. Anomalies are detected when reconstruction error is high\n")
        f.write("3. The threshold is set to identify the top 5% of errors as anomalies\n")
        f.write("4. Lower mean reconstruction error indicates better model performance\n")

if __name__ == "__main__":
    main() 