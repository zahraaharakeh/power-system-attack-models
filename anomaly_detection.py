import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)

class ConvAutoencoder(nn.Module):
    def __init__(self):
        super(ConvAutoencoder, self).__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=2, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, padding=1),
            nn.Conv2d(16, 8, kernel_size=2, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, padding=1)
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(8, 16, kernel_size=2, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 1, kernel_size=2, stride=2, padding=1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

def load_and_preprocess_data(file_path):
    """Load and preprocess the data."""
    # Load data
    df = pd.read_excel(file_path)
    
    # Extract features
    X = df[['Pd_new', 'Qd_new', 'Vm', 'Va']].values
    
    # Standardize the data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Reshape for CNN (2x2x1)
    X_reshaped = X_scaled.reshape(-1, 1, 2, 2)
    
    return X_reshaped, scaler

def train_model(model, train_loader, val_loader, num_epochs=10000, device='cuda'):
    """Train the autoencoder model."""
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=50)
    
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    patience = 200
    patience_counter = 0
    current_lr = optimizer.param_groups[0]['lr']
    
    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0
        for batch in train_loader:
            data = batch[0].to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, data)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                data = batch[0].to(device)
                output = model(data)
                loss = criterion(output, data)
                val_loss += loss.item()
        
        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        
        # Learning rate scheduling
        old_lr = optimizer.param_groups[0]['lr']
        scheduler.step(val_loss)
        new_lr = optimizer.param_groups[0]['lr']
        
        # Print learning rate change
        if new_lr != old_lr:
            print(f'Learning rate decreased from {old_lr:.6f} to {new_lr:.6f}')
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            # Save best model
            torch.save(model.state_dict(), 'best_model.pth')
        else:
            patience_counter += 1
        
        if (epoch + 1) % 100 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}, LR: {new_lr:.6f}')
        
        if patience_counter >= patience:
            print(f'Early stopping at epoch {epoch+1}')
            break
    
    # Load best model
    model.load_state_dict(torch.load('best_model.pth'))
    return train_losses, val_losses

def calculate_metrics(reconstruction_errors, threshold):
    """Calculate various metrics based on reconstruction errors."""
    # Convert reconstruction errors to binary predictions (0: normal, 1: anomaly)
    predictions = (reconstruction_errors > threshold).astype(int)
    
    # Since we're using only benign data, all true labels are 0 (normal)
    true_labels = np.zeros_like(predictions)
    
    # Calculate metrics
    accuracy = accuracy_score(true_labels, predictions)
    precision = precision_score(true_labels, predictions, zero_division=0)
    recall = recall_score(true_labels, predictions, zero_division=0)
    f1 = f1_score(true_labels, predictions, zero_division=0)
    
    # Calculate confusion matrix
    cm = confusion_matrix(true_labels, predictions)
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'confusion_matrix': cm
    }

def plot_results(original, reconstructed, reconstruction_errors, metrics):
    """Plot original vs reconstructed values, error distribution, and metrics."""
    plt.figure(figsize=(15, 5))
    
    # Plot original vs reconstructed values
    plt.subplot(1, 3, 1)
    plt.scatter(original.flatten(), reconstructed.flatten(), alpha=0.5)
    plt.plot([-3, 3], [-3, 3], 'r--')  # Perfect reconstruction line
    plt.xlabel('Original Values')
    plt.ylabel('Reconstructed Values')
    plt.title('Original vs Reconstructed Values')

    # Plot reconstruction error distribution
    plt.subplot(1, 3, 2)
    plt.hist(reconstruction_errors, bins=50)
    plt.xlabel('Reconstruction Error')
    plt.ylabel('Frequency')
    plt.title('Reconstruction Error Distribution')
    
    # Plot confusion matrix
    plt.subplot(1, 3, 3)
    sns.heatmap(metrics['confusion_matrix'], annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    
    plt.tight_layout()
    plt.savefig('reconstruction_results.png')
    plt.close()
    
    # Save metrics to file
    with open('model_metrics.txt', 'w') as f:
        f.write(f"Model Performance Metrics:\n")
        f.write(f"Accuracy: {metrics['accuracy']:.4f}\n")
        f.write(f"Precision: {metrics['precision']:.4f}\n")
        f.write(f"Recall: {metrics['recall']:.4f}\n")
        f.write(f"F1 Score: {metrics['f1_score']:.4f}\n")

def main():
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load and preprocess data
    X_reshaped, scaler = load_and_preprocess_data('benign_bus14.xlsx')
    
    # Convert to PyTorch tensors
    X_tensor = torch.FloatTensor(X_reshaped)
    
    # Split data into train and test sets
    X_train, X_test = train_test_split(X_tensor, test_size=0.2, random_state=42)
    
    # Create data loaders
    train_dataset = TensorDataset(X_train)
    test_dataset = TensorDataset(X_test)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    # Create and train the autoencoder
    model = ConvAutoencoder().to(device)
    train_losses, val_losses = train_model(model, train_loader, test_loader, device=device)
    
    # Get reconstructions
    model.eval()
    reconstructed = []
    with torch.no_grad():
        for batch in test_loader:
            data = batch[0].to(device)
            output = model(data)
            reconstructed.append(output.cpu().numpy())
    reconstructed = np.concatenate(reconstructed)
    
    # Calculate reconstruction errors
    reconstruction_errors = np.mean(np.square(X_test.numpy() - reconstructed), axis=(1, 2, 3))
    
    # Calculate threshold (95th percentile)
    threshold = np.percentile(reconstruction_errors, 95)
    print(f"Anomaly threshold (95th percentile): {threshold:.4f}")
    
    # Calculate metrics
    metrics = calculate_metrics(reconstruction_errors, threshold)
    print("\nModel Performance Metrics:")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"F1 Score: {metrics['f1_score']:.4f}")
    
    # Plot results
    plot_results(X_test.numpy(), reconstructed, reconstruction_errors, metrics)
    
    # Save the model
    torch.save(model.state_dict(), 'anomaly_detection_model.pth')
    
    # Plot training history
    plt.figure(figsize=(10, 4))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training History')
    plt.legend()
    plt.savefig('training_history.png')
    plt.close()

if __name__ == "__main__":
    main() 