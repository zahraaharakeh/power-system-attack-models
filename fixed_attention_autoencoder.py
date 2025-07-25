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

class AttentionLayer(nn.Module):
    """Attention mechanism layer for auto-encoder"""
    def __init__(self, input_dim, attention_dim):
        super(AttentionLayer, self).__init__()
        self.attention_dim = attention_dim
        self.query = nn.Linear(input_dim, attention_dim)
        self.key = nn.Linear(input_dim, attention_dim)
        self.value = nn.Linear(input_dim, attention_dim)
        self.softmax = nn.Softmax(dim=-1)
        
    def forward(self, x):
        # x shape: (batch_size, seq_len, input_dim)
        batch_size, seq_len, input_dim = x.size()
        
        # Compute Q, K, V
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)
        
        # Compute attention scores
        attention_scores = torch.bmm(Q, K.transpose(1, 2)) / np.sqrt(self.attention_dim)
        attention_weights = self.softmax(attention_scores)
        
        # Apply attention to values
        attended_output = torch.bmm(attention_weights, V)
        
        return attended_output, attention_weights

class AttentionAutoencoder(nn.Module):
    """Auto-encoder with attention mechanism for power system attack detection"""
    def __init__(self, input_dim, hidden_dims, latent_dim, attention_dim=64):
        super(AttentionAutoencoder, self).__init__()
        
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.attention_dim = attention_dim
        
        # Encoder layers
        encoder_layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            encoder_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.2)
            ])
            prev_dim = hidden_dim
        
        # Create encoder
        self.encoder = nn.Sequential(*encoder_layers)
        
        # Attention layer
        self.attention = AttentionLayer(prev_dim, attention_dim)
        
        # Latent layer
        self.latent_layer = nn.Sequential(
            nn.Linear(attention_dim, latent_dim),
            nn.ReLU()
        )
        
        # Decoder layers
        decoder_layers = []
        decoder_dims = hidden_dims[::-1]  # Reverse the hidden dimensions
        prev_dim = latent_dim
        
        for hidden_dim in decoder_dims:
            decoder_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.2)
            ])
            prev_dim = hidden_dim
        
        # Output layer
        decoder_layers.append(nn.Linear(prev_dim, input_dim))
        
        self.decoder = nn.Sequential(*decoder_layers)
        
    def encode(self, x):
        # Encode
        encoded = self.encoder(x)
        
        # Reshape for attention (add sequence dimension)
        batch_size = encoded.size(0)
        encoded = encoded.unsqueeze(1)  # (batch_size, 1, hidden_dim)
        
        # Apply attention
        attended, attention_weights = self.attention(encoded)
        attended = attended.squeeze(1)  # Remove sequence dimension
        
        # Latent representation
        latent = self.latent_layer(attended)
        
        return latent, attention_weights
    
    def decode(self, latent):
        return self.decoder(latent)
    
    def forward(self, x):
        latent, attention_weights = self.encode(x)
        reconstructed = self.decode(latent)
        return reconstructed, latent, attention_weights

def load_and_preprocess_data(benign_file):
    """Load and preprocess the data from benign file and generate malicious data."""
    print("Loading and preprocessing data...")
    
    # Load benign data
    benign_df = pd.read_excel(benign_file)
    
    # Extract features
    feature_columns = ['Pd_new', 'Qd_new', 'Vm', 'Va']
    X_benign = benign_df[feature_columns].values
    
    print(f"Benign samples loaded: {len(X_benign)}")
    print(f"Features: {feature_columns}")
    
    # Generate malicious data with more sophisticated attacks
    X_malicious = generate_sophisticated_malicious_data(X_benign)
    
    # Create labels (0 for benign, 1 for malicious)
    y_benign = np.zeros(len(X_benign))
    y_malicious = np.ones(len(X_malicious))
    
    # Combine data
    X = np.vstack([X_benign, X_malicious])
    y = np.concatenate([y_benign, y_malicious])
    
    print(f"Total samples: {len(X)}")
    print(f"Benign samples: {len(X_benign)}")
    print(f"Malicious samples: {len(X_malicious)}")
    print(f"Class distribution: {np.bincount(y.astype(int))}")
    
    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    return X_scaled, y, scaler, feature_columns

def generate_sophisticated_malicious_data(benign_data):
    """Generate sophisticated malicious data with realistic attack patterns."""
    print("Generating sophisticated malicious data...")
    
    n_samples = len(benign_data)
    
    # 1. False Data Injection (FDI) - Subtle manipulation
    fdi_attack = benign_data.copy()
    # Add subtle perturbations that maintain system observability
    noise_scale = np.random.uniform(0.05, 0.15, benign_data.shape)
    fdi_attack += np.random.normal(0, noise_scale, benign_data.shape)
    
    # 2. Replay Attack - Reuse previous measurements with noise
    replay_attack = benign_data.copy()
    # Shift data and add noise
    shift_indices = np.random.randint(0, len(benign_data), len(benign_data))
    replay_attack = benign_data[shift_indices] + np.random.normal(0, 0.1, benign_data.shape)
    
    # 3. Covert Attack - Maintains system observability
    covert_attack = benign_data.copy()
    # Add proportional noise based on measurement type
    proportional_noise = benign_data * np.random.uniform(0.1, 0.2, benign_data.shape)
    covert_attack += proportional_noise
    
    # 4. Realistic Noise Attack - Proportional to measurement characteristics
    realistic_noise = benign_data.copy()
    # Add noise proportional to the magnitude of each feature
    feature_std = np.std(benign_data, axis=0)
    realistic_noise += np.random.normal(0, feature_std * 0.3, benign_data.shape)
    
    # Combine all attacks
    malicious_data = np.vstack([fdi_attack, replay_attack, covert_attack, realistic_noise])
    
    print(f"Generated {len(malicious_data)} sophisticated malicious samples")
    return malicious_data

def train_attention_autoencoder(model, train_loader, val_loader, epochs=100, lr=0.001):
    """Train the attention auto-encoder"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)
    
    train_losses = []
    val_losses = []
    
    print(f"Training on device: {device}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0.0
        for batch_x in train_loader:
            batch_x = batch_x[0].to(device)
            
            optimizer.zero_grad()
            reconstructed, latent, attention_weights = model(batch_x)
            loss = criterion(reconstructed, batch_x)
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_x in val_loader:
                batch_x = batch_x[0].to(device)
                reconstructed, latent, attention_weights = model(batch_x)
                loss = criterion(reconstructed, batch_x)
                val_loss += loss.item()
        
        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        
        scheduler.step(val_loss)
        
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}')
    
    return train_losses, val_losses

def evaluate_attention_autoencoder(model, test_loader, threshold_percentile=95):
    """Evaluate the attention auto-encoder for anomaly detection"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()
    
    reconstruction_errors = []
    true_labels = []
    
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            
            reconstructed, latent, attention_weights = model(batch_x)
            
            # Calculate reconstruction error
            error = torch.mean((batch_x - reconstructed) ** 2, dim=1)
            reconstruction_errors.extend(error.cpu().numpy())
            true_labels.extend(batch_y.cpu().numpy())
    
    reconstruction_errors = np.array(reconstruction_errors)
    true_labels = np.array(true_labels)
    
    # Set threshold based on percentile of reconstruction errors
    threshold = np.percentile(reconstruction_errors, threshold_percentile)
    
    # Predict anomalies (high reconstruction error = anomaly)
    predicted_labels = (reconstruction_errors > threshold).astype(int)
    
    # Calculate metrics
    accuracy = accuracy_score(true_labels, predicted_labels)
    precision = precision_score(true_labels, predicted_labels)
    recall = recall_score(true_labels, predicted_labels)
    f1 = f1_score(true_labels, predicted_labels)
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'reconstruction_errors': reconstruction_errors,
        'threshold': threshold,
        'predicted_labels': predicted_labels,
        'true_labels': true_labels
    }

def plot_attention_autoencoder_results(train_losses, val_losses, results, model_name="Attention Autoencoder"):
    """Plot training history and results"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. Training History
    axes[0, 0].plot(train_losses, label='Training Loss', color='blue')
    axes[0, 0].plot(val_losses, label='Validation Loss', color='red')
    axes[0, 0].set_title('Training History')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # 2. Reconstruction Error Distribution
    axes[0, 1].hist(results['reconstruction_errors'], bins=50, alpha=0.7, color='green')
    axes[0, 1].axvline(results['threshold'], color='red', linestyle='--', 
                       label=f'Threshold ({results["threshold"]:.4f})')
    axes[0, 1].set_title('Reconstruction Error Distribution')
    axes[0, 1].set_xlabel('Reconstruction Error')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].legend()
    
    # 3. Confusion Matrix
    cm = confusion_matrix(results['true_labels'], results['predicted_labels'])
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[1, 0])
    axes[1, 0].set_title('Confusion Matrix')
    axes[1, 0].set_xlabel('Predicted')
    axes[1, 0].set_ylabel('Actual')
    
    # 4. Metrics Bar Plot
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
    values = [results['accuracy'], results['precision'], 
              results['recall'], results['f1_score']]
    
    bars = axes[1, 1].bar(metrics, values, color=['blue', 'green', 'orange', 'red'])
    axes[1, 1].set_title('Model Performance Metrics')
    axes[1, 1].set_ylabel('Score')
    axes[1, 1].set_ylim(0, 1)
    
    # Add value labels on bars
    for bar, value in zip(bars, values):
        axes[1, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                       f'{value:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('attention_autoencoder_results.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("Visualizations saved as 'attention_autoencoder_results.png'")

def save_attention_autoencoder_results(results, model_name="Attention Autoencoder"):
    """Save results to text file"""
    print("Saving results...")
    
    with open('attention_autoencoder_results.txt', 'w') as f:
        f.write("Attention Autoencoder Model Results\n")
        f.write("=" * 50 + "\n\n")
        
        f.write("Model Performance Metrics:\n")
        f.write(f"Accuracy: {results['accuracy']:.4f}\n")
        f.write(f"Precision: {results['precision']:.4f}\n")
        f.write(f"Recall: {results['recall']:.4f}\n")
        f.write(f"F1 Score: {results['f1_score']:.4f}\n")
        f.write(f"Reconstruction Error Threshold: {results['threshold']:.4f}\n\n")
        
        f.write("Model Architecture:\n")
        f.write("- Attention mechanism for feature importance\n")
        f.write("- Auto-encoder with sophisticated encoding/decoding\n")
        f.write("- Anomaly detection based on reconstruction error\n")
        f.write("- Unsupervised learning approach\n\n")
        
        f.write("Attack Types Detected:\n")
        f.write("1. False Data Injection (FDI) - Subtle manipulation\n")
        f.write("2. Replay Attack - Reuse of previous measurements\n")
        f.write("3. Covert Attack - Maintains system observability\n")
        f.write("4. Realistic Noise Attack - Proportional noise\n")
    
    print("Results saved to 'attention_autoencoder_results.txt'")

def main():
    """Main function to run the complete Attention Autoencoder analysis."""
    print("Attention Autoencoder Model for Power System Attack Detection")
    print("=" * 60)
    
    # Load and preprocess data
    X, y, scaler, feature_names = load_and_preprocess_data('benign_bus14.xlsx')
    
    # Split data into train, validation, and test
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.2, random_state=42, stratify=y_temp
    )
    
    # Create data loaders
    train_dataset = TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train))
    val_dataset = TensorDataset(torch.FloatTensor(X_val), torch.FloatTensor(y_val))
    test_dataset = TensorDataset(torch.FloatTensor(X_test), torch.FloatTensor(y_test))
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    print(f"Training set size: {len(X_train)}")
    print(f"Validation set size: {len(X_val)}")
    print(f"Test set size: {len(X_test)}")
    
    # Create model
    input_dim = X.shape[1]
    model = AttentionAutoencoder(
        input_dim=input_dim,
        hidden_dims=[128, 64],
        latent_dim=32,
        attention_dim=64
    )
    
    print(f"Input dimension: {input_dim}")
    print(f"Model created successfully!")
    
    # Train model
    print("\nTraining Attention Autoencoder...")
    train_losses, val_losses = train_attention_autoencoder(
        model, train_loader, val_loader, epochs=50, lr=0.001
    )
    
    # Evaluate model
    print("\nEvaluating Attention Autoencoder...")
    results = evaluate_attention_autoencoder(model, test_loader)
    
    # Print results
    print(f"\nAttention Autoencoder Results:")
    print(f"Accuracy: {results['accuracy']:.4f}")
    print(f"Precision: {results['precision']:.4f}")
    print(f"Recall: {results['recall']:.4f}")
    print(f"F1 Score: {results['f1_score']:.4f}")
    print(f"Reconstruction Error Threshold: {results['threshold']:.4f}")
    
    # Plot results
    plot_attention_autoencoder_results(train_losses, val_losses, results)
    
    # Save results
    save_attention_autoencoder_results(results)
    
    # Save model
    torch.save(model.state_dict(), 'attention_autoencoder_model.pth')
    print("Model saved as 'attention_autoencoder_model.pth'")
    
    print("\n" + "="*60)
    print("ATTENTION AUTOENCODER ANALYSIS COMPLETE!")
    print("="*60)
    print("Files generated:")
    print("- attention_autoencoder_results.png (visualizations)")
    print("- attention_autoencoder_results.txt (detailed results)")
    print("- attention_autoencoder_model.pth (trained model)")
    
    return model, results

if __name__ == "__main__":
    model, results = main() 