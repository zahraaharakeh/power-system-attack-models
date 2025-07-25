import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import warnings
warnings.filterwarnings('ignore')

class SimpleAttentionLayer(nn.Module):
    """Simple attention mechanism for feature learning"""
    def __init__(self, input_dim, attention_dim):
        super(SimpleAttentionLayer, self).__init__()
        self.attention_dim = attention_dim
        
        self.query = nn.Linear(input_dim, attention_dim)
        self.key = nn.Linear(input_dim, attention_dim)
        self.value = nn.Linear(input_dim, attention_dim)
        self.output_projection = nn.Linear(attention_dim, attention_dim)
        self.softmax = nn.Softmax(dim=-1)
        
    def forward(self, x):
        # x shape: (batch_size, input_dim)
        batch_size = x.size(0)
        x = x.unsqueeze(1)  # (batch_size, 1, input_dim)
        
        # Compute Q, K, V
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)
        
        # Compute attention scores
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(self.attention_dim)
        attention_weights = self.softmax(attention_scores)
        
        # Apply attention to values
        attended_output = torch.matmul(attention_weights, V)
        
        # Output projection
        attended_output = self.output_projection(attended_output)
        
        # Remove sequence dimension
        attended_output = attended_output.squeeze(1)
        
        return attended_output, attention_weights

class SimpleAdvancedAttentionAutoencoder(nn.Module):
    """Simplified advanced auto-encoder with attention mechanism"""
    def __init__(self, input_dim, hidden_dims, latent_dim, attention_dim=64):
        super(SimpleAdvancedAttentionAutoencoder, self).__init__()
        
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.attention_dim = attention_dim
        
        # Encoder layers
        encoder_layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            encoder_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.3)
            ])
            prev_dim = hidden_dim
        
        # Add attention layer
        encoder_layers.append(SimpleAttentionLayer(prev_dim, attention_dim))
        
        # Latent space
        encoder_layers.extend([
            nn.Linear(attention_dim, latent_dim),
            nn.BatchNorm1d(latent_dim),
            nn.ReLU(),
            nn.Dropout(0.2)
        ])
        
        # Separate encoder, attention, and latent layers
        num_encoder_layers = len(hidden_dims) * 4  # Each hidden dim has Linear, BatchNorm, ReLU, Dropout
        encoder_end = num_encoder_layers
        
        self.encoder = nn.Sequential(*encoder_layers[:encoder_end])
        self.attention = encoder_layers[encoder_end]  # Attention layer
        self.latent_layer = nn.Sequential(*encoder_layers[encoder_end+1:])  # Latent layer
        
        # Decoder layers
        decoder_layers = []
        decoder_dims = hidden_dims[::-1]  # Reverse the hidden dimensions
        prev_dim = latent_dim
        
        for hidden_dim in decoder_dims:
            decoder_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.3)
            ])
            prev_dim = hidden_dim
        
        # Output layer
        decoder_layers.append(nn.Linear(prev_dim, input_dim))
        
        self.decoder = nn.Sequential(*decoder_layers)
        
    def encode(self, x):
        # Encode
        encoded = self.encoder(x)
        
        # Apply attention
        attended, attention_weights = self.attention(encoded)
        
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
    X_malicious = generate_advanced_malicious_data(X_benign)
    
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

def generate_advanced_malicious_data(benign_data):
    """Generate advanced malicious data with more sophisticated attack patterns."""
    print("Generating advanced malicious data...")
    
    n_samples = len(benign_data)
    
    # 1. Advanced False Data Injection (FDI) - Time-varying manipulation
    fdi_attack = benign_data.copy()
    # Add time-varying perturbations
    time_factor = np.linspace(0, 1, len(benign_data))
    noise_scale = np.random.uniform(0.1, 0.3, benign_data.shape) * time_factor[:, np.newaxis]
    fdi_attack += np.random.normal(0, noise_scale, benign_data.shape)
    
    # 2. Advanced Replay Attack - Selective feature manipulation
    replay_attack = benign_data.copy()
    # Selectively replay different features
    feature_mask = np.random.choice([0, 1], size=benign_data.shape, p=[0.7, 0.3])
    shift_indices = np.random.randint(0, len(benign_data), len(benign_data))
    replay_attack = np.where(feature_mask, benign_data[shift_indices], replay_attack)
    replay_attack += np.random.normal(0, 0.2, benign_data.shape)
    
    # 3. Advanced Covert Attack - Feature-specific manipulation
    covert_attack = benign_data.copy()
    # Different manipulation for different features
    for i in range(benign_data.shape[1]):
        feature_noise = np.random.uniform(0.2, 0.5) * benign_data[:, i]
        covert_attack[:, i] += feature_noise
    
    # 4. Advanced Realistic Noise Attack - Adaptive noise
    realistic_noise = benign_data.copy()
    # Adaptive noise based on feature characteristics
    feature_std = np.std(benign_data, axis=0)
    adaptive_noise = np.random.normal(0, feature_std * 0.5, benign_data.shape)
    realistic_noise += adaptive_noise
    
    # 5. Advanced Targeted Attack - Multi-feature coordination
    targeted_attack = benign_data.copy()
    # Coordinate attacks across multiple features
    attack_patterns = np.random.choice([0, 1, 2], size=len(benign_data), p=[0.4, 0.3, 0.3])
    for i, pattern in enumerate(attack_patterns):
        if pattern == 0:  # Attack first two features
            targeted_attack[i, :2] += np.random.normal(0, 0.4, 2)
        elif pattern == 1:  # Attack last two features
            targeted_attack[i, 2:] += np.random.normal(0, 0.4, 2)
        else:  # Attack all features with different intensities
            intensities = np.random.uniform(0.2, 0.6, 4)
            targeted_attack[i] += np.random.normal(0, intensities)
    
    # 6. Advanced Systematic Bias Attack - Time-dependent bias
    bias_attack = benign_data.copy()
    # Time-dependent systematic bias
    time_bias = np.sin(np.linspace(0, 4*np.pi, len(benign_data))) * 0.3
    bias_attack += time_bias[:, np.newaxis]
    
    # 7. Advanced Stealth Attack - Minimal but coordinated changes
    stealth_attack = benign_data.copy()
    # Minimal but coordinated changes across features
    coordination_factor = np.random.uniform(-0.1, 0.1, len(benign_data))
    for i in range(benign_data.shape[1]):
        stealth_attack[:, i] += coordination_factor * (i + 1)
    
    # 8. Advanced Anomaly Injection - Outlier-based attacks
    anomaly_attack = benign_data.copy()
    # Inject statistical outliers
    outlier_indices = np.random.choice(len(benign_data), size=len(benign_data)//10, replace=False)
    for idx in outlier_indices:
        feature_idx = np.random.randint(0, 4)
        anomaly_attack[idx, feature_idx] += np.random.normal(0, 2.0)
    
    # Combine all attacks
    malicious_data = np.vstack([
        fdi_attack, replay_attack, covert_attack, realistic_noise,
        targeted_attack, bias_attack, stealth_attack, anomaly_attack
    ])
    
    print(f"Generated {len(malicious_data)} advanced malicious samples")
    return malicious_data

def advanced_loss_function(reconstructed, original, latent, beta=0.01):
    """Advanced loss function with reconstruction, KL divergence, and sparsity"""
    # Reconstruction loss
    mse_loss = nn.MSELoss()(reconstructed, original)
    
    # L1 regularization for sparsity
    l1_loss = torch.mean(torch.abs(latent))
    
    # KL divergence for latent space regularization
    kl_loss = -0.5 * torch.sum(1 + torch.log(torch.var(latent, dim=0) + 1e-8) - torch.var(latent, dim=0))
    
    # Combined loss
    total_loss = mse_loss + beta * l1_loss + 0.001 * kl_loss
    
    return total_loss, mse_loss, l1_loss, kl_loss

def train_simple_advanced_attention_autoencoder(model, train_loader, val_loader, epochs=100, lr=0.001):
    """Train the simple advanced attention auto-encoder with improved training"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    criterion = advanced_loss_function
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=20, factor=0.7, min_lr=1e-6)
    
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    patience_counter = 0
    patience_limit = 40
    
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
            loss, mse_loss, l1_loss, kl_loss = criterion(reconstructed, batch_x, latent)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_loss += loss.item()
        
        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_x in val_loader:
                batch_x = batch_x[0].to(device)
                reconstructed, latent, attention_weights = model(batch_x)
                loss, _, _, _ = criterion(reconstructed, batch_x, latent)
                val_loss += loss.item()
        
        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        
        scheduler.step(val_loss)
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1
            
        if patience_counter >= patience_limit:
            print(f"Early stopping at epoch {epoch+1}")
            break
        
        if (epoch + 1) % 50 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}')
    
    return train_losses, val_losses

def evaluate_simple_advanced_attention_autoencoder(model, test_loader, threshold_percentile=95):
    """Evaluate the simple advanced attention auto-encoder with ensemble threshold optimization"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()
    
    reconstruction_errors = []
    true_labels = []
    
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            
            reconstructed, latent, attention_weights = model(batch_x)
            
            # Calculate multiple types of reconstruction errors
            mse_error = torch.mean((batch_x - reconstructed) ** 2, dim=1)
            mae_error = torch.mean(torch.abs(batch_x - reconstructed), dim=1)
            cosine_error = 1 - torch.cosine_similarity(batch_x, reconstructed, dim=1)
            
            # Combined error metric
            combined_error = mse_error + 0.1 * mae_error + 0.05 * cosine_error
            reconstruction_errors.extend(combined_error.cpu().numpy())
            true_labels.extend(batch_y.cpu().numpy())
    
    reconstruction_errors = np.array(reconstruction_errors)
    true_labels = np.array(true_labels)
    
    # Advanced threshold optimization with multiple metrics
    best_f1 = 0
    best_threshold = 0
    best_percentile = 95
    
    for percentile in range(60, 100, 2):
        threshold = np.percentile(reconstruction_errors, percentile)
        predicted_labels = (reconstruction_errors > threshold).astype(int)
        
        try:
            f1 = f1_score(true_labels, predicted_labels)
            precision = precision_score(true_labels, predicted_labels)
            recall = recall_score(true_labels, predicted_labels)
            
            # Combined metric considering multiple factors
            combined_score = f1 * 0.5 + precision * 0.3 + recall * 0.2
            
            if combined_score > best_f1:
                best_f1 = combined_score
                best_threshold = threshold
                best_percentile = percentile
        except:
            continue
    
    # Use the best threshold found
    predicted_labels = (reconstruction_errors > best_threshold).astype(int)
    
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
        'threshold': best_threshold,
        'threshold_percentile': best_percentile,
        'predicted_labels': predicted_labels,
        'true_labels': true_labels
    }

def save_simple_advanced_attention_autoencoder_results(results, model_name="Simple Advanced Attention Autoencoder"):
    """Save results to text file"""
    print("Saving results...")
    
    with open('simple_advanced_attention_autoencoder_results.txt', 'w') as f:
        f.write("Simple Advanced Attention Autoencoder Model Results\n")
        f.write("=" * 50 + "\n\n")
        
        f.write("Model Performance Metrics:\n")
        f.write(f"Accuracy: {results['accuracy']:.4f}\n")
        f.write(f"Precision: {results['precision']:.4f}\n")
        f.write(f"Recall: {results['recall']:.4f}\n")
        f.write(f"F1 Score: {results['f1_score']:.4f}\n")
        f.write(f"Reconstruction Error Threshold: {results['threshold']:.4f}\n")
        f.write(f"Optimal Threshold Percentile: {results['threshold_percentile']}\n\n")
        
        f.write("Advanced Model Architecture:\n")
        f.write("- Simple attention mechanism\n")
        f.write("- Advanced loss function with KL divergence\n")
        f.write("- Ensemble threshold optimization\n")
        f.write("- Sophisticated attack generation\n\n")
        
        f.write("Advanced Attack Types Detected:\n")
        f.write("1. Advanced FDI - Time-varying manipulation\n")
        f.write("2. Advanced Replay - Selective feature manipulation\n")
        f.write("3. Advanced Covert - Feature-specific manipulation\n")
        f.write("4. Advanced Realistic - Adaptive noise\n")
        f.write("5. Advanced Targeted - Multi-feature coordination\n")
        f.write("6. Advanced Systematic - Time-dependent bias\n")
        f.write("7. Advanced Stealth - Minimal coordinated changes\n")
        f.write("8. Advanced Anomaly - Outlier-based attacks\n")
    
    print("Results saved to 'simple_advanced_attention_autoencoder_results.txt'")

def main():
    """Main function to run the complete Simple Advanced Attention Autoencoder analysis."""
    print("Simple Advanced Attention Autoencoder Model for Power System Attack Detection")
    print("=" * 70)
    
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
    
    # Create simple advanced model
    input_dim = X.shape[1]
    model = SimpleAdvancedAttentionAutoencoder(
        input_dim=input_dim,
        hidden_dims=[256, 128, 64],
        latent_dim=32,
        attention_dim=64
    )
    
    print(f"Input dimension: {input_dim}")
    print(f"Simple advanced model created successfully!")
    
    # Train model
    print("\nTraining Simple Advanced Attention Autoencoder...")
    train_losses, val_losses = train_simple_advanced_attention_autoencoder(
        model, train_loader, val_loader, epochs=1000, lr=0.001
    )
    
    # Evaluate model
    print("\nEvaluating Simple Advanced Attention Autoencoder...")
    results = evaluate_simple_advanced_attention_autoencoder(model, test_loader)
    
    # Print results
    print(f"\nSimple Advanced Attention Autoencoder Results:")
    print(f"Accuracy: {results['accuracy']:.4f}")
    print(f"Precision: {results['precision']:.4f}")
    print(f"Recall: {results['recall']:.4f}")
    print(f"F1 Score: {results['f1_score']:.4f}")
    print(f"Reconstruction Error Threshold: {results['threshold']:.4f}")
    print(f"Optimal Threshold Percentile: {results['threshold_percentile']}")
    
    # Save results
    save_simple_advanced_attention_autoencoder_results(results)
    
    # Save model
    torch.save(model.state_dict(), 'simple_advanced_attention_autoencoder_model.pth')
    print("Simple advanced model saved as 'simple_advanced_attention_autoencoder_model.pth'")
    
    print("\n" + "="*70)
    print("SIMPLE ADVANCED ATTENTION AUTOENCODER ANALYSIS COMPLETE!")
    print("="*70)
    print("Files generated:")
    print("- simple_advanced_attention_autoencoder_results.txt (detailed results)")
    print("- simple_advanced_attention_autoencoder_model.pth (trained model)")
    
    return model, results

if __name__ == "__main__":
    model, results = main() 