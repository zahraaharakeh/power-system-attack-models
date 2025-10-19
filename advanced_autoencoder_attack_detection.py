import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from imblearn.over_sampling import SMOTE
warnings.filterwarnings('ignore')

class AttentionLayer(nn.Module):
    """Self-attention layer for autoencoder."""
    
    def __init__(self, input_dim, attention_dim=64):
        super(AttentionLayer, self).__init__()
        self.attention_dim = attention_dim
        
        self.W_q = nn.Linear(input_dim, attention_dim)
        self.W_k = nn.Linear(input_dim, attention_dim)
        self.W_v = nn.Linear(input_dim, attention_dim)
        self.W_o = nn.Linear(attention_dim, input_dim)
        
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, x):
        batch_size, seq_len, input_dim = x.shape
        
        # Compute Q, K, V
        Q = self.W_q(x)  # [batch, seq_len, attention_dim]
        K = self.W_k(x)  # [batch, seq_len, attention_dim]
        V = self.W_v(x)  # [batch, seq_len, attention_dim]
        
        # Compute attention scores
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(self.attention_dim)
        attention_weights = F.softmax(attention_scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Apply attention to values
        attended_values = torch.matmul(attention_weights, V)
        output = self.W_o(attended_values)
        
        return output, attention_weights

class VariationalLayer(nn.Module):
    """Variational layer for VAE-style autoencoder."""
    
    def __init__(self, input_dim, latent_dim):
        super(VariationalLayer, self).__init__()
        self.latent_dim = latent_dim
        
        self.mu_layer = nn.Linear(input_dim, latent_dim)
        self.logvar_layer = nn.Linear(input_dim, latent_dim)
        
    def forward(self, x):
        mu = self.mu_layer(x)
        logvar = self.logvar_layer(x)
        
        # Reparameterization trick
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        
        return z, mu, logvar

class AdvancedAutoencoder(nn.Module):
    """Advanced autoencoder with attention and variational components."""
    
    def __init__(self, input_dim, latent_dim=32, hidden_dims=[128, 64], use_attention=True, use_variational=True):
        super(AdvancedAutoencoder, self).__init__()
        
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.use_attention = use_attention
        self.use_variational = use_variational
        
        # Encoder
        encoder_layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            encoder_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.2)
            ])
            prev_dim = hidden_dim
        
        self.encoder = nn.Sequential(*encoder_layers)
        
        # Attention layer (if enabled)
        if use_attention:
            self.attention = AttentionLayer(prev_dim, attention_dim=64)
        
        # Variational layer (if enabled)
        if use_variational:
            self.variational = VariationalLayer(prev_dim, latent_dim)
        else:
            self.encoder_to_latent = nn.Linear(prev_dim, latent_dim)
        
        # Decoder
        decoder_layers = []
        prev_dim = latent_dim
        
        # Reverse the hidden dimensions for decoder
        for hidden_dim in reversed(hidden_dims):
            decoder_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.2)
            ])
            prev_dim = hidden_dim
        
        decoder_layers.append(nn.Linear(prev_dim, input_dim))
        self.decoder = nn.Sequential(*decoder_layers)
        
        # Anomaly detection head
        self.anomaly_detector = nn.Sequential(
            nn.Linear(latent_dim, latent_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(latent_dim // 2, 1),
            nn.Sigmoid()
        )
        
    def encode(self, x):
        # Encoder
        encoded = self.encoder(x)
        
        # Attention (if enabled)
        if self.use_attention:
            encoded, attention_weights = self.attention(encoded.unsqueeze(1))
            encoded = encoded.squeeze(1)
        else:
            attention_weights = None
        
        # Variational encoding (if enabled)
        if self.use_variational:
            z, mu, logvar = self.variational(encoded)
            return z, mu, logvar, attention_weights
        else:
            z = self.encoder_to_latent(encoded)
            return z, None, None, attention_weights
    
    def decode(self, z):
        return self.decoder(z)
    
    def forward(self, x):
        # Encode
        z, mu, logvar, attention_weights = self.encode(x)
        
        # Decode
        reconstructed = self.decode(z)
        
        # Anomaly detection
        anomaly_score = self.anomaly_detector(z)
        
        return reconstructed, z, mu, logvar, attention_weights, anomaly_score

class AutoencoderLoss(nn.Module):
    """Combined loss function for autoencoder training."""
    
    def __init__(self, reconstruction_weight=1.0, kl_weight=0.1, anomaly_weight=0.5, attention_weight=0.01):
        super(AutoencoderLoss, self).__init__()
        self.reconstruction_weight = reconstruction_weight
        self.kl_weight = kl_weight
        self.anomaly_weight = anomaly_weight
        self.attention_weight = attention_weight
        
        self.mse_loss = nn.MSELoss()
        self.bce_loss = nn.BCELoss()
        
    def forward(self, reconstructed, original, mu, logvar, attention_weights, anomaly_score, labels):
        # Reconstruction loss
        reconstruction_loss = self.mse_loss(reconstructed, original)
        
        # KL divergence loss (for variational autoencoder)
        kl_loss = 0
        if mu is not None and logvar is not None:
            kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
            kl_loss = kl_loss / mu.size(0)  # Normalize by batch size
        
        # Anomaly detection loss
        anomaly_loss = self.bce_loss(anomaly_score.squeeze(), labels.float())
        
        # Attention regularization
        attention_reg = 0
        if attention_weights is not None:
            attention_reg = torch.mean(torch.abs(attention_weights - 0.5))
        
        # Total loss
        total_loss = (self.reconstruction_weight * reconstruction_loss + 
                     self.kl_weight * kl_loss + 
                     self.anomaly_weight * anomaly_loss + 
                     self.attention_weight * attention_reg)
        
        return total_loss, reconstruction_loss, kl_loss, anomaly_loss, attention_reg

def generate_advanced_malicious_data(benign_data):
    """Generate advanced malicious data with sophisticated attack patterns."""
    print("Generating advanced malicious data for autoencoder...")
    
    n_samples = len(benign_data)
    all_attacks = []
    
    # 1. Advanced False Data Injection (FDI)
    fdi_attack = benign_data.copy()
    for i in range(benign_data.shape[1]):
        # Different attack intensity for different features
        if i < 2:  # Power features (Pd, Qd)
            noise_scale = np.random.uniform(0.1, 0.4, (len(benign_data), 1))
        else:  # Voltage features (Vm, Va)
            noise_scale = np.random.uniform(0.05, 0.3, (len(benign_data), 1))
        
        fdi_attack[:, i:i+1] += np.random.normal(0, noise_scale, (len(benign_data), 1)).flatten()
    all_attacks.append(fdi_attack)
    
    # 2. Coordinated Multi-Feature Attack
    coord_attack = benign_data.copy()
    attack_mask = np.random.choice([0, 1], size=benign_data.shape, p=[0.5, 0.5])
    coord_attack += attack_mask * np.random.normal(0, 0.3, benign_data.shape)
    all_attacks.append(coord_attack)
    
    # 3. Stealth Attack with Statistical Preservation
    stealth_attack = benign_data.copy()
    feature_means = np.mean(benign_data, axis=0)
    feature_stds = np.std(benign_data, axis=0)
    
    for i in range(benign_data.shape[1]):
        # Add noise while maintaining statistical properties
        noise = np.random.normal(0, feature_stds[i] * 0.4, len(benign_data))
        stealth_attack[:, i] += noise
        # Adjust to maintain original mean
        stealth_attack[:, i] -= np.mean(stealth_attack[:, i]) - feature_means[i]
    all_attacks.append(stealth_attack)
    
    # 4. Replay Attack with Temporal Drift
    replay_attack = benign_data.copy()
    for i in range(len(benign_data)):
        # Replay previous measurements with drift
        replay_idx = max(0, i - np.random.randint(1, 10))
        replay_attack[i] = benign_data[replay_idx]
        # Add temporal drift
        drift = np.random.normal(0, 0.2, benign_data.shape[1])
        replay_attack[i] += drift
    all_attacks.append(replay_attack)
    
    # 5. Adversarial Attack (Feature-specific)
    adversarial_attack = benign_data.copy()
    for i in range(benign_data.shape[1]):
        # Add adversarial perturbations
        perturbation = np.random.uniform(-0.2, 0.2, len(benign_data))
        adversarial_attack[:, i] += perturbation
    all_attacks.append(adversarial_attack)
    
    # Combine all attacks
    malicious_data = np.vstack(all_attacks)
    
    # Ensure we have enough samples
    if len(malicious_data) > n_samples:
        indices = np.random.choice(len(malicious_data), n_samples, replace=False)
        malicious_data = malicious_data[indices]
    elif len(malicious_data) < n_samples:
        additional_needed = n_samples - len(malicious_data)
        additional_indices = np.random.choice(len(malicious_data), additional_needed, replace=True)
        additional_data = malicious_data[additional_indices]
        malicious_data = np.vstack([malicious_data, additional_data])
    
    print(f"Generated {len(malicious_data)} advanced malicious samples")
    return malicious_data

def load_and_preprocess_autoencoder_data(benign_file, balance_classes=True):
    """Load and preprocess data for autoencoder model."""
    print("Loading and preprocessing data for advanced autoencoder...")
    
    # Load benign data
    benign_df = pd.read_excel(benign_file)
    feature_columns = ['Pd_new', 'Qd_new', 'Vm', 'Va']
    X_benign = benign_df[feature_columns].values
    
    print(f"Benign samples loaded: {len(X_benign)}")
    print(f"Features: {feature_columns}")
    
    # Generate malicious data
    X_malicious = generate_advanced_malicious_data(X_benign)
    
    # Create labels
    y_benign = np.zeros(len(X_benign))
    y_malicious = np.ones(len(X_malicious))
    
    # Combine data
    X = np.vstack([X_benign, X_malicious])
    y = np.concatenate([y_benign, y_malicious])
    
    print(f"Total samples before balancing: {len(X)}")
    print(f"Class distribution: Benign={np.sum(y==0)}, Malicious={np.sum(y==1)}")
    
    # Apply SMOTE for class balancing
    if balance_classes:
        smote = SMOTE(random_state=42, k_neighbors=3)
        X_balanced, y_balanced = smote.fit_resample(X, y)
        print(f"After SMOTE balancing: {len(X_balanced)} samples")
        print(f"Balanced distribution: Benign={np.sum(y_balanced==0)}, Malicious={np.sum(y_balanced==1)}")
        X, y = X_balanced, y_balanced
    
    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    return X_scaled, y, scaler, feature_columns

def train_autoencoder_model(model, train_loader, val_loader, num_epochs=100, device='cuda', class_weights=None):
    """Train the advanced autoencoder model."""
    model = model.to(device)
    
    # Loss function
    criterion = AutoencoderLoss(
        reconstruction_weight=1.0,
        kl_weight=0.1,
        anomaly_weight=0.5,
        attention_weight=0.01
    )
    
    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-5)
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=10, factor=0.5)
    
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    patience = 15
    patience_counter = 0
    
    for epoch in range(num_epochs):
        # Training
        model.train()
        total_train_loss = 0
        
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            
            reconstructed, z, mu, logvar, attention_weights, anomaly_score = model(batch_x)
            
            total_loss, recon_loss, kl_loss, anomaly_loss, attention_reg = criterion(
                reconstructed, batch_x, mu, logvar, attention_weights, anomaly_score, batch_y
            )
            
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            total_train_loss += total_loss.item()
        
        avg_train_loss = total_train_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        
        # Validation
        model.eval()
        total_val_loss = 0
        
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                reconstructed, z, mu, logvar, attention_weights, anomaly_score = model(batch_x)
                
                total_loss, _, _, _, _ = criterion(
                    reconstructed, batch_x, mu, logvar, attention_weights, anomaly_score, batch_y
                )
                total_val_loss += total_loss.item()
        
        avg_val_loss = total_val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        
        # Learning rate scheduling
        scheduler.step(avg_val_loss)
        
        # Early stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            # Save best model
            torch.save(model.state_dict(), 'best_autoencoder_model.pth')
        else:
            patience_counter += 1
        
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
    
    # Load best model
    model.load_state_dict(torch.load('best_autoencoder_model.pth'))
    return train_losses, val_losses

def evaluate_autoencoder_model(model, test_loader, device='cuda'):
    """Comprehensive evaluation of autoencoder model."""
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []
    all_reconstruction_errors = []
    all_attention_weights = []
    
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            reconstructed, z, mu, logvar, attention_weights, anomaly_score = model(batch_x)
            
            # Calculate reconstruction error
            reconstruction_error = torch.mean((reconstructed - batch_x) ** 2, dim=1)
            
            # Use anomaly score as prediction
            preds = (anomaly_score.squeeze() > 0.5).long()
            probs = anomaly_score.squeeze()
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(batch_y.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            all_reconstruction_errors.extend(reconstruction_error.cpu().numpy())
            
            if attention_weights is not None:
                all_attention_weights.extend(attention_weights.cpu().numpy())
    
    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='weighted')
    recall = recall_score(all_labels, all_preds, average='weighted')
    f1 = f1_score(all_labels, all_preds, average='weighted')
    
    # ROC AUC
    all_probs = np.array(all_probs)
    roc_auc = roc_auc_score(all_labels, all_probs)
    
    # Class-specific metrics
    precision_per_class = precision_score(all_labels, all_preds, average=None)
    recall_per_class = recall_score(all_labels, all_preds, average=None)
    f1_per_class = f1_score(all_labels, all_preds, average=None)
    
    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'roc_auc': roc_auc,
        'precision_per_class': precision_per_class,
        'recall_per_class': recall_per_class,
        'f1_per_class': f1_per_class,
        'confusion_matrix': confusion_matrix(all_labels, all_preds),
        'reconstruction_errors': np.array(all_reconstruction_errors),
        'attention_weights': np.array(all_attention_weights) if all_attention_weights else None
    }
    
    return metrics

def plot_autoencoder_results(train_losses, val_losses, metrics, feature_names):
    """Plot comprehensive autoencoder results."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Training history
    axes[0, 0].plot(train_losses, label='Training Loss', color='blue')
    axes[0, 0].plot(val_losses, label='Validation Loss', color='red')
    axes[0, 0].set_title('Advanced Autoencoder Training History')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # Confusion Matrix
    cm = metrics['confusion_matrix']
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0, 1])
    axes[0, 1].set_title('Autoencoder Confusion Matrix')
    axes[0, 1].set_xlabel('Predicted')
    axes[0, 1].set_ylabel('Actual')
    
    # Metrics comparison
    metrics_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']
    metrics_values = [metrics['accuracy'], metrics['precision'], metrics['recall'], 
                     metrics['f1_score'], metrics['roc_auc']]
    
    bars = axes[0, 2].bar(metrics_names, metrics_values, color=['skyblue', 'lightgreen', 'lightcoral', 'lightsalmon', 'lightpink'])
    axes[0, 2].set_title('Autoencoder Performance Metrics')
    axes[0, 2].set_ylabel('Score')
    axes[0, 2].set_ylim(0, 1)
    
    # Add value labels on bars
    for bar, value in zip(bars, metrics_values):
        axes[0, 2].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                       f'{value:.3f}', ha='center', va='bottom')
    
    # Class-specific performance
    class_names = ['Benign', 'Malicious']
    x_pos = np.arange(len(class_names))
    width = 0.25
    
    axes[1, 0].bar(x_pos - width, metrics['precision_per_class'], width, label='Precision', alpha=0.8)
    axes[1, 0].bar(x_pos, metrics['recall_per_class'], width, label='Recall', alpha=0.8)
    axes[1, 0].bar(x_pos + width, metrics['f1_per_class'], width, label='F1-Score', alpha=0.8)
    
    axes[1, 0].set_title('Autoencoder Class-specific Performance')
    axes[1, 0].set_xlabel('Class')
    axes[1, 0].set_ylabel('Score')
    axes[1, 0].set_xticks(x_pos)
    axes[1, 0].set_xticklabels(class_names)
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Reconstruction error analysis
    reconstruction_errors = metrics['reconstruction_errors']
    axes[1, 1].hist(reconstruction_errors, bins=50, alpha=0.7, color='purple', edgecolor='black')
    axes[1, 1].set_title('Reconstruction Error Distribution')
    axes[1, 1].set_xlabel('Reconstruction Error')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].grid(True, alpha=0.3)
    
    # Attention analysis (if available)
    if metrics['attention_weights'] is not None:
        attention_weights = metrics['attention_weights']
        mean_attention = np.mean(attention_weights, axis=(0, 1))  # Average across batch and heads
        
        axes[1, 2].bar(range(len(mean_attention)), mean_attention, color='orange', alpha=0.7)
        axes[1, 2].set_title('Autoencoder Attention Weights')
        axes[1, 2].set_xlabel('Features')
        axes[1, 2].set_ylabel('Average Attention Weight')
        axes[1, 2].set_xticks(range(len(feature_names)))
        axes[1, 2].set_xticklabels(feature_names, rotation=45)
        axes[1, 2].grid(True, alpha=0.3)
    else:
        axes[1, 2].text(0.5, 0.5, 'Attention weights not available', ha='center', va='center', transform=axes[1, 2].transAxes)
        axes[1, 2].set_title('Autoencoder Attention Analysis')
    
    plt.tight_layout()
    plt.savefig('autoencoder_comprehensive_results.png', dpi=300, bbox_inches='tight')
    plt.show()

def main():
    """Main function to run advanced autoencoder attack detection."""
    print("🔋 Advanced Autoencoder Power System Attack Detection")
    print("=" * 60)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load and preprocess data
    X, y, scaler, feature_names = load_and_preprocess_autoencoder_data('benign_bus14.xlsx', balance_classes=True)
    
    print(f"\nDataset Statistics:")
    print(f"Total samples: {len(X)}")
    print(f"Benign samples: {np.sum(y == 0)}")
    print(f"Malicious samples: {np.sum(y == 1)}")
    print(f"Feature shape: {X.shape}")
    print(f"Features: {feature_names}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42, stratify=y_train)
    
    # Convert to PyTorch tensors
    X_train = torch.FloatTensor(X_train)
    y_train = torch.LongTensor(y_train)
    X_val = torch.FloatTensor(X_val)
    y_val = torch.LongTensor(y_val)
    X_test = torch.FloatTensor(X_test)
    y_test = torch.LongTensor(y_test)
    
    # Create data loaders
    from torch.utils.data import TensorDataset, DataLoader
    
    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)
    test_dataset = TensorDataset(X_test, y_test)
    
    batch_size = 64
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # Create advanced autoencoder model
    model = AdvancedAutoencoder(
        input_dim=len(feature_names),
        latent_dim=32,
        hidden_dims=[128, 64],
        use_attention=True,
        use_variational=True
    )
    
    print(f"\nAdvanced Autoencoder Model Architecture:")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Input dimension: {len(feature_names)}")
    print(f"Latent dimension: 32")
    print(f"Hidden dimensions: [128, 64]")
    print(f"Attention mechanism: Enabled")
    print(f"Variational encoding: Enabled")
    print(model)
    
    # Train model
    print("\nTraining advanced autoencoder model...")
    train_losses, val_losses = train_autoencoder_model(
        model, train_loader, val_loader, 
        num_epochs=100, device=device
    )
    
    # Evaluate model
    print("\nEvaluating autoencoder model...")
    metrics = evaluate_autoencoder_model(model, test_loader, device=device)
    
    # Print results
    print("\n🏆 Advanced Autoencoder Model Performance:")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"F1-Score: {metrics['f1_score']:.4f}")
    print(f"ROC-AUC: {metrics['roc_auc']:.4f}")
    
    print(f"\nClass-specific Performance:")
    print(f"Benign - Precision: {metrics['precision_per_class'][0]:.4f}, Recall: {metrics['recall_per_class'][0]:.4f}, F1: {metrics['f1_per_class'][0]:.4f}")
    print(f"Malicious - Precision: {metrics['precision_per_class'][1]:.4f}, Recall: {metrics['recall_per_class'][1]:.4f}, F1: {metrics['f1_per_class'][1]:.4f}")
    
    # Plot results
    plot_autoencoder_results(train_losses, val_losses, metrics, feature_names)
    
    # Save model and results
    torch.save(model.state_dict(), 'advanced_autoencoder_final.pth')
    
    # Save detailed results
    with open('autoencoder_results_report.txt', 'w') as f:
        f.write("Advanced Autoencoder Power System Attack Detection Results\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Dataset: {len(X)} samples, {X.shape[1]} features\n")
        f.write(f"Model: Advanced Autoencoder with {sum(p.numel() for p in model.parameters()):,} parameters\n\n")
        f.write("Performance Metrics:\n")
        f.write(f"Accuracy: {metrics['accuracy']:.4f}\n")
        f.write(f"Precision: {metrics['precision']:.4f}\n")
        f.write(f"Recall: {metrics['recall']:.4f}\n")
        f.write(f"F1-Score: {metrics['f1_score']:.4f}\n")
        f.write(f"ROC-AUC: {metrics['roc_auc']:.4f}\n\n")
        f.write("Model Architecture:\n")
        f.write(f"Input Dimension: {len(feature_names)}\n")
        f.write(f"Latent Dimension: 32\n")
        f.write(f"Hidden Dimensions: [128, 64]\n")
        f.write(f"Attention Mechanism: Enabled\n")
        f.write(f"Variational Encoding: Enabled\n")
    
    print(f"\n✅ Advanced autoencoder implementation complete!")
    print(f"📊 Results saved to 'autoencoder_results_report.txt'")
    print(f"🎯 Model saved to 'advanced_autoencoder_final.pth'")
    
    return model, metrics

if __name__ == "__main__":
    main()
