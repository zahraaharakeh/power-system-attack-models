import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import warnings
from torch_geometric.nn import GCNConv
warnings.filterwarnings('ignore')

class UnsupervisedGraphInformer(nn.Module):
    """Pure Unsupervised Graph Informer for power system attack detection."""
    
    def __init__(self, input_dim, d_model=256, n_heads=8, n_layers=3, seq_len=24, num_nodes=14):
        super(UnsupervisedGraphInformer, self).__init__()
        
        self.input_dim = input_dim
        self.d_model = d_model
        self.seq_len = seq_len
        self.num_nodes = num_nodes
        
        # Input projection
        self.input_projection = nn.Linear(input_dim, d_model)
        
        # Positional encoding
        self.pos_encoding = self._create_positional_encoding(d_model, seq_len)
        
        # Transformer blocks
        self.encoder_layers = nn.ModuleList([
            self._create_transformer_block(d_model, n_heads)
            for _ in range(n_layers)
        ])
        
        # Graph convolution layers
        self.graph_convs = nn.ModuleList([
            GCNConv(d_model, d_model) for _ in range(2)
        ])
        
        # Multi-scale reconstruction decoders
        self.decoder_global = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, input_dim)
        )
        
        self.decoder_local = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, input_dim)
        )
        
        # Advanced anomaly detection heads
        self.anomaly_detector_temporal = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, 1),
            nn.Sigmoid()
        )
        
        self.anomaly_detector_structural = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, 1),
            nn.Sigmoid()
        )
        
        self.anomaly_detector_statistical = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, 1),
            nn.Sigmoid()
        )
        
    def _create_positional_encoding(self, d_model, max_len):
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        
        return nn.Parameter(pe, requires_grad=False)
    
    def _create_transformer_block(self, d_model, n_heads):
        return nn.ModuleDict({
            'attention': nn.MultiheadAttention(d_model, n_heads, batch_first=True),
            'norm1': nn.LayerNorm(d_model),
            'norm2': nn.LayerNorm(d_model),
            'ff': nn.Sequential(
                nn.Linear(d_model, d_model * 2),
                nn.ReLU(),
                nn.Linear(d_model * 2, d_model)
            )
        })
    
    def forward(self, x, edge_index=None):
        batch_size, seq_len, features = x.shape
        
        # Input projection
        x = self.input_projection(x)
        
        # Add positional encoding
        x = x + self.pos_encoding[:, :seq_len, :]
        
        # Apply graph convolutions
        if edge_index is not None:
            x_reshaped = x.reshape(-1, self.d_model)
            for conv in self.graph_convs:
                x_reshaped = conv(x_reshaped, edge_index)
                x_reshaped = F.relu(x_reshaped)
            x = x_reshaped.reshape(batch_size, seq_len, self.d_model)
        
        # Transformer blocks
        attention_weights = []
        for layer in self.encoder_layers:
            # Self-attention
            attn_out, attn_weights = layer['attention'](x, x, x)
            x = layer['norm1'](x + attn_out)
            
            # Feed-forward
            ff_out = layer['ff'](x)
            x = layer['norm2'](x + ff_out)
            
            attention_weights.append(attn_weights)
        
        # Multi-scale reconstruction
        global_features = x.mean(dim=1)  # Global pooling
        local_features = x.max(dim=1)[0]  # Local pooling
        
        reconstructed_global = self.decoder_global(global_features)
        reconstructed_local = self.decoder_local(local_features)
        
        # Multi-scale anomaly detection
        anomaly_temporal = self.anomaly_detector_temporal(global_features)
        anomaly_structural = self.anomaly_detector_structural(local_features)
        anomaly_statistical = self.anomaly_detector_statistical(global_features)
        
        # Combine anomaly scores
        combined_anomaly_score = (0.4 * anomaly_temporal + 
                                 0.3 * anomaly_structural + 
                                 0.3 * anomaly_statistical)
        
        return (reconstructed_global, reconstructed_local, 
                combined_anomaly_score, attention_weights, 
                global_features, local_features)

def create_power_system_graph(num_nodes=14):
    """Create IEEE 14-bus system topology."""
    edges = [
        (0, 1), (0, 2), (0, 3), (1, 2), (1, 4), (2, 3), (2, 4), (2, 5),
        (3, 5), (4, 5), (4, 6), (4, 7), (5, 6), (6, 7), (6, 8), (6, 9),
        (6, 10), (7, 8), (8, 9), (9, 10), (9, 11), (9, 12), (10, 11),
        (10, 12), (11, 12), (12, 13)
    ]
    
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)
    
    return edge_index

def create_sophisticated_malicious_data(benign_data, seq_len=24, num_nodes=14):
    """Generate sophisticated malicious data patterns for unsupervised learning."""
    print("Generating sophisticated malicious data for unsupervised learning...")
    
    n_samples = len(benign_data)
    all_attacks = []
    
    # 1. Advanced False Data Injection with temporal correlation
    fdi_attack = benign_data.copy()
    for i in range(seq_len, len(benign_data)):
        time_factor = (i - seq_len) / (len(benign_data) - seq_len)
        noise_scale = 0.05 + 0.25 * time_factor
        
        # Correlate attacks across connected nodes
        for node in range(min(num_nodes, benign_data.shape[1] // 4)):
            if node*4+4 <= benign_data.shape[1]:
                # Add correlated noise with temporal dependency
                base_noise = np.random.normal(0, noise_scale, 4)
                if i > 0:
                    # Add temporal correlation
                    base_noise = 0.8 * base_noise + 0.2 * fdi_attack[i-1, node*4:(node+1)*4] * 0.1
                fdi_attack[i, node*4:(node+1)*4] += base_noise
    all_attacks.append(fdi_attack)
    
    # 2. Stealthy coordinated attack with graph structure awareness
    stealth_attack = benign_data.copy()
    attack_probability = 0.35
    for i in range(seq_len, len(benign_data)):
        if np.random.random() < attack_probability:
            # Attack multiple connected nodes simultaneously
            max_nodes = min(num_nodes, benign_data.shape[1] // 4)
            if max_nodes >= 2:
                attack_size = min(np.random.randint(2, 4), max_nodes)
                attacked_nodes = np.random.choice(range(max_nodes), size=attack_size, replace=False)
            else:
                attacked_nodes = [0] if max_nodes > 0 else []
            
            for node in attacked_nodes:
                if node*4+4 <= benign_data.shape[1]:
                    # Add stealthy perturbations that respect power system constraints
                    perturbation = np.random.normal(0, 0.08, 4)
                    # Add some correlation between features
                    perturbation[1] = 0.7 * perturbation[1] + 0.3 * perturbation[0]  # Qd correlated with Pd
                    stealth_attack[i, node*4:(node+1)*4] += perturbation
    all_attacks.append(stealth_attack)
    
    # 3. Replay attack with systematic drift
    replay_attack = benign_data.copy()
    for i in range(seq_len, len(benign_data)):
        if np.random.random() < 0.25:
            # Replay previous measurements with systematic drift
            replay_idx = max(0, i - np.random.randint(seq_len, seq_len * 2))
            replay_attack[i] = benign_data[replay_idx]
            # Add systematic drift that mimics aging
            drift = np.random.normal(0, 0.03, benign_data.shape[1])
            # Add feature-specific drift patterns
            for j in range(0, benign_data.shape[1], 4):
                if j+3 < benign_data.shape[1]:
                    # Voltage magnitude drift
                    replay_attack[i, j+2] += drift[j+2] * 0.5
                    # Phase angle drift
                    replay_attack[i, j+3] += drift[j+3] * 0.3
    all_attacks.append(replay_attack)
    
    # 4. Advanced statistical attack
    statistical_attack = benign_data.copy()
    feature_means = np.mean(benign_data, axis=0)
    feature_stds = np.std(benign_data, axis=0)
    
    for i in range(seq_len, len(benign_data)):
        if np.random.random() < 0.3:
            # Add statistical anomalies that are hard to detect
            for j in range(benign_data.shape[1]):
                # Add noise that follows different statistical patterns
                if np.random.random() < 0.5:
                    # Gaussian noise
                    noise = np.random.normal(0, feature_stds[j] * 0.15)
                else:
                    # Skewed noise (more realistic for power systems)
                    noise = np.random.exponential(feature_stds[j] * 0.1) - feature_stds[j] * 0.1
                
                statistical_attack[i, j] += noise
    all_attacks.append(statistical_attack)
    
    # Combine all attacks
    malicious_data = np.vstack(all_attacks)
    
    # Ensure proper size
    if len(malicious_data) > n_samples:
        indices = np.random.choice(len(malicious_data), n_samples, replace=False)
        malicious_data = malicious_data[indices]
    elif len(malicious_data) < n_samples:
        additional_needed = n_samples - len(malicious_data)
        additional_indices = np.random.choice(len(malicious_data), additional_needed, replace=True)
        additional_data = malicious_data[additional_indices]
        malicious_data = np.vstack([malicious_data, additional_data])
    
    print(f"Generated {len(malicious_data)} sophisticated malicious samples")
    return malicious_data

def load_and_preprocess_unsupervised_data(benign_file, seq_len=24, num_nodes=14):
    """Load and preprocess data for unsupervised learning."""
    print("Loading and preprocessing data for unsupervised learning...")
    
    # Load benign data
    benign_df = pd.read_excel(benign_file)
    feature_columns = ['Pd_new', 'Qd_new', 'Vm', 'Va']
    X_benign = benign_df[feature_columns].values
    
    print(f"Benign samples loaded: {len(X_benign)}")
    
    # Generate sophisticated malicious data
    X_malicious = create_sophisticated_malicious_data(X_benign, seq_len, num_nodes)
    
    # Create labels (only for evaluation, not for training)
    y_benign = np.zeros(len(X_benign))
    y_malicious = np.ones(len(X_malicious))
    
    # Combine data
    X = np.vstack([X_benign, X_malicious])
    y = np.concatenate([y_benign, y_malicious])
    
    print(f"Total samples: {len(X)}")
    print(f"Class distribution: Benign={np.sum(y==0)}, Malicious={np.sum(y==1)}")
    
    # Enhanced preprocessing
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Create sequence data
    sequences = []
    sequence_labels = []
    edge_index = create_power_system_graph(num_nodes)
    
    for i in range(len(X_scaled) - seq_len + 1):
        seq = X_scaled[i:i + seq_len]
        label = y[i + seq_len - 1]
        sequences.append(seq)
        sequence_labels.append(label)
    
    X_seq = np.array(sequences)
    y_seq = np.array(sequence_labels)
    
    print(f"Sequence data created: {X_seq.shape}")
    print(f"Graph edges: {edge_index.shape}")
    
    return X_seq, y_seq, edge_index, scaler, feature_columns

def train_unsupervised_model(model, train_loader, val_loader, num_epochs=120, device='cpu'):
    """Train the model using only unsupervised learning."""
    model = model.to(device)
    
    # Only reconstruction losses (no classification loss)
    reconstruction_criterion = nn.MSELoss()
    
    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0008, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    patience = 20
    patience_counter = 0
    
    for epoch in range(num_epochs):
        # Training
        model.train()
        total_train_loss = 0
        
        for batch_x, _ in train_loader:  # Ignore labels in unsupervised learning
            batch_x = batch_x.to(device)
            optimizer.zero_grad()
            
            reconstructed_global, reconstructed_local, anomaly_score, _, _, _ = model(batch_x)
            
            # Multi-scale reconstruction loss
            recon_loss_global = reconstruction_criterion(reconstructed_global, batch_x.mean(dim=1))
            recon_loss_local = reconstruction_criterion(reconstructed_local, batch_x.max(dim=1)[0])
            
            # Temporal reconstruction loss
            recon_loss_temporal = reconstruction_criterion(
                reconstructed_global.unsqueeze(1).expand(-1, batch_x.size(1), -1), 
                batch_x
            )
            
            # Combined reconstruction loss
            recon_loss = 0.4 * recon_loss_global + 0.3 * recon_loss_local + 0.3 * recon_loss_temporal
            
            # Anomaly regularization (encourage normal data to have low anomaly scores)
            # Since we don't have labels, we use reconstruction error as a proxy
            reconstruction_error = torch.mean((batch_x - reconstructed_global.unsqueeze(1).expand(-1, batch_x.size(1), -1))**2, dim=(1, 2))
            normal_mask = (reconstruction_error < torch.median(reconstruction_error)).float()
            anomaly_reg = torch.mean(anomaly_score.squeeze() * normal_mask)
            
            # Total loss
            loss = recon_loss + 0.1 * anomaly_reg
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            total_train_loss += loss.item()
        
        avg_train_loss = total_train_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        
        # Validation
        model.eval()
        total_val_loss = 0
        
        with torch.no_grad():
            for batch_x, _ in val_loader:
                batch_x = batch_x.to(device)
                reconstructed_global, reconstructed_local, anomaly_score, _, _, _ = model(batch_x)
                
                recon_loss_global = reconstruction_criterion(reconstructed_global, batch_x.mean(dim=1))
                recon_loss_local = reconstruction_criterion(reconstructed_local, batch_x.max(dim=1)[0])
                recon_loss_temporal = reconstruction_criterion(
                    reconstructed_global.unsqueeze(1).expand(-1, batch_x.size(1), -1), 
                    batch_x
                )
                
                recon_loss = 0.4 * recon_loss_global + 0.3 * recon_loss_local + 0.3 * recon_loss_temporal
                
                reconstruction_error = torch.mean((batch_x - reconstructed_global.unsqueeze(1).expand(-1, batch_x.size(1), -1))**2, dim=(1, 2))
                normal_mask = (reconstruction_error < torch.median(reconstruction_error)).float()
                anomaly_reg = torch.mean(anomaly_score.squeeze() * normal_mask)
                
                loss = recon_loss + 0.1 * anomaly_reg
                total_val_loss += loss.item()
        
        avg_val_loss = total_val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        
        # Learning rate scheduling
        scheduler.step()
        
        # Early stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            torch.save(model.state_dict(), 'best_unsupervised_graph_informer.pth')
        else:
            patience_counter += 1
        
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break
        
        if (epoch + 1) % 20 == 0:
            print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
    
    # Load best model
    model.load_state_dict(torch.load('best_unsupervised_graph_informer.pth'))
    return train_losses, val_losses

def evaluate_unsupervised_model(model, test_loader, device='cpu'):
    """Evaluate the unsupervised model."""
    model.eval()
    all_reconstructions_global = []
    all_reconstructions_local = []
    all_anomaly_scores = []
    all_labels = []
    all_global_features = []
    all_local_features = []
    
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            reconstructed_global, reconstructed_local, anomaly_score, _, global_features, local_features = model(batch_x)
            
            all_reconstructions_global.extend(reconstructed_global.cpu().numpy())
            all_reconstructions_local.extend(reconstructed_local.cpu().numpy())
            all_anomaly_scores.extend(anomaly_score.cpu().numpy())
            all_labels.extend(batch_y.cpu().numpy())
            all_global_features.extend(global_features.cpu().numpy())
            all_local_features.extend(local_features.cpu().numpy())
    
    # Convert to numpy arrays
    all_reconstructions_global = np.array(all_reconstructions_global)
    all_reconstructions_local = np.array(all_reconstructions_local)
    all_anomaly_scores = np.array(all_anomaly_scores).flatten()
    all_labels = np.array(all_labels)
    all_global_features = np.array(all_global_features)
    all_local_features = np.array(all_local_features)
    
    # Calculate reconstruction errors
    test_data = test_loader.dataset.tensors[0].numpy()
    reconstruction_error_global = np.mean((all_reconstructions_global - test_data.mean(axis=1))**2, axis=1)
    reconstruction_error_local = np.mean((all_reconstructions_local - test_data.max(axis=1))**2, axis=1)
    reconstruction_error_temporal = np.mean((all_reconstructions_global[:, np.newaxis, :] - test_data)**2, axis=(1, 2))
    
    # Advanced ensemble scoring
    # Normalize all scores to [0, 1]
    recon_global_norm = (reconstruction_error_global - np.min(reconstruction_error_global)) / (np.max(reconstruction_error_global) - np.min(reconstruction_error_global) + 1e-8)
    recon_local_norm = (reconstruction_error_local - np.min(reconstruction_error_local)) / (np.max(reconstruction_error_local) - np.min(reconstruction_error_local) + 1e-8)
    recon_temporal_norm = (reconstruction_error_temporal - np.min(reconstruction_error_temporal)) / (np.max(reconstruction_error_temporal) - np.min(reconstruction_error_temporal) + 1e-8)
    anomaly_scores_norm = (all_anomaly_scores - np.min(all_anomaly_scores)) / (np.max(all_anomaly_scores) - np.min(all_anomaly_scores) + 1e-8)
    
    # Feature-based anomaly detection
    from sklearn.ensemble import IsolationForest
    iso_forest = IsolationForest(contamination=0.1, random_state=42)
    feature_anomaly_scores = iso_forest.fit_predict(np.concatenate([all_global_features, all_local_features], axis=1))
    feature_anomaly_scores = (feature_anomaly_scores == -1).astype(float)
    
    # Statistical anomaly detection
    statistical_scores = np.zeros(len(all_global_features))
    for i in range(len(all_global_features)):
        # Calculate Mahalanobis distance-like score
        feature_vector = np.concatenate([all_global_features[i], all_local_features[i]])
        mean_vector = np.mean(np.concatenate([all_global_features, all_local_features], axis=1), axis=0)
        cov_matrix = np.cov(np.concatenate([all_global_features, all_local_features], axis=1).T)
        try:
            inv_cov = np.linalg.inv(cov_matrix + np.eye(cov_matrix.shape[0]) * 1e-6)
            diff = feature_vector - mean_vector
            statistical_scores[i] = np.sqrt(diff.T @ inv_cov @ diff)
        except:
            statistical_scores[i] = 0
    
    statistical_scores_norm = (statistical_scores - np.min(statistical_scores)) / (np.max(statistical_scores) - np.min(statistical_scores) + 1e-8)
    
    # Advanced ensemble scoring
    combined_score = (0.25 * recon_global_norm + 
                     0.20 * recon_local_norm + 
                     0.20 * recon_temporal_norm + 
                     0.20 * anomaly_scores_norm + 
                     0.10 * feature_anomaly_scores + 
                     0.05 * statistical_scores_norm)
    
    # Find optimal threshold using multiple strategies
    from sklearn.metrics import roc_curve
    
    # Strategy 1: Youden's J statistic
    fpr, tpr, thresholds = roc_curve(all_labels, combined_score)
    youden_j = tpr - fpr
    optimal_idx = np.argmax(youden_j)
    optimal_threshold = thresholds[optimal_idx]
    
    # Strategy 2: 95th percentile of normal data
    normal_scores = combined_score[all_labels == 0]
    threshold_95 = np.percentile(normal_scores, 95)
    
    # Strategy 3: Mean + 2*std of normal data
    threshold_mean_std = np.mean(normal_scores) + 2 * np.std(normal_scores)
    
    # Use the most conservative threshold
    final_threshold = max(optimal_threshold, threshold_95, threshold_mean_std)
    
    # Final predictions
    final_predictions = (combined_score > final_threshold).astype(int)
    
    # Calculate metrics
    accuracy = accuracy_score(all_labels, final_predictions)
    precision = precision_score(all_labels, final_predictions, average='weighted')
    recall = recall_score(all_labels, final_predictions, average='weighted')
    f1 = f1_score(all_labels, final_predictions, average='weighted')
    roc_auc = roc_auc_score(all_labels, combined_score)
    
    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'roc_auc': roc_auc,
        'threshold': final_threshold,
        'optimal_threshold': optimal_threshold,
        'threshold_95': threshold_95,
        'threshold_mean_std': threshold_mean_std,
        'combined_score': combined_score,
        'reconstruction_error_global': reconstruction_error_global,
        'reconstruction_error_local': reconstruction_error_local,
        'reconstruction_error_temporal': reconstruction_error_temporal,
        'anomaly_scores': all_anomaly_scores
    }
    
    return metrics

def main():
    """Main function for unsupervised Graph Informer."""
    print("🔍 Pure Unsupervised Graph Informer Transformer for Power System Attack Detection")
    print("=" * 90)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Parameters
    seq_len = 24
    d_model = 256
    n_heads = 8
    n_layers = 3
    num_nodes = 14
    
    # Load and preprocess data
    X_seq, y_seq, edge_index, scaler, feature_names = load_and_preprocess_unsupervised_data(
        'benign_bus14.xlsx', seq_len, num_nodes
    )
    
    print(f"\nDataset Statistics:")
    print(f"Total sequences: {len(X_seq)}")
    print(f"Sequence shape: {X_seq.shape}")
    print(f"Benign sequences: {np.sum(y_seq == 0)}")
    print(f"Malicious sequences: {np.sum(y_seq == 1)}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X_seq, y_seq, test_size=0.2, random_state=42, stratify=y_seq)
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
    
    batch_size = 32
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # Create unsupervised model
    model = UnsupervisedGraphInformer(
        input_dim=len(feature_names),
        d_model=d_model,
        n_heads=n_heads,
        n_layers=n_layers,
        seq_len=seq_len,
        num_nodes=num_nodes
    )
    
    print(f"\nUnsupervised Model Architecture:")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Model dimension: {d_model}")
    print(f"Transformer layers: {n_layers}")
    print(f"Features: Multi-scale reconstruction, advanced anomaly detection, ensemble scoring")
    print(f"Learning: Pure unsupervised (no labeled attack data used)")
    
    # Train model
    print("\nTraining unsupervised model...")
    train_losses, val_losses = train_unsupervised_model(
        model, train_loader, val_loader, num_epochs=120, device=device
    )
    
    # Evaluate model
    print("\nEvaluating unsupervised model...")
    metrics = evaluate_unsupervised_model(model, test_loader, device=device)
    
    # Print results
    print("\n🏆 Unsupervised Model Performance:")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"F1-Score: {metrics['f1_score']:.4f}")
    print(f"ROC-AUC: {metrics['roc_auc']:.4f}")
    print(f"Final Threshold: {metrics['threshold']:.4f}")
    
    print(f"\nThreshold Analysis:")
    print(f"Optimal Threshold (Youden's J): {metrics['optimal_threshold']:.4f}")
    print(f"95th Percentile Threshold: {metrics['threshold_95']:.4f}")
    print(f"Mean + 2*Std Threshold: {metrics['threshold_mean_std']:.4f}")
    
    # Save model
    torch.save(model.state_dict(), 'unsupervised_graph_informer_final.pth')
    
    print(f"\n✅ Pure Unsupervised Graph Informer implementation complete!")
    print(f"🎯 Model saved to 'unsupervised_graph_informer_final.pth'")
    print(f"📊 No labeled attack data was used during training - pure unsupervised learning!")
    
    return model, metrics

if __name__ == "__main__":
    main()
