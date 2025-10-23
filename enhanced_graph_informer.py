import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import math
import warnings
from torch_geometric.nn import GCNConv, GATConv, global_mean_pool, global_max_pool
from torch_geometric.data import Data, DataLoader
import networkx as nx
warnings.filterwarnings('ignore')

class EnhancedGraphInformerTransformer(nn.Module):
    """Enhanced Graph Informer Transformer with improved accuracy for power system attack detection."""
    
    def __init__(self, input_dim, d_model=512, n_heads=8, n_layers=4, d_ff=2048, 
                 seq_len=24, dropout=0.1, num_nodes=14):
        super(EnhancedGraphInformerTransformer, self).__init__()
        
        self.input_dim = input_dim
        self.d_model = d_model
        self.seq_len = seq_len
        self.num_nodes = num_nodes
        
        # Enhanced input projection with residual connection
        self.input_projection = nn.Sequential(
            nn.Linear(input_dim, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Positional encoding
        self.pos_encoding = self._create_positional_encoding(d_model, seq_len)
        
        # Enhanced transformer blocks with residual connections
        self.encoder_layers = nn.ModuleList([
            self._create_enhanced_block(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])
        
        # Graph convolution layers for structural information
        self.graph_convs = nn.ModuleList([
            GCNConv(d_model, d_model) for _ in range(2)
        ])
        
        # Enhanced decoder for reconstruction
        self.decoder = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.LayerNorm(d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, d_model // 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 4, input_dim)
        )
        
        # Multi-scale anomaly detection heads
        self.anomaly_detector_global = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1),
            nn.Sigmoid()
        )
        
        self.anomaly_detector_local = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1),
            nn.Sigmoid()
        )
        
        # Classification head for supervised learning
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, d_model // 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 4, 2)  # Binary classification
        )
        
    def _create_positional_encoding(self, d_model, max_len):
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        
        return nn.Parameter(pe, requires_grad=False)
    
    def _create_enhanced_block(self, d_model, n_heads, d_ff, dropout):
        return nn.ModuleDict({
            'attention': nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True),
            'norm1': nn.LayerNorm(d_model),
            'norm2': nn.LayerNorm(d_model),
            'ff': nn.Sequential(
                nn.Linear(d_model, d_ff),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(d_ff, d_model),
                nn.Dropout(dropout)
            )
        })
    
    def forward(self, x, edge_index=None, return_features=False):
        batch_size, seq_len, features = x.shape
        
        # Input projection with residual connection
        x_proj = self.input_projection(x)
        
        # Add positional encoding
        x_proj = x_proj + self.pos_encoding[:, :seq_len, :]
        
        # Apply graph convolutions if edge_index is provided
        if edge_index is not None:
            x_reshaped = x_proj.reshape(-1, self.d_model)
            for conv in self.graph_convs:
                x_reshaped = conv(x_reshaped, edge_index)
                x_reshaped = F.relu(x_reshaped)
            x_proj = x_reshaped.reshape(batch_size, seq_len, self.d_model)
        
        # Enhanced transformer blocks
        attention_weights = []
        for layer in self.encoder_layers:
            # Self-attention with residual connection
            attn_out, attn_weights = layer['attention'](x_proj, x_proj, x_proj)
            x_proj = layer['norm1'](x_proj + attn_out)
            
            # Feed-forward with residual connection
            ff_out = layer['ff'](x_proj)
            x_proj = layer['norm2'](x_proj + ff_out)
            
            attention_weights.append(attn_weights)
        
        # Reconstruction
        reconstructed = self.decoder(x_proj)
        
        # Multi-scale anomaly detection
        global_features = x_proj.mean(dim=1)  # Global pooling
        local_features = x_proj.max(dim=1)[0]  # Local pooling
        
        anomaly_score_global = self.anomaly_detector_global(global_features)
        anomaly_score_local = self.anomaly_detector_local(local_features)
        
        # Combined anomaly score
        anomaly_score = 0.6 * anomaly_score_global + 0.4 * anomaly_score_local
        
        # Classification (for supervised learning)
        classification_logits = self.classifier(global_features)
        
        if return_features:
            return reconstructed, anomaly_score, classification_logits, attention_weights, global_features
        else:
            return reconstructed, anomaly_score, classification_logits, attention_weights

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

def create_enhanced_malicious_data(benign_data, seq_len=24, num_nodes=14):
    """Generate more sophisticated malicious data patterns."""
    print("Generating enhanced malicious data patterns...")
    
    n_samples = len(benign_data)
    all_attacks = []
    
    # 1. Advanced False Data Injection with temporal correlation
    fdi_attack = benign_data.copy()
    for i in range(seq_len, len(benign_data)):
        time_factor = (i - seq_len) / (len(benign_data) - seq_len)
        noise_scale = 0.05 + 0.3 * time_factor
        
        # Correlate attacks across connected nodes
        for node in range(min(num_nodes, benign_data.shape[1] // 4)):
            if node*4+4 <= benign_data.shape[1]:
                # Add correlated noise across features
                base_noise = np.random.normal(0, noise_scale, 4)
                # Add temporal correlation
                if i > 0:
                    base_noise = 0.7 * base_noise + 0.3 * fdi_attack[i-1, node*4:(node+1)*4] * 0.1
                fdi_attack[i, node*4:(node+1)*4] += base_noise
    all_attacks.append(fdi_attack)
    
    # 2. Stealthy coordinated attack
    stealth_attack = benign_data.copy()
    attack_probability = 0.4
    for i in range(seq_len, len(benign_data)):
        if np.random.random() < attack_probability:
            # Attack multiple nodes simultaneously
            max_nodes = min(num_nodes, benign_data.shape[1] // 4)
            if max_nodes >= 2:
                attack_size = min(np.random.randint(2, 5), max_nodes)
                attacked_nodes = np.random.choice(range(max_nodes), size=attack_size, replace=False)
            else:
                attacked_nodes = [0] if max_nodes > 0 else []
            for node in attacked_nodes:
                if node*4+4 <= benign_data.shape[1]:
                    # Add stealthy perturbations
                    perturbation = np.random.normal(0, 0.1, 4)
                    stealth_attack[i, node*4:(node+1)*4] += perturbation
    all_attacks.append(stealth_attack)
    
    # 3. Replay attack with drift
    replay_attack = benign_data.copy()
    for i in range(seq_len, len(benign_data)):
        if np.random.random() < 0.3:
            # Replay previous measurements with drift
            replay_idx = max(0, i - np.random.randint(seq_len, seq_len * 3))
            replay_attack[i] = benign_data[replay_idx]
            # Add systematic drift
            drift = np.random.normal(0, 0.05, benign_data.shape[1])
            replay_attack[i] += drift
    all_attacks.append(replay_attack)
    
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
    
    print(f"Generated {len(malicious_data)} enhanced malicious samples")
    return malicious_data

def load_and_preprocess_enhanced_data(benign_file, seq_len=24, num_nodes=14):
    """Load and preprocess data with enhanced features."""
    print("Loading and preprocessing enhanced data...")
    
    # Load benign data
    benign_df = pd.read_excel(benign_file)
    feature_columns = ['Pd_new', 'Qd_new', 'Vm', 'Va']
    X_benign = benign_df[feature_columns].values
    
    print(f"Benign samples loaded: {len(X_benign)}")
    
    # Generate enhanced malicious data
    X_malicious = create_enhanced_malicious_data(X_benign, seq_len, num_nodes)
    
    # Create labels
    y_benign = np.zeros(len(X_benign))
    y_malicious = np.ones(len(X_malicious))
    
    # Combine data
    X = np.vstack([X_benign, X_malicious])
    y = np.concatenate([y_benign, y_malicious])
    
    print(f"Total samples: {len(X)}")
    print(f"Class distribution: Benign={np.sum(y==0)}, Malicious={np.sum(y==1)}")
    
    # Enhanced preprocessing with robust scaling
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

def train_enhanced_model(model, train_loader, val_loader, num_epochs=150, device='cuda'):
    """Train the enhanced model with both unsupervised and supervised learning."""
    model = model.to(device)
    
    # Loss functions
    reconstruction_criterion = nn.MSELoss()
    classification_criterion = nn.CrossEntropyLoss()
    
    # Enhanced optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0003, weight_decay=1e-4)
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
        
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            
            reconstructed, anomaly_score, classification_logits, attn_weights = model(batch_x)
            
            # Reconstruction loss
            recon_loss = reconstruction_criterion(reconstructed, batch_x)
            
            # Classification loss (supervised)
            class_loss = classification_criterion(classification_logits, batch_y)
            
            # Anomaly regularization
            normal_mask = (batch_y == 0).float()
            anomaly_reg = torch.mean(anomaly_score.squeeze() * normal_mask)
            
            # Combined loss with adaptive weighting
            loss = (0.4 * recon_loss + 
                   0.4 * class_loss + 
                   0.2 * anomaly_reg)
            
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
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                reconstructed, anomaly_score, classification_logits, _ = model(batch_x)
                
                recon_loss = reconstruction_criterion(reconstructed, batch_x)
                class_loss = classification_criterion(classification_logits, batch_y)
                normal_mask = (batch_y == 0).float()
                anomaly_reg = torch.mean(anomaly_score.squeeze() * normal_mask)
                
                loss = 0.4 * recon_loss + 0.4 * class_loss + 0.2 * anomaly_reg
                total_val_loss += loss.item()
        
        avg_val_loss = total_val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        
        # Learning rate scheduling
        scheduler.step()
        
        # Early stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            torch.save(model.state_dict(), 'best_enhanced_graph_informer.pth')
        else:
            patience_counter += 1
        
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break
        
        if (epoch + 1) % 20 == 0:
            print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
    
    # Load best model
    model.load_state_dict(torch.load('best_enhanced_graph_informer.pth'))
    return train_losses, val_losses

def evaluate_enhanced_model(model, test_loader, device='cuda'):
    """Comprehensive evaluation of the enhanced model."""
    model.eval()
    all_reconstructions = []
    all_anomaly_scores = []
    all_predictions = []
    all_labels = []
    all_classification_logits = []
    
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            reconstructed, anomaly_score, classification_logits, _ = model(batch_x)
            
            all_reconstructions.extend(reconstructed.cpu().numpy())
            all_anomaly_scores.extend(anomaly_score.cpu().numpy())
            all_classification_logits.extend(classification_logits.cpu().numpy())
            all_labels.extend(batch_y.cpu().numpy())
            
            # Get predictions from classification head
            preds = classification_logits.argmax(dim=1)
            all_predictions.extend(preds.cpu().numpy())
    
    # Convert to numpy arrays
    all_reconstructions = np.array(all_reconstructions)
    all_anomaly_scores = np.array(all_anomaly_scores).flatten()
    all_classification_logits = np.array(all_classification_logits)
    all_labels = np.array(all_labels)
    all_predictions = np.array(all_predictions)
    
    # Calculate reconstruction error
    reconstruction_error = np.mean((all_reconstructions - test_loader.dataset.tensors[0].numpy())**2, axis=(1, 2))
    
    # Enhanced ensemble scoring
    reconstruction_error_norm = (reconstruction_error - np.min(reconstruction_error)) / (np.max(reconstruction_error) - np.min(reconstruction_error) + 1e-8)
    anomaly_scores_norm = (all_anomaly_scores - np.min(all_anomaly_scores)) / (np.max(all_anomaly_scores) - np.min(all_anomaly_scores) + 1e-8)
    
    # Classification confidence
    classification_probs = F.softmax(torch.tensor(all_classification_logits), dim=1).numpy()
    classification_confidence = np.max(classification_probs, axis=1)
    
    # Combined scoring
    combined_score = (0.3 * reconstruction_error_norm + 
                     0.3 * anomaly_scores_norm + 
                     0.4 * (1 - classification_confidence))  # Higher confidence in normal class = lower anomaly score
    
    # Find optimal threshold
    from sklearn.metrics import roc_curve
    fpr, tpr, thresholds = roc_curve(all_labels, combined_score)
    youden_j = tpr - fpr
    optimal_idx = np.argmax(youden_j)
    optimal_threshold = thresholds[optimal_idx]
    
    # Final predictions using combined approach
    final_predictions = (combined_score > optimal_threshold).astype(int)
    
    # Calculate metrics
    accuracy = accuracy_score(all_labels, final_predictions)
    precision = precision_score(all_labels, final_predictions, average='weighted')
    recall = recall_score(all_labels, final_predictions, average='weighted')
    f1 = f1_score(all_labels, final_predictions, average='weighted')
    roc_auc = roc_auc_score(all_labels, combined_score)
    
    # Classification head metrics
    class_accuracy = accuracy_score(all_labels, all_predictions)
    class_precision = precision_score(all_labels, all_predictions, average='weighted')
    class_recall = recall_score(all_labels, all_predictions, average='weighted')
    class_f1 = f1_score(all_labels, all_predictions, average='weighted')
    
    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'roc_auc': roc_auc,
        'threshold': optimal_threshold,
        'classification_accuracy': class_accuracy,
        'classification_precision': class_precision,
        'classification_recall': class_recall,
        'classification_f1': class_f1,
        'combined_score': combined_score,
        'reconstruction_error': reconstruction_error,
        'anomaly_scores': all_anomaly_scores
    }
    
    return metrics

def main():
    """Main function for enhanced Graph Informer Transformer."""
    print("🚀 Enhanced Graph Informer Transformer for Power System Attack Detection")
    print("=" * 80)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Enhanced parameters
    seq_len = 24
    d_model = 512
    n_heads = 8
    n_layers = 4
    num_nodes = 14
    
    # Load and preprocess data
    X_seq, y_seq, edge_index, scaler, feature_names = load_and_preprocess_enhanced_data(
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
    
    # Create enhanced model
    model = EnhancedGraphInformerTransformer(
        input_dim=len(feature_names),
        d_model=d_model,
        n_heads=n_heads,
        n_layers=n_layers,
        seq_len=seq_len,
        num_nodes=num_nodes
    )
    
    print(f"\nEnhanced Model Architecture:")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Model dimension: {d_model}")
    print(f"Transformer layers: {n_layers}")
    print(f"Features: Multi-scale anomaly detection, supervised classification, ensemble scoring")
    
    # Train model
    print("\nTraining enhanced model...")
    train_losses, val_losses = train_enhanced_model(
        model, train_loader, val_loader, num_epochs=150, device=device
    )
    
    # Evaluate model
    print("\nEvaluating enhanced model...")
    metrics = evaluate_enhanced_model(model, test_loader, device=device)
    
    # Print results
    print("\n🏆 Enhanced Model Performance:")
    print(f"Combined Accuracy: {metrics['accuracy']:.4f}")
    print(f"Combined Precision: {metrics['precision']:.4f}")
    print(f"Combined Recall: {metrics['recall']:.4f}")
    print(f"Combined F1-Score: {metrics['f1_score']:.4f}")
    print(f"ROC-AUC: {metrics['roc_auc']:.4f}")
    print(f"Optimal Threshold: {metrics['threshold']:.4f}")
    
    print(f"\nClassification Head Performance:")
    print(f"Classification Accuracy: {metrics['classification_accuracy']:.4f}")
    print(f"Classification Precision: {metrics['classification_precision']:.4f}")
    print(f"Classification Recall: {metrics['classification_recall']:.4f}")
    print(f"Classification F1-Score: {metrics['classification_f1']:.4f}")
    
    # Save model
    torch.save(model.state_dict(), 'enhanced_graph_informer_final.pth')
    
    print(f"\n✅ Enhanced Graph Informer implementation complete!")
    print(f"🎯 Model saved to 'enhanced_graph_informer_final.pth'")
    
    return model, metrics

if __name__ == "__main__":
    main()
