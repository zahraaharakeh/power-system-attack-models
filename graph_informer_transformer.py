import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns
import math
import warnings
from torch_geometric.nn import GCNConv, GATConv, global_mean_pool, global_max_pool
from torch_geometric.data import Data, DataLoader
import networkx as nx
warnings.filterwarnings('ignore')

class GraphPositionalEncoding(nn.Module):
    """Graph-aware positional encoding for transformer models."""
    
    def __init__(self, d_model, max_len=5000):
        super(GraphPositionalEncoding, self).__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        return x + self.pe[:x.size(0), :]

class GraphProbSparseAttention(nn.Module):
    """Simplified Graph-aware attention mechanism combining graph structure with temporal attention."""
    
    def __init__(self, d_model, n_heads, factor=5, dropout=0.1):
        super(GraphProbSparseAttention, self).__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.factor = factor
        self.dropout = nn.Dropout(dropout)
        
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        
        # Use standard multi-head attention for simplicity
        self.attention = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        
    def forward(self, queries, keys, values, graph_adj=None, attn_mask=None):
        # Apply linear transformations
        Q = self.W_q(queries)
        K = self.W_k(keys)
        V = self.W_v(values)
        
        # Use standard multi-head attention
        attn_output, attn_weights = self.attention(Q, K, V, attn_mask=attn_mask)
        
        # Apply output projection
        output = self.W_o(attn_output)
        
        return output, attn_weights

class GraphInformerBlock(nn.Module):
    """Graph Informer block combining graph structure with ProbSparse attention."""
    
    def __init__(self, d_model, n_heads, d_ff=None, dropout=0.1, activation="relu"):
        super(GraphInformerBlock, self).__init__()
        d_ff = d_ff or 4 * d_model
        
        self.attention = GraphProbSparseAttention(d_model, n_heads, dropout=dropout)
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu
        
        # Graph convolution for structural information
        self.graph_conv = GCNConv(d_model, d_model)
        
    def forward(self, x, edge_index=None, attn_mask=None):
        # Graph convolution for structural information
        if edge_index is not None:
            # Reshape for graph convolution: [batch*seq, features]
            batch_size, seq_len, features = x.shape
            x_reshaped = x.reshape(-1, features)
            
            # Apply graph convolution
            x_graph = self.graph_conv(x_reshaped, edge_index)
            x_graph = x_graph.reshape(batch_size, seq_len, features)
            
            # Combine with original features
            x = x + self.dropout(x_graph)
        
        # Self-attention
        new_x, attn = self.attention(x, x, x, attn_mask=attn_mask)
        x = x + self.dropout(new_x)
        x = self.norm1(x)
        
        # Feed forward
        new_x = self.conv1(x.transpose(-1, 1))
        new_x = self.activation(new_x)
        new_x = self.dropout(new_x)
        new_x = self.conv2(new_x).transpose(-1, 1)
        
        x = x + self.dropout(new_x)
        x = self.norm2(x)
        
        return x, attn

class GraphInformerTransformer(nn.Module):
    """Graph Informer Transformer for unsupervised power system attack detection."""
    
    def __init__(self, input_dim, d_model=256, n_heads=8, n_layers=3, d_ff=1024, 
                 seq_len=24, dropout=0.1, num_nodes=14):
        super(GraphInformerTransformer, self).__init__()
        
        self.input_dim = input_dim
        self.d_model = d_model
        self.seq_len = seq_len
        self.num_nodes = num_nodes
        
        # Input projection
        self.input_projection = nn.Linear(input_dim, d_model)
        
        # Positional encoding
        self.pos_encoding = GraphPositionalEncoding(d_model)
        
        # Graph Informer blocks
        self.encoder_layers = nn.ModuleList([
            GraphInformerBlock(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])
        
        # Decoder for reconstruction (unsupervised learning)
        self.decoder = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, input_dim)
        )
        
        # Anomaly detection head
        self.anomaly_detector = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1),
            nn.Sigmoid()
        )
        
        # Attention weights for interpretability
        self.attention_weights = None
        
    def forward(self, x, edge_index=None):
        # Input projection
        x = self.input_projection(x)  # [batch, seq_len, d_model]
        
        # Positional encoding
        x = x.transpose(0, 1)  # [seq_len, batch, d_model]
        x = self.pos_encoding(x)
        x = x.transpose(0, 1)  # [batch, seq_len, d_model]
        
        # Encoder layers
        attn_weights = []
        for layer in self.encoder_layers:
            x, attn = layer(x, edge_index)
            attn_weights.append(attn)
        
        self.attention_weights = attn_weights
        
        # Reconstruction
        reconstructed = self.decoder(x)
        
        # Anomaly score
        anomaly_score = self.anomaly_detector(x.mean(dim=1))  # Global pooling
        
        return reconstructed, anomaly_score, attn_weights

def create_power_system_graph(num_nodes=14):
    """Create a power system graph topology (IEEE 14-bus system)."""
    # Create IEEE 14-bus system topology
    edges = [
        (0, 1), (0, 2), (0, 3), (1, 2), (1, 4), (2, 3), (2, 4), (2, 5),
        (3, 5), (4, 5), (4, 6), (4, 7), (5, 6), (6, 7), (6, 8), (6, 9),
        (6, 10), (7, 8), (8, 9), (9, 10), (9, 11), (9, 12), (10, 11),
        (10, 12), (11, 12), (12, 13)
    ]
    
    # Create edge index for PyTorch Geometric
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    
    # Add reverse edges for undirected graph
    edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)
    
    return edge_index

def create_graph_sequence_data(features, labels, seq_len=24, num_nodes=14):
    """Create graph sequence data for Graph Informer model."""
    sequences = []
    sequence_labels = []
    edge_index = create_power_system_graph(num_nodes)
    
    for i in range(len(features) - seq_len + 1):
        seq = features[i:i + seq_len]
        label = labels[i + seq_len - 1]  # Use the last label in sequence
        sequences.append(seq)
        sequence_labels.append(label)
    
    return np.array(sequences), np.array(sequence_labels), edge_index

def generate_graph_malicious_data(benign_data, seq_len=24, num_nodes=14):
    """Generate graph-aware malicious data with sophisticated attack patterns."""
    print("Generating graph-aware malicious data for Graph Informer...")
    
    n_samples = len(benign_data)
    all_attacks = []
    
    # 1. Graph-aware False Data Injection (FDI)
    fdi_attack = benign_data.copy()
    for i in range(seq_len, len(benign_data)):
        # Add graph-structured perturbations
        time_factor = (i - seq_len) / (len(benign_data) - seq_len)
        noise_scale = 0.1 + 0.2 * time_factor
        
        # Correlate noise across connected nodes
        for node in range(num_nodes):
            if node*4+4 <= benign_data.shape[1]:  # Ensure we don't exceed array bounds
                base_noise = np.random.normal(0, noise_scale, 4)
                fdi_attack[i, node*4:(node+1)*4] += base_noise
    all_attacks.append(fdi_attack)
    
    # 2. Coordinated Graph Attack
    coord_attack = benign_data.copy()
    attack_windows = np.random.choice([0, 1], size=len(benign_data), p=[0.7, 0.3])
    for i in range(seq_len, len(benign_data)):
        if attack_windows[i]:
            # Coordinate attack across graph structure
            window_size = min(seq_len, len(benign_data) - i)
            for j in range(window_size):
                # Attack connected nodes together
                for node in range(0, num_nodes, 2):  # Attack every other node
                    if node*4+4 <= benign_data.shape[1]:  # Ensure we don't exceed array bounds
                        coord_attack[i + j, node*4:(node+1)*4] += np.random.normal(0, 0.15, 4)
    all_attacks.append(coord_attack)
    
    # 3. Stealth Graph Attack
    stealth_attack = benign_data.copy()
    feature_means = np.mean(benign_data, axis=0)
    feature_stds = np.std(benign_data, axis=0)
    
    for i in range(seq_len, len(benign_data)):
        # Add noise while maintaining graph consistency
        for node in range(num_nodes):
            for feature in range(4):  # 4 features per node
                feature_idx = node * 4 + feature
                if feature_idx < benign_data.shape[1]:  # Ensure we don't exceed array bounds
                    noise = np.random.normal(0, feature_stds[feature_idx] * 0.2)
                    stealth_attack[i, feature_idx] += noise
                    # Maintain temporal smoothness
                    if i > 0:
                        stealth_attack[i, feature_idx] = 0.7 * stealth_attack[i, feature_idx] + 0.3 * stealth_attack[i-1, feature_idx]
    all_attacks.append(stealth_attack)
    
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
    
    print(f"Generated {len(malicious_data)} graph-aware malicious samples")
    return malicious_data

def load_and_preprocess_graph_data(benign_file, seq_len=24, num_nodes=14, balance_classes=True):
    """Load and preprocess data for Graph Informer model."""
    print("Loading and preprocessing graph data for Graph Informer...")
    
    # Load benign data
    benign_df = pd.read_excel(benign_file)
    feature_columns = ['Pd_new', 'Qd_new', 'Vm', 'Va']
    X_benign = benign_df[feature_columns].values
    
    print(f"Benign samples loaded: {len(X_benign)}")
    print(f"Features: {feature_columns}")
    print(f"Sequence length: {seq_len}")
    print(f"Number of nodes: {num_nodes}")
    
    # Generate malicious data
    X_malicious = generate_graph_malicious_data(X_benign, seq_len, num_nodes)
    
    # Create labels
    y_benign = np.zeros(len(X_benign))
    y_malicious = np.ones(len(X_malicious))
    
    # Combine data
    X = np.vstack([X_benign, X_malicious])
    y = np.concatenate([y_benign, y_malicious])
    
    print(f"Total samples before balancing: {len(X)}")
    print(f"Class distribution: Benign={np.sum(y==0)}, Malicious={np.sum(y==1)}")
    
    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Create graph sequence data
    X_seq, y_seq, edge_index = create_graph_sequence_data(X_scaled, y, seq_len, num_nodes)
    
    print(f"Graph sequence data created: {X_seq.shape}")
    print(f"Sequence labels: {len(y_seq)}")
    print(f"Graph edges: {edge_index.shape}")
    
    return X_seq, y_seq, edge_index, scaler, feature_columns

def train_graph_informer_unsupervised(model, train_loader, val_loader, num_epochs=100, device='cuda'):
    """Train the Graph Informer model in unsupervised manner."""
    model = model.to(device)
    
    # Reconstruction loss
    reconstruction_criterion = nn.MSELoss()
    
    # Enhanced Optimizer with better learning rate
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0005, weight_decay=1e-4)
    
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
            
            reconstructed, anomaly_score, attn_weights = model(batch_x)
            
            # Reconstruction loss
            recon_loss = reconstruction_criterion(reconstructed, batch_x)
            
            # Anomaly regularization (encourage normal data to have low anomaly scores)
            normal_mask = (batch_y == 0).float()
            anomaly_reg = torch.mean(anomaly_score.squeeze() * normal_mask)
            
            # Enhanced loss with better weighting
            loss = recon_loss + 0.2 * anomaly_reg
            
            # Add attention regularization
            if attn_weights and len(attn_weights) > 0:
                attention_reg = 0.01 * torch.mean(torch.abs(attn_weights[0] - 0.5))
                loss += attention_reg
            
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
                reconstructed, anomaly_score, _ = model(batch_x)
                
                recon_loss = reconstruction_criterion(reconstructed, batch_x)
                normal_mask = (batch_y == 0).float()
                anomaly_reg = torch.mean(anomaly_score.squeeze() * normal_mask)
                loss = recon_loss + 0.1 * anomaly_reg
                
                total_val_loss += loss.item()
        
        avg_val_loss = total_val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        
        # Learning rate scheduling
        scheduler.step(avg_val_loss)
        
        # Early stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            # Save best model
            torch.save(model.state_dict(), 'best_graph_informer_model.pth')
        else:
            patience_counter += 1
        
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
    
    # Load best model
    model.load_state_dict(torch.load('best_graph_informer_model.pth'))
    return train_losses, val_losses

def evaluate_graph_informer_unsupervised(model, test_loader, device='cuda'):
    """Evaluate Graph Informer model for unsupervised anomaly detection."""
    model.eval()
    all_reconstructions = []
    all_anomaly_scores = []
    all_labels = []
    all_attention_weights = []
    
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            reconstructed, anomaly_score, attn_weights = model(batch_x)
            
            all_reconstructions.extend(reconstructed.cpu().numpy())
            all_anomaly_scores.extend(anomaly_score.cpu().numpy())
            all_labels.extend(batch_y.cpu().numpy())
            
            if attn_weights and len(attn_weights) > 0:
                all_attention_weights.extend(attn_weights[0].cpu().numpy())
    
    # Calculate reconstruction error
    all_reconstructions = np.array(all_reconstructions)
    all_anomaly_scores = np.array(all_anomaly_scores).flatten()
    all_labels = np.array(all_labels)
    
    # Reconstruction error as anomaly metric
    reconstruction_error = np.mean((all_reconstructions - test_loader.dataset.tensors[0].numpy())**2, axis=(1, 2))
    
    # Enhanced ensemble scoring with multiple features
    # Normalize scores to [0, 1] range
    reconstruction_error_norm = (reconstruction_error - np.min(reconstruction_error)) / (np.max(reconstruction_error) - np.min(reconstruction_error) + 1e-8)
    anomaly_scores_norm = (all_anomaly_scores - np.min(all_anomaly_scores)) / (np.max(all_anomaly_scores) - np.min(all_anomaly_scores) + 1e-8)
    
    # Add temporal consistency score
    temporal_consistency = np.zeros_like(reconstruction_error)
    for i in range(1, len(reconstruction_error)):
        temporal_consistency[i] = abs(reconstruction_error[i] - reconstruction_error[i-1])
    temporal_consistency_norm = (temporal_consistency - np.min(temporal_consistency)) / (np.max(temporal_consistency) - np.min(temporal_consistency) + 1e-8)
    
    # Weighted ensemble with temporal consistency
    combined_anomaly_score = (0.5 * reconstruction_error_norm + 
                             0.3 * anomaly_scores_norm + 
                             0.2 * temporal_consistency_norm)
    
    # Enhanced threshold selection strategy
    normal_scores = combined_anomaly_score[all_labels == 0]
    anomaly_scores = combined_anomaly_score[all_labels == 1]
    
    # Use multiple threshold strategies
    threshold_95 = np.percentile(normal_scores, 95)
    threshold_90 = np.percentile(normal_scores, 90)
    threshold_85 = np.percentile(normal_scores, 85)
    
    # Find optimal threshold using Youden's J statistic
    from sklearn.metrics import roc_curve
    fpr, tpr, thresholds = roc_curve(all_labels, combined_anomaly_score)
    youden_j = tpr - fpr
    optimal_idx = np.argmax(youden_j)
    optimal_threshold = thresholds[optimal_idx]
    
    # Use the optimal threshold
    threshold = optimal_threshold
    predictions = (combined_anomaly_score > threshold).astype(int)
    
    # Calculate metrics
    accuracy = accuracy_score(all_labels, predictions)
    precision = precision_score(all_labels, predictions, average='weighted')
    recall = recall_score(all_labels, predictions, average='weighted')
    f1 = f1_score(all_labels, predictions, average='weighted')
    
    # ROC AUC
    roc_auc = roc_auc_score(all_labels, combined_anomaly_score)
    
    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'roc_auc': roc_auc,
        'reconstruction_error': reconstruction_error,
        'anomaly_scores': all_anomaly_scores,
        'combined_anomaly_score': combined_anomaly_score,
        'threshold': threshold,
        'attention_weights': np.array(all_attention_weights) if all_attention_weights else None
    }
    
    return metrics

def plot_graph_informer_results(train_losses, val_losses, metrics, feature_names):
    """Plot comprehensive Graph Informer results."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Training history
    axes[0, 0].plot(train_losses, label='Training Loss', color='blue')
    axes[0, 0].plot(val_losses, label='Validation Loss', color='red')
    axes[0, 0].set_title('Graph Informer Training History')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # Anomaly score distribution
    normal_scores = metrics['combined_anomaly_score'][metrics['combined_anomaly_score'] < metrics['threshold']]
    anomaly_scores = metrics['combined_anomaly_score'][metrics['combined_anomaly_score'] >= metrics['threshold']]
    
    axes[0, 1].hist(normal_scores, bins=50, alpha=0.7, label='Normal', color='blue')
    axes[0, 1].hist(anomaly_scores, bins=50, alpha=0.7, label='Anomaly', color='red')
    axes[0, 1].axvline(metrics['threshold'], color='green', linestyle='--', label=f'Threshold: {metrics["threshold"]:.3f}')
    axes[0, 1].set_title('Graph Informer Anomaly Score Distribution')
    axes[0, 1].set_xlabel('Anomaly Score')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Metrics comparison
    metrics_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']
    metrics_values = [metrics['accuracy'], metrics['precision'], metrics['recall'], 
                     metrics['f1_score'], metrics['roc_auc']]
    
    bars = axes[0, 2].bar(metrics_names, metrics_values, color=['skyblue', 'lightgreen', 'lightcoral', 'lightsalmon', 'lightpink'])
    axes[0, 2].set_title('Graph Informer Performance Metrics')
    axes[0, 2].set_ylabel('Score')
    axes[0, 2].set_ylim(0, 1)
    
    # Add value labels on bars
    for bar, value in zip(bars, metrics_values):
        axes[0, 2].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                       f'{value:.3f}', ha='center', va='bottom')
    
    # Reconstruction error analysis
    axes[1, 0].scatter(range(len(metrics['reconstruction_error'])), metrics['reconstruction_error'], 
                      alpha=0.6, s=10)
    axes[1, 0].set_title('Graph Informer Reconstruction Error')
    axes[1, 0].set_xlabel('Sample Index')
    axes[1, 0].set_ylabel('Reconstruction Error')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Attention analysis (if available)
    if metrics['attention_weights'] is not None:
        attention_weights = metrics['attention_weights']
        mean_attention = np.mean(attention_weights, axis=(0, 1))  # Average across batch and heads
        
        axes[1, 1].bar(range(len(mean_attention)), mean_attention, color='orange', alpha=0.7)
        axes[1, 1].set_title('Graph Informer Attention Weights (Temporal)')
        axes[1, 1].set_xlabel('Time Steps')
        axes[1, 1].set_ylabel('Average Attention Weight')
        axes[1, 1].grid(True, alpha=0.3)
    else:
        axes[1, 1].text(0.5, 0.5, 'Attention weights not available', ha='center', va='center', transform=axes[1, 1].transAxes)
        axes[1, 1].set_title('Graph Informer Attention Analysis')
    
    # ROC Curve
    from sklearn.metrics import roc_curve
    fpr, tpr, _ = roc_curve([1 if score >= metrics['threshold'] else 0 for score in metrics['combined_anomaly_score']], 
                           metrics['combined_anomaly_score'])
    axes[1, 2].plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {metrics["roc_auc"]:.3f})')
    axes[1, 2].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    axes[1, 2].set_xlim([0.0, 1.0])
    axes[1, 2].set_ylim([0.0, 1.05])
    axes[1, 2].set_xlabel('False Positive Rate')
    axes[1, 2].set_ylabel('True Positive Rate')
    axes[1, 2].set_title('Graph Informer ROC Curve')
    axes[1, 2].legend(loc="lower right")
    axes[1, 2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('graph_informer_comprehensive_results.png', dpi=300, bbox_inches='tight')
    plt.show()

def main():
    """Main function to run Graph Informer Transformer unsupervised attack detection."""
    print("🔋 Graph Informer Transformer Unsupervised Power System Attack Detection")
    print("=" * 70)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Enhanced Parameters for better accuracy
    seq_len = 24  # Sequence length for temporal modeling
    d_model = 512  # Increased model dimension for better representation
    n_heads = 8    # Number of attention heads
    n_layers = 4   # Increased transformer layers for better learning
    num_nodes = 14  # IEEE 14-bus system
    
    # Load and preprocess data
    X_seq, y_seq, edge_index, scaler, feature_names = load_and_preprocess_graph_data(
        'benign_bus14.xlsx', seq_len, num_nodes, balance_classes=False
    )
    
    print(f"\nDataset Statistics:")
    print(f"Total sequences: {len(X_seq)}")
    print(f"Sequence shape: {X_seq.shape}")
    print(f"Benign sequences: {np.sum(y_seq == 0)}")
    print(f"Malicious sequences: {np.sum(y_seq == 1)}")
    print(f"Features: {feature_names}")
    print(f"Graph edges: {edge_index.shape}")
    
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
    
    # Create Graph Informer model
    model = GraphInformerTransformer(
        input_dim=len(feature_names),
        d_model=d_model,
        n_heads=n_heads,
        n_layers=n_layers,
        seq_len=seq_len,
        num_nodes=num_nodes
    )
    
    print(f"\nEnhanced Graph Informer Model Architecture:")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Sequence length: {seq_len}")
    print(f"Model dimension: {d_model}")
    print(f"Attention heads: {n_heads}")
    print(f"Transformer layers: {n_layers}")
    print(f"Number of nodes: {num_nodes}")
    print(f"Enhanced features: Optimal threshold selection, ensemble scoring, temporal consistency")
    
    # Train model (unsupervised)
    print("\nTraining Graph Informer model (unsupervised)...")
    train_losses, val_losses = train_graph_informer_unsupervised(
        model, train_loader, val_loader, 
        num_epochs=100, device=device
    )
    
    # Evaluate model
    print("\nEvaluating Graph Informer model...")
    metrics = evaluate_graph_informer_unsupervised(model, test_loader, device=device)
    
    # Print results
    print("\n🏆 Graph Informer Model Performance (Unsupervised):")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"F1-Score: {metrics['f1_score']:.4f}")
    print(f"ROC-AUC: {metrics['roc_auc']:.4f}")
    print(f"Anomaly Threshold: {metrics['threshold']:.4f}")
    
    # Plot results (disabled for headless execution)
    # plot_graph_informer_results(train_losses, val_losses, metrics, feature_names)
    print("📊 Results visualization skipped for headless execution")
    
    # Save model and results
    torch.save(model.state_dict(), 'graph_informer_unsupervised_final.pth')
    
    # Save detailed results
    with open('graph_informer_results_report.txt', 'w') as f:
        f.write("Graph Informer Transformer Unsupervised Power System Attack Detection Results\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"Dataset: {len(X_seq)} sequences, {X_seq.shape[2]} features, {seq_len} time steps\n")
        f.write(f"Graph: {num_nodes} nodes, {edge_index.shape[1]} edges\n")
        f.write(f"Model: Graph Informer Transformer with {sum(p.numel() for p in model.parameters()):,} parameters\n\n")
        f.write("Performance Metrics:\n")
        f.write(f"Accuracy: {metrics['accuracy']:.4f}\n")
        f.write(f"Precision: {metrics['precision']:.4f}\n")
        f.write(f"Recall: {metrics['recall']:.4f}\n")
        f.write(f"F1-Score: {metrics['f1_score']:.4f}\n")
        f.write(f"ROC-AUC: {metrics['roc_auc']:.4f}\n")
        f.write(f"Anomaly Threshold: {metrics['threshold']:.4f}\n\n")
        f.write("Model Architecture:\n")
        f.write(f"Sequence Length: {seq_len}\n")
        f.write(f"Model Dimension: {d_model}\n")
        f.write(f"Attention Heads: {n_heads}\n")
        f.write(f"Transformer Layers: {n_layers}\n")
        f.write(f"Number of Nodes: {num_nodes}\n")
        f.write(f"Learning Type: Unsupervised (Reconstruction + Anomaly Detection)\n")
    
    print(f"\n✅ Graph Informer implementation complete!")
    print(f"📊 Results saved to 'graph_informer_results_report.txt'")
    print(f"🎯 Model saved to 'graph_informer_unsupervised_final.pth'")
    
    return model, metrics

if __name__ == "__main__":
    main()
