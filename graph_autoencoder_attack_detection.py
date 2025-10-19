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
import networkx as nx
from torch_geometric.nn import GCNConv, GATConv, global_mean_pool, global_max_pool
from torch_geometric.data import Data, DataLoader
from torch_geometric.utils import to_networkx, from_networkx
warnings.filterwarnings('ignore')

class GraphEncoder(nn.Module):
    """Graph encoder with multiple GCN layers and attention."""
    
    def __init__(self, input_dim, hidden_dims=[64, 32], latent_dim=16, dropout=0.2):
        super(GraphEncoder, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.latent_dim = latent_dim
        
        # GCN layers
        self.gcn_layers = nn.ModuleList()
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            self.gcn_layers.append(GCNConv(prev_dim, hidden_dim))
            prev_dim = hidden_dim
        
        # Final encoding layer
        self.encoder_final = nn.Linear(prev_dim, latent_dim)
        
        # Batch normalization and dropout
        self.batch_norms = nn.ModuleList()
        for hidden_dim in hidden_dims:
            self.batch_norms.append(nn.BatchNorm1d(hidden_dim))
        
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.ReLU()
        
    def forward(self, x, edge_index, batch):
        # GCN encoding
        for i, (gcn, bn) in enumerate(zip(self.gcn_layers, self.batch_norms)):
            x = gcn(x, edge_index)
            x = bn(x)
            x = self.activation(x)
            x = self.dropout(x)
        
        # Global pooling
        x = global_mean_pool(x, batch)
        
        # Final encoding
        z = self.encoder_final(x)
        
        return z

class GraphDecoder(nn.Module):
    """Graph decoder with MLP and graph reconstruction."""
    
    def __init__(self, latent_dim, hidden_dims=[32, 64], output_dim=4, num_nodes=14, dropout=0.2):
        super(GraphDecoder, self).__init__()
        
        self.latent_dim = latent_dim
        self.output_dim = output_dim
        self.num_nodes = num_nodes
        
        # MLP decoder
        decoder_layers = []
        prev_dim = latent_dim
        
        for hidden_dim in hidden_dims:
            decoder_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim
        
        decoder_layers.append(nn.Linear(prev_dim, output_dim * num_nodes))
        self.decoder = nn.Sequential(*decoder_layers)
        
        # Edge reconstruction head
        self.edge_decoder = nn.Sequential(
            nn.Linear(latent_dim, latent_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(latent_dim // 2, 1),
            nn.Sigmoid()
        )
        
    def forward(self, z, batch_size):
        # Node feature reconstruction
        node_features = self.decoder(z)
        node_features = node_features.view(batch_size, self.num_nodes, self.output_dim)
        
        # Edge reconstruction (simplified - using node similarity)
        edge_probs = self.edge_decoder(z)
        
        return node_features, edge_probs

class GraphAutoencoder(nn.Module):
    """Complete Graph Autoencoder for power system attack detection."""
    
    def __init__(self, input_dim=4, hidden_dims=[64, 32], latent_dim=16, num_nodes=14, dropout=0.2):
        super(GraphAutoencoder, self).__init__()
        
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.num_nodes = num_nodes
        
        # Encoder and decoder
        self.encoder = GraphEncoder(input_dim, hidden_dims, latent_dim, dropout)
        self.decoder = GraphDecoder(latent_dim, hidden_dims[::-1], input_dim, num_nodes, dropout)
        
        # Anomaly detection head
        self.anomaly_detector = nn.Sequential(
            nn.Linear(latent_dim, latent_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(latent_dim // 2, 1),
            nn.Sigmoid()
        )
        
        # Feature importance head
        self.feature_importance = nn.Sequential(
            nn.Linear(latent_dim, latent_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(latent_dim // 2, input_dim),
            nn.Softmax(dim=1)
        )
        
    def forward(self, x, edge_index, batch):
        # Encode
        z = self.encoder(x, edge_index, batch)
        
        # Decode
        batch_size = z.size(0)
        reconstructed_nodes, edge_probs = self.decoder(z, batch_size)
        
        # Anomaly detection
        anomaly_score = self.anomaly_detector(z)
        
        # Feature importance
        feature_importance = self.feature_importance(z)
        
        return reconstructed_nodes, z, anomaly_score, feature_importance, edge_probs

class GraphAutoencoderLoss(nn.Module):
    """Combined loss function for graph autoencoder."""
    
    def __init__(self, reconstruction_weight=1.0, anomaly_weight=0.5, feature_weight=0.1, edge_weight=0.2):
        super(GraphAutoencoderLoss, self).__init__()
        self.reconstruction_weight = reconstruction_weight
        self.anomaly_weight = anomaly_weight
        self.feature_weight = feature_weight
        self.edge_weight = edge_weight
        
        self.mse_loss = nn.MSELoss()
        self.bce_loss = nn.BCELoss()
        
    def forward(self, reconstructed_nodes, original_nodes, anomaly_score, labels, feature_importance, edge_probs, original_edges):
        # Node reconstruction loss
        reconstruction_loss = self.mse_loss(reconstructed_nodes, original_nodes)
        
        # Anomaly detection loss
        anomaly_loss = self.bce_loss(anomaly_score.squeeze(), labels.float())
        
        # Feature importance regularization (encourage diversity)
        feature_reg = -torch.mean(torch.sum(feature_importance * torch.log(feature_importance + 1e-8), dim=1))
        
        # Edge reconstruction loss (simplified)
        edge_loss = self.mse_loss(edge_probs, torch.ones_like(edge_probs) * 0.1)  # Sparse graph assumption
        
        # Total loss
        total_loss = (self.reconstruction_weight * reconstruction_loss + 
                     self.anomaly_weight * anomaly_loss + 
                     self.feature_weight * feature_reg + 
                     self.edge_weight * edge_loss)
        
        return total_loss, reconstruction_loss, anomaly_loss, feature_reg, edge_loss

def create_power_system_graph_data(X, num_nodes=14):
    """Create graph data from power system measurements."""
    print("Creating power system graph data...")
    
    # IEEE 14-bus system topology
    edges = [
        (0, 1), (0, 2), (1, 2), (1, 3), (2, 3), (2, 4), (3, 4),
        (4, 5), (4, 6), (5, 6), (6, 7), (6, 8), (6, 9), (6, 10),
        (6, 11), (6, 12), (6, 13), (9, 10), (9, 14), (10, 11),
        (12, 13), (13, 14)
    ]
    
    # Convert to edge_index format
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    
    # Create graph data for each sample
    graph_data_list = []
    
    for i, sample in enumerate(X):
        # Create node features (repeat sample for all nodes with some variation)
        node_features = []
        for node in range(num_nodes):
            # Add node-specific variations
            node_variation = np.random.normal(0, 0.1, 4)
            node_feature = sample + node_variation
            node_features.append(node_feature)
        
        node_features = torch.FloatTensor(np.array(node_features))
        
        # Create batch tensor
        batch = torch.zeros(num_nodes, dtype=torch.long)
        
        # Create graph data
        graph_data = Data(x=node_features, edge_index=edge_index, batch=batch)
        graph_data_list.append(graph_data)
    
    return graph_data_list

def generate_graph_malicious_data(benign_data, num_nodes=14):
    """Generate sophisticated malicious data for graph autoencoder."""
    print("Generating graph-based malicious data...")
    
    n_samples = len(benign_data)
    all_attacks = []
    
    # 1. Graph-based FDI Attack
    fdi_attack = benign_data.copy()
    for i in range(benign_data.shape[1]):
        # Different attack intensity for different features
        if i < 2:  # Power features
            noise_scale = np.random.uniform(0.15, 0.5, (len(benign_data), 1))
        else:  # Voltage features
            noise_scale = np.random.uniform(0.1, 0.4, (len(benign_data), 1))
        
        fdi_attack[:, i:i+1] += np.random.normal(0, noise_scale, (len(benign_data), 1)).flatten()
    all_attacks.append(fdi_attack)
    
    # 2. Coordinated Graph Attack
    coord_attack = benign_data.copy()
    # Simulate coordinated attack across multiple nodes
    attack_strength = np.random.uniform(0.2, 0.6, len(benign_data))
    for i, strength in enumerate(attack_strength):
        coord_attack[i] += np.random.normal(0, strength, benign_data.shape[1])
    all_attacks.append(coord_attack)
    
    # 3. Stealth Graph Attack
    stealth_attack = benign_data.copy()
    feature_means = np.mean(benign_data, axis=0)
    feature_stds = np.std(benign_data, axis=0)
    
    for i in range(benign_data.shape[1]):
        # Add subtle noise while maintaining statistical properties
        noise = np.random.normal(0, feature_stds[i] * 0.3, len(benign_data))
        stealth_attack[:, i] += noise
        # Adjust to maintain original mean
        stealth_attack[:, i] -= np.mean(stealth_attack[:, i]) - feature_means[i]
    all_attacks.append(stealth_attack)
    
    # 4. Graph Structure Attack
    structure_attack = benign_data.copy()
    for i in range(len(benign_data)):
        # Simulate attack that affects graph structure
        perturbation = np.random.uniform(-0.3, 0.3, benign_data.shape[1])
        structure_attack[i] += perturbation
    all_attacks.append(structure_attack)
    
    # 5. Multi-node Coordinated Attack
    multi_node_attack = benign_data.copy()
    for i in range(len(benign_data)):
        # Simulate attack affecting multiple nodes simultaneously
        node_effects = np.random.uniform(-0.4, 0.4, benign_data.shape[1])
        multi_node_attack[i] += node_effects
    all_attacks.append(multi_node_attack)
    
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
    
    print(f"Generated {len(malicious_data)} graph-based malicious samples")
    return malicious_data

def load_and_preprocess_graph_data(benign_file, balance_classes=True):
    """Load and preprocess data for graph autoencoder."""
    print("Loading and preprocessing data for graph autoencoder...")
    
    # Load benign data
    benign_df = pd.read_excel(benign_file)
    feature_columns = ['Pd_new', 'Qd_new', 'Vm', 'Va']
    X_benign = benign_df[feature_columns].values
    
    print(f"Benign samples loaded: {len(X_benign)}")
    print(f"Features: {feature_columns}")
    
    # Generate malicious data
    X_malicious = generate_graph_malicious_data(X_benign)
    
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

def train_graph_autoencoder(model, train_loader, val_loader, num_epochs=100, device='cuda'):
    """Train the graph autoencoder model."""
    model = model.to(device)
    
    # Loss function
    criterion = GraphAutoencoderLoss(
        reconstruction_weight=1.0,
        anomaly_weight=0.5,
        feature_weight=0.1,
        edge_weight=0.2
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
        
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            
            # Get labels (assuming they're stored in batch.y)
            labels = batch.y if hasattr(batch, 'y') else torch.zeros(batch.num_graphs, device=device)
            
            reconstructed_nodes, z, anomaly_score, feature_importance, edge_probs = model(
                batch.x, batch.edge_index, batch.batch
            )
            
            # Reshape for loss calculation
            original_nodes = batch.x.view(batch.num_graphs, -1, batch.x.size(1))
            
            total_loss, recon_loss, anomaly_loss, feature_reg, edge_loss = criterion(
                reconstructed_nodes, original_nodes, anomaly_score, labels, 
                feature_importance, edge_probs, batch.edge_index
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
            for batch in val_loader:
                batch = batch.to(device)
                labels = batch.y if hasattr(batch, 'y') else torch.zeros(batch.num_graphs, device=device)
                
                reconstructed_nodes, z, anomaly_score, feature_importance, edge_probs = model(
                    batch.x, batch.edge_index, batch.batch
                )
                
                original_nodes = batch.x.view(batch.num_graphs, -1, batch.x.size(1))
                
                total_loss, _, _, _, _ = criterion(
                    reconstructed_nodes, original_nodes, anomaly_score, labels, 
                    feature_importance, edge_probs, batch.edge_index
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
            torch.save(model.state_dict(), 'best_graph_autoencoder_model.pth')
        else:
            patience_counter += 1
        
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
    
    # Load best model
    model.load_state_dict(torch.load('best_graph_autoencoder_model.pth'))
    return train_losses, val_losses

def evaluate_graph_autoencoder(model, test_loader, device='cuda'):
    """Comprehensive evaluation of graph autoencoder model."""
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []
    all_reconstruction_errors = []
    all_feature_importance = []
    
    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            labels = batch.y if hasattr(batch, 'y') else torch.zeros(batch.num_graphs, device=device)
            
            reconstructed_nodes, z, anomaly_score, feature_importance, edge_probs = model(
                batch.x, batch.edge_index, batch.batch
            )
            
            # Calculate reconstruction error
            original_nodes = batch.x.view(batch.num_graphs, -1, batch.x.size(1))
            reconstruction_error = torch.mean((reconstructed_nodes - original_nodes) ** 2, dim=(1, 2))
            
            # Use anomaly score as prediction
            preds = (anomaly_score.squeeze() > 0.5).long()
            probs = anomaly_score.squeeze()
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            all_reconstruction_errors.extend(reconstruction_error.cpu().numpy())
            all_feature_importance.extend(feature_importance.cpu().numpy())
    
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
        'feature_importance': np.array(all_feature_importance)
    }
    
    return metrics

def plot_graph_autoencoder_results(train_losses, val_losses, metrics, feature_names):
    """Plot comprehensive graph autoencoder results."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Training history
    axes[0, 0].plot(train_losses, label='Training Loss', color='blue', linewidth=2)
    axes[0, 0].plot(val_losses, label='Validation Loss', color='red', linewidth=2)
    axes[0, 0].set_title('Graph Autoencoder Training History', fontsize=14, fontweight='bold')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Confusion Matrix
    cm = metrics['confusion_matrix']
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0, 1],
                xticklabels=['Benign', 'Malicious'], yticklabels=['Benign', 'Malicious'])
    axes[0, 1].set_title('Graph Autoencoder Confusion Matrix', fontsize=14, fontweight='bold')
    axes[0, 1].set_xlabel('Predicted')
    axes[0, 1].set_ylabel('Actual')
    
    # Metrics comparison
    metrics_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']
    metrics_values = [metrics['accuracy'], metrics['precision'], metrics['recall'], 
                     metrics['f1_score'], metrics['roc_auc']]
    
    colors = ['#2E8B57', '#4169E1', '#DC143C', '#FF8C00', '#9932CC']
    bars = axes[0, 2].bar(metrics_names, metrics_values, color=colors, alpha=0.8)
    axes[0, 2].set_title('Graph Autoencoder Performance Metrics', fontsize=14, fontweight='bold')
    axes[0, 2].set_ylabel('Score')
    axes[0, 2].set_ylim(0, 1)
    axes[0, 2].tick_params(axis='x', rotation=45)
    
    # Add value labels on bars
    for bar, value in zip(bars, metrics_values):
        axes[0, 2].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                       f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
    
    # Class-specific performance
    class_names = ['Benign', 'Malicious']
    x_pos = np.arange(len(class_names))
    width = 0.25
    
    axes[1, 0].bar(x_pos - width, metrics['precision_per_class'], width, label='Precision', 
                   color='skyblue', alpha=0.8)
    axes[1, 0].bar(x_pos, metrics['recall_per_class'], width, label='Recall', 
                   color='lightcoral', alpha=0.8)
    axes[1, 0].bar(x_pos + width, metrics['f1_per_class'], width, label='F1-Score', 
                   color='lightgreen', alpha=0.8)
    
    axes[1, 0].set_title('Graph Autoencoder Class-specific Performance', fontsize=14, fontweight='bold')
    axes[1, 0].set_xlabel('Class')
    axes[1, 0].set_ylabel('Score')
    axes[1, 0].set_xticks(x_pos)
    axes[1, 0].set_xticklabels(class_names)
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].set_ylim(0, 1)
    
    # Reconstruction error analysis
    reconstruction_errors = metrics['reconstruction_errors']
    axes[1, 1].hist(reconstruction_errors, bins=50, alpha=0.7, color='purple', edgecolor='black')
    axes[1, 1].set_title('Graph Reconstruction Error Distribution', fontsize=14, fontweight='bold')
    axes[1, 1].set_xlabel('Reconstruction Error')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].grid(True, alpha=0.3)
    
    # Add statistics text
    mean_error = np.mean(reconstruction_errors)
    std_error = np.std(reconstruction_errors)
    axes[1, 1].axvline(mean_error, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_error:.3f}')
    axes[1, 1].legend()
    
    # Feature importance analysis
    feature_importance = metrics['feature_importance']
    mean_importance = np.mean(feature_importance, axis=0)
    
    bars = axes[1, 2].bar(range(len(mean_importance)), mean_importance, 
                          color='orange', alpha=0.7, edgecolor='black')
    axes[1, 2].set_title('Graph Autoencoder Feature Importance', fontsize=14, fontweight='bold')
    axes[1, 2].set_xlabel('Features')
    axes[1, 2].set_ylabel('Average Importance Weight')
    axes[1, 2].set_xticks(range(len(feature_names)))
    axes[1, 2].set_xticklabels(feature_names, rotation=45)
    axes[1, 2].grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, value in zip(bars, mean_importance):
        axes[1, 2].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005, 
                       f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('graph_autoencoder_comprehensive_results.png', dpi=300, bbox_inches='tight')
    plt.show()

def main():
    """Main function to run graph autoencoder attack detection."""
    print("🔋 Graph Autoencoder Power System Attack Detection")
    print("=" * 60)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load and preprocess data
    X, y, scaler, feature_names = load_and_preprocess_graph_data('benign_bus14.xlsx', balance_classes=True)
    
    print(f"\nDataset Statistics:")
    print(f"Total samples: {len(X)}")
    print(f"Benign samples: {np.sum(y == 0)}")
    print(f"Malicious samples: {np.sum(y == 1)}")
    print(f"Feature shape: {X.shape}")
    print(f"Features: {feature_names}")
    
    # Create graph data
    graph_data_list = create_power_system_graph_data(X)
    
    # Add labels to graph data
    for i, graph_data in enumerate(graph_data_list):
        graph_data.y = torch.tensor([y[i]], dtype=torch.float)
    
    # Split data
    train_data, test_data = train_test_split(graph_data_list, test_size=0.2, random_state=42, stratify=y)
    train_data, val_data = train_test_split(train_data, test_size=0.2, random_state=42)
    
    # Create data loaders
    train_loader = DataLoader(train_data, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=16, shuffle=False)
    test_loader = DataLoader(test_data, batch_size=16, shuffle=False)
    
    # Create graph autoencoder model
    model = GraphAutoencoder(
        input_dim=len(feature_names),
        hidden_dims=[64, 32],
        latent_dim=16,
        num_nodes=14,
        dropout=0.2
    )
    
    print(f"\nGraph Autoencoder Model Architecture:")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Input dimension: {len(feature_names)}")
    print(f"Latent dimension: 16")
    print(f"Hidden dimensions: [64, 32]")
    print(f"Number of nodes: 14 (IEEE 14-bus system)")
    print(f"Graph structure: IEEE 14-bus topology")
    print(model)
    
    # Train model
    print("\nTraining graph autoencoder model...")
    train_losses, val_losses = train_graph_autoencoder(
        model, train_loader, val_loader, 
        num_epochs=100, device=device
    )
    
    # Evaluate model
    print("\nEvaluating graph autoencoder model...")
    metrics = evaluate_graph_autoencoder(model, test_loader, device=device)
    
    # Print results
    print("\n🏆 Graph Autoencoder Model Performance:")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"F1-Score: {metrics['f1_score']:.4f}")
    print(f"ROC-AUC: {metrics['roc_auc']:.4f}")
    
    print(f"\nClass-specific Performance:")
    print(f"Benign - Precision: {metrics['precision_per_class'][0]:.4f}, Recall: {metrics['recall_per_class'][0]:.4f}, F1: {metrics['f1_per_class'][0]:.4f}")
    print(f"Malicious - Precision: {metrics['precision_per_class'][1]:.4f}, Recall: {metrics['recall_per_class'][1]:.4f}, F1: {metrics['f1_per_class'][1]:.4f}")
    
    # Plot results
    plot_graph_autoencoder_results(train_losses, val_losses, metrics, feature_names)
    
    # Save model and results
    torch.save(model.state_dict(), 'graph_autoencoder_final.pth')
    
    # Save detailed results
    with open('graph_autoencoder_results_report.txt', 'w') as f:
        f.write("Graph Autoencoder Power System Attack Detection Results\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Dataset: {len(X)} samples, {X.shape[1]} features\n")
        f.write(f"Model: Graph Autoencoder with {sum(p.numel() for p in model.parameters()):,} parameters\n\n")
        f.write("Performance Metrics:\n")
        f.write(f"Accuracy: {metrics['accuracy']:.4f}\n")
        f.write(f"Precision: {metrics['precision']:.4f}\n")
        f.write(f"Recall: {metrics['recall']:.4f}\n")
        f.write(f"F1-Score: {metrics['f1_score']:.4f}\n")
        f.write(f"ROC-AUC: {metrics['roc_auc']:.4f}\n\n")
        f.write("Model Architecture:\n")
        f.write(f"Input Dimension: {len(feature_names)}\n")
        f.write(f"Latent Dimension: 16\n")
        f.write(f"Hidden Dimensions: [64, 32]\n")
        f.write(f"Number of Nodes: 14 (IEEE 14-bus system)\n")
        f.write(f"Graph Structure: IEEE 14-bus topology\n")
    
    print(f"\n✅ Graph autoencoder implementation complete!")
    print(f"📊 Results saved to 'graph_autoencoder_results_report.txt'")
    print(f"🎯 Model saved to 'graph_autoencoder_final.pth'")
    
    return model, metrics

if __name__ == "__main__":
    main()
