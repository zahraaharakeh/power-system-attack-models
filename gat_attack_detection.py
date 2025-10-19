import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_mean_pool, global_max_pool
from torch_geometric.data import Data, DataLoader
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt
import seaborn as sns
import optuna
from imblearn.over_sampling import SMOTE
import warnings
warnings.filterwarnings('ignore')

class GATModel(nn.Module):
    def __init__(self, num_features, num_classes, hidden_channels=64, heads=8, num_layers=3, dropout=0.2):
        super(GATModel, self).__init__()
        self.num_layers = num_layers
        self.dropout = dropout
        
        # GAT layers
        self.gat_layers = nn.ModuleList()
        
        # First GAT layer
        self.gat_layers.append(GATConv(num_features, hidden_channels, heads=heads, dropout=dropout, concat=True))
        
        # Middle GAT layers
        for _ in range(num_layers - 2):
            self.gat_layers.append(GATConv(hidden_channels * heads, hidden_channels, heads=heads, dropout=dropout, concat=True))
        
        # Final GAT layer
        self.gat_layers.append(GATConv(hidden_channels * heads, hidden_channels, heads=1, dropout=dropout, concat=False))
        
        # Batch normalization layers
        self.batch_norms = nn.ModuleList()
        for _ in range(num_layers - 1):
            self.batch_norms.append(nn.BatchNorm1d(hidden_channels * heads))
        self.batch_norms.append(nn.BatchNorm1d(hidden_channels))
        
        # Global pooling combination
        self.pool_combination = nn.Linear(hidden_channels * 2, hidden_channels)
        
        # Classification head with attention
        self.classifier = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_channels // 2),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels // 2, hidden_channels // 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels // 4, num_classes)
        )
        
        # Feature attention
        self.feature_attention = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.ReLU(),
            nn.Linear(hidden_channels // 2, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x, edge_index, batch):
        # GAT layers with residual connections
        for i, gat_layer in enumerate(self.gat_layers):
            x_residual = x if i > 0 and x.size(1) == self.gat_layers[i](x, edge_index).size(1) else None
            
            x = gat_layer(x, edge_index)
            x = self.batch_norms[i](x)
            
            if i < len(self.gat_layers) - 1:  # Not the last layer
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
            
            # Add residual connection if dimensions match
            if x_residual is not None and x_residual.size(1) == x.size(1):
                x = x + x_residual
        
        # Global pooling (combine mean and max pooling)
        x_mean = global_mean_pool(x, batch)
        x_max = global_max_pool(x, batch)
        x_combined = torch.cat([x_mean, x_max], dim=1)
        x = self.pool_combination(x_combined)
        x = F.relu(x)
        
        # Feature attention
        attention_weights = self.feature_attention(x)
        x = x * attention_weights
        
        # Classification
        x = self.classifier(x)
        return x, attention_weights

def create_power_system_graph(features, labels, use_ieee14_topology=True):
    """Create proper power system graph structure based on IEEE 14-bus topology."""
    data_list = []
    
    if use_ieee14_topology:
        # IEEE 14-bus system topology (simplified)
        # Each sample represents measurements from 4 buses
        edges = [
            [0, 1], [1, 2], [2, 3],  # Linear connections
            [0, 2], [1, 3],          # Cross connections
            [0, 0], [1, 1], [2, 2], [3, 3]  # Self loops
        ]
        edge_index = torch.LongTensor(edges).t().contiguous()
    else:
        # Fully connected graph for comparison
        edges = []
        for i in range(4):
            for j in range(4):
                edges.append([i, j])
        edge_index = torch.LongTensor(edges).t().contiguous()
    
    for i in range(len(features)):
        # Each feature becomes a node
        x = torch.FloatTensor(features[i]).view(-1, 1)  # Shape: (4, 1)
        
        # Add node features (feature type encoding)
        node_types = torch.FloatTensor([[1, 0, 0, 0],  # Pd_new
                                       [0, 1, 0, 0],   # Qd_new  
                                       [0, 0, 1, 0],   # Vm
                                       [0, 0, 0, 1]])  # Va
        
        x = torch.cat([x, node_types], dim=1)  # Shape: (4, 5)
        
        data = Data(
            x=x,
            edge_index=edge_index,
            y=torch.LongTensor([labels[i]])
        )
        data_list.append(data)
    
    return data_list

def generate_sophisticated_malicious_data(benign_data):
    """Generate sophisticated malicious data with realistic attack patterns."""
    print("Generating sophisticated malicious data for GAT...")
    
    n_samples = len(benign_data)
    all_attacks = []
    
    # 1. False Data Injection (FDI) - Feature-specific attacks
    fdi_attack = benign_data.copy()
    for i in range(benign_data.shape[1]):
        # Different attack intensity for different features
        if i < 2:  # Power features (Pd, Qd)
            noise_scale = np.random.uniform(0.1, 0.3, (len(benign_data), 1))
        else:  # Voltage features (Vm, Va)
            noise_scale = np.random.uniform(0.05, 0.2, (len(benign_data), 1))
        
        fdi_attack[:, i:i+1] += np.random.normal(0, noise_scale, (len(benign_data), 1)).flatten()
    all_attacks.append(fdi_attack)
    
    # 2. Coordinated Attack - Attack multiple features simultaneously
    coord_attack = benign_data.copy()
    attack_mask = np.random.choice([0, 1], size=benign_data.shape, p=[0.6, 0.4])
    coord_attack += attack_mask * np.random.normal(0, 0.25, benign_data.shape)
    all_attacks.append(coord_attack)
    
    # 3. Replay Attack with temporal drift
    replay_attack = benign_data.copy()
    shift_indices = np.random.randint(0, len(benign_data), len(benign_data))
    replay_attack = benign_data[shift_indices]
    # Add temporal drift
    drift = np.linspace(0, 0.2, len(benign_data))
    replay_attack += drift[:, np.newaxis] * np.random.normal(0, 0.1, benign_data.shape)
    all_attacks.append(replay_attack)
    
    # 4. Stealth Attack - Maintain statistical properties
    stealth_attack = benign_data.copy()
    feature_means = np.mean(benign_data, axis=0)
    feature_stds = np.std(benign_data, axis=0)
    
    for i in range(benign_data.shape[1]):
        # Add noise while maintaining mean
        noise = np.random.normal(0, feature_stds[i] * 0.3, len(benign_data))
        stealth_attack[:, i] += noise
        # Adjust to maintain original mean
        stealth_attack[:, i] -= np.mean(stealth_attack[:, i]) - feature_means[i]
    all_attacks.append(stealth_attack)
    
    # Combine all attacks
    malicious_data = np.vstack(all_attacks)
    
    # Ensure we have the same number as benign samples
    if len(malicious_data) > n_samples:
        indices = np.random.choice(len(malicious_data), n_samples, replace=False)
        malicious_data = malicious_data[indices]
    elif len(malicious_data) < n_samples:
        # Repeat some samples
        additional_needed = n_samples - len(malicious_data)
        additional_indices = np.random.choice(len(malicious_data), additional_needed, replace=True)
        additional_data = malicious_data[additional_indices]
        malicious_data = np.vstack([malicious_data, additional_data])
    
    print(f"Generated {len(malicious_data)} sophisticated malicious samples")
    return malicious_data

def load_and_preprocess_data(benign_file, balance_classes=True):
    """Load and preprocess data with class balancing."""
    print("Loading and preprocessing data for GAT...")
    
    # Load benign data
    benign_df = pd.read_excel(benign_file)
    feature_columns = ['Pd_new', 'Qd_new', 'Vm', 'Va']
    X_benign = benign_df[feature_columns].values
    
    print(f"Benign samples loaded: {len(X_benign)}")
    print(f"Features: {feature_columns}")
    
    # Generate malicious data
    X_malicious = generate_sophisticated_malicious_data(X_benign)
    
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

def train_gat_model(model, train_loader, val_loader, num_epochs=100, device='cuda', class_weights=None):
    """Train the GAT model with advanced techniques."""
    model = model.to(device)
    
    # Loss function with class weights
    if class_weights is not None:
        criterion = nn.CrossEntropyLoss(weight=torch.FloatTensor(class_weights).to(device))
    else:
        criterion = nn.CrossEntropyLoss()
    
    # Optimizer with different learning rates for different parts
    optimizer = torch.optim.AdamW([
        {'params': model.gat_layers.parameters(), 'lr': 0.001},
        {'params': model.classifier.parameters(), 'lr': 0.005},
        {'params': model.feature_attention.parameters(), 'lr': 0.002}
    ], weight_decay=1e-5)
    
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
            
            out, attention_weights = model(batch.x, batch.edge_index, batch.batch)
            loss = criterion(out, batch.y)
            
            # Add attention regularization
            attention_reg = 0.01 * torch.mean(torch.abs(attention_weights - 0.5))
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
            for batch in val_loader:
                batch = batch.to(device)
                out, _ = model(batch.x, batch.edge_index, batch.batch)
                loss = criterion(out, batch.y)
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
            torch.save(model.state_dict(), 'best_gat_model.pth')
        else:
            patience_counter += 1
        
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
    
    # Load best model
    model.load_state_dict(torch.load('best_gat_model.pth'))
    return train_losses, val_losses

def evaluate_gat_model(model, test_loader, device='cuda'):
    """Comprehensive evaluation of GAT model."""
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []
    all_attention_weights = []
    
    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            out, attention_weights = model(batch.x, batch.edge_index, batch.batch)
            
            probs = F.softmax(out, dim=1)
            preds = out.argmax(dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(batch.y.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            all_attention_weights.extend(attention_weights.cpu().numpy())
    
    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='weighted')
    recall = recall_score(all_labels, all_preds, average='weighted')
    f1 = f1_score(all_labels, all_preds, average='weighted')
    
    # ROC AUC
    all_probs = np.array(all_probs)
    roc_auc = roc_auc_score(all_labels, all_probs[:, 1])
    
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
        'attention_weights': np.array(all_attention_weights)
    }
    
    return metrics

def plot_gat_results(train_losses, val_losses, metrics, feature_names):
    """Plot comprehensive GAT results."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Training history
    axes[0, 0].plot(train_losses, label='Training Loss', color='blue')
    axes[0, 0].plot(val_losses, label='Validation Loss', color='red')
    axes[0, 0].set_title('GAT Training History')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # Confusion Matrix
    cm = metrics['confusion_matrix']
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0, 1])
    axes[0, 1].set_title('GAT Confusion Matrix')
    axes[0, 1].set_xlabel('Predicted')
    axes[0, 1].set_ylabel('Actual')
    
    # Metrics comparison
    metrics_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']
    metrics_values = [metrics['accuracy'], metrics['precision'], metrics['recall'], 
                     metrics['f1_score'], metrics['roc_auc']]
    
    bars = axes[0, 2].bar(metrics_names, metrics_values, color=['skyblue', 'lightgreen', 'lightcoral', 'lightsalmon', 'lightpink'])
    axes[0, 2].set_title('GAT Performance Metrics')
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
    
    axes[1, 0].set_title('GAT Class-specific Performance')
    axes[1, 0].set_xlabel('Class')
    axes[1, 0].set_ylabel('Score')
    axes[1, 0].set_xticks(x_pos)
    axes[1, 0].set_xticklabels(class_names)
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Feature attention analysis
    attention_weights = metrics['attention_weights']
    mean_attention = np.mean(attention_weights, axis=0).flatten()
    
    axes[1, 1].bar(range(len(mean_attention)), mean_attention, color='orange', alpha=0.7)
    axes[1, 1].set_title('GAT Feature Attention Weights')
    axes[1, 1].set_xlabel('Features')
    axes[1, 1].set_ylabel('Average Attention Weight')
    axes[1, 1].set_xticks(range(len(feature_names)))
    axes[1, 1].set_xticklabels(feature_names, rotation=45)
    axes[1, 1].grid(True, alpha=0.3)
    
    # Attention distribution
    axes[1, 2].hist(attention_weights.flatten(), bins=50, alpha=0.7, color='purple', edgecolor='black')
    axes[1, 2].set_title('GAT Attention Weight Distribution')
    axes[1, 2].set_xlabel('Attention Weight')
    axes[1, 2].set_ylabel('Frequency')
    axes[1, 2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('gat_comprehensive_results.png', dpi=300, bbox_inches='tight')
    plt.show()

def main():
    """Main function to run GAT attack detection."""
    print("🔋 GAT Power System Attack Detection")
    print("=" * 50)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load and preprocess data
    X, y, scaler, feature_names = load_and_preprocess_data('benign_bus14.xlsx', balance_classes=True)
    
    print(f"\nDataset Statistics:")
    print(f"Total samples: {len(X)}")
    print(f"Benign samples: {np.sum(y == 0)}")
    print(f"Malicious samples: {np.sum(y == 1)}")
    print(f"Feature shape: {X.shape}")
    print(f"Features: {feature_names}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42, stratify=y_train)
    
    # Create graph data
    print("\nCreating power system graph structure...")
    train_data = create_power_system_graph(X_train, y_train)
    val_data = create_power_system_graph(X_val, y_val)
    test_data = create_power_system_graph(X_test, y_test)
    
    # Create data loaders
    batch_size = 64
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)
    
    # Calculate class weights
    class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
    print(f"Class weights: {class_weights}")
    
    # Create GAT model
    model = GATModel(
        num_features=5,  # 1 feature value + 4 node type encodings
        num_classes=2,
        hidden_channels=128,
        heads=8,
        num_layers=3,
        dropout=0.2
    )
    
    print(f"\nGAT Model Architecture:")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(model)
    
    # Train model
    print("\nTraining GAT model...")
    train_losses, val_losses = train_gat_model(
        model, train_loader, val_loader, 
        num_epochs=100, device=device, class_weights=class_weights
    )
    
    # Evaluate model
    print("\nEvaluating GAT model...")
    metrics = evaluate_gat_model(model, test_loader, device=device)
    
    # Print results
    print("\n🏆 GAT Model Performance:")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"F1-Score: {metrics['f1_score']:.4f}")
    print(f"ROC-AUC: {metrics['roc_auc']:.4f}")
    
    print(f"\nClass-specific Performance:")
    print(f"Benign - Precision: {metrics['precision_per_class'][0]:.4f}, Recall: {metrics['recall_per_class'][0]:.4f}, F1: {metrics['f1_per_class'][0]:.4f}")
    print(f"Malicious - Precision: {metrics['precision_per_class'][1]:.4f}, Recall: {metrics['recall_per_class'][1]:.4f}, F1: {metrics['f1_per_class'][1]:.4f}")
    
    # Feature attention analysis
    mean_attention = np.mean(metrics['attention_weights'], axis=0).flatten()
    print(f"\nFeature Attention Weights:")
    for i, (feature, attention) in enumerate(zip(feature_names, mean_attention[:len(feature_names)])):
        print(f"{feature}: {attention:.4f}")
    
    # Plot results
    plot_gat_results(train_losses, val_losses, metrics, feature_names)
    
    # Save model and results
    torch.save(model.state_dict(), 'gat_attack_detection_final.pth')
    
    # Save detailed results
    with open('gat_results_report.txt', 'w') as f:
        f.write("GAT Power System Attack Detection Results\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Dataset: {len(X)} samples, {X.shape[1]} features\n")
        f.write(f"Model: GAT with {sum(p.numel() for p in model.parameters()):,} parameters\n\n")
        f.write("Performance Metrics:\n")
        f.write(f"Accuracy: {metrics['accuracy']:.4f}\n")
        f.write(f"Precision: {metrics['precision']:.4f}\n")
        f.write(f"Recall: {metrics['recall']:.4f}\n")
        f.write(f"F1-Score: {metrics['f1_score']:.4f}\n")
        f.write(f"ROC-AUC: {metrics['roc_auc']:.4f}\n\n")
        f.write("Feature Attention Analysis:\n")
        for feature, attention in zip(feature_names, mean_attention[:len(feature_names)]):
            f.write(f"{feature}: {attention:.4f}\n")
    
    print(f"\n✅ GAT implementation complete!")
    print(f"📊 Results saved to 'gat_results_report.txt'")
    print(f"🎯 Model saved to 'gat_attack_detection_final.pth'")
    
    return model, metrics

if __name__ == "__main__":
    main()
