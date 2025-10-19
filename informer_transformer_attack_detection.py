import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt
import seaborn as sns
import math
import warnings
from imblearn.over_sampling import SMOTE
warnings.filterwarnings('ignore')

class PositionalEncoding(nn.Module):
    """Positional encoding for transformer models."""
    
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        return x + self.pe[:x.size(0), :]

class ProbSparseAttention(nn.Module):
    """ProbSparse attention mechanism from Informer paper."""
    
    def __init__(self, d_model, n_heads, factor=5, dropout=0.1):
        super(ProbSparseAttention, self).__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.factor = factor
        self.dropout = nn.Dropout(dropout)
        
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        
    def _prob_QK(self, Q, K, sample_k, n_top):
        """ProbSparse attention computation."""
        B, H, L_K, E = K.shape
        _, _, L_Q, _ = Q.shape
        
        # Calculate the Q, K score
        K_expand = K.unsqueeze(-3).expand(B, H, L_Q, L_K, E)
        Q_sample = Q.unsqueeze(-2).expand(B, H, L_Q, sample_k, E)
        
        Q_K_sample = torch.matmul(Q_sample, K_expand.transpose(-2, -1))
        
        # Find the Top_k query with sparisty measurement
        M = Q_K_sample.max(-1)[0] - torch.div(Q_K_sample.sum(-1), L_K)
        M_top = M.topk(n_top, sorted=False)[1]
        
        # Use the reduced Q to calculate Q_K
        Q_reduce = Q[torch.arange(B)[:, None, None], torch.arange(H)[None, :, None], M_top, :]
        Q_K = torch.matmul(Q_reduce, K.transpose(-2, -1))
        
        return Q_K, M_top
    
    def _get_initial_context(self, V, L_Q):
        B, H, L_V, D = V.shape
        if not self.mask_flag:
            V_sum = V.sum(dim=-2)
            contex = V_sum.unsqueeze(-2).expand(B, H, L_Q, V_sum.shape[-1]).clone()
        else:
            contex = V.cumsum(dim=-2)
        return contex
    
    def _update_context(self, context_in, V, scores, index, L_Q, attn_mask):
        B, H, L_V, D = V.shape
        
        attn = torch.softmax(scores, dim=-1)
        context_in[torch.arange(B)[:, None, None], torch.arange(H)[None, :, None], index, :] = torch.matmul(attn, V).type_as(context_in)
        
        if self.output_attention:
            attns = (torch.ones([B, H, L_V, L_V]) / L_V).type_as(attn).to(attn.device)
            attns[torch.arange(B)[:, None, None], torch.arange(H)[None, :, None], index, :] = attn
            return (context_in, attns)
        else:
            return (context_in, None)
    
    def forward(self, queries, keys, values, attn_mask):
        B, L_Q, H, D = queries.shape
        _, L_K, _, _ = keys.shape
        
        queries = queries.transpose(2, 1)
        keys = keys.transpose(2, 1)
        values = values.transpose(2, 1)
        
        U_part = self.factor * np.ceil(np.log(L_K)).astype('int').item()
        u = self.factor * np.ceil(np.log(L_Q)).astype('int').item()
        
        U_part = U_part if U_part < L_K else L_K
        u = u if u < L_Q else L_Q
        
        scores_top, index = self._prob_QK(queries, keys, sample_k=U_part, n_top=u)
        
        # Get context
        context = self._get_initial_context(values, L_Q)
        # Update context with selected top_k queries
        context, attn = self._update_context(context, values, scores_top, index, L_Q, attn_mask)
        
        return context.transpose(2, 1).contiguous(), attn

class InformerBlock(nn.Module):
    """Informer block with ProbSparse attention."""
    
    def __init__(self, d_model, n_heads, d_ff=None, dropout=0.1, activation="relu"):
        super(InformerBlock, self).__init__()
        d_ff = d_ff or 4 * d_model
        
        self.attention = ProbSparseAttention(d_model, n_heads, dropout=dropout)
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu
        
    def forward(self, x, attn_mask=None):
        # Self-attention
        new_x, attn = self.attention(x, x, x, attn_mask)
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

class InformerTransformer(nn.Module):
    """Informer Transformer for power system attack detection."""
    
    def __init__(self, input_dim, d_model=512, n_heads=8, n_layers=3, d_ff=2048, 
                 seq_len=24, pred_len=1, dropout=0.1, num_classes=2):
        super(InformerTransformer, self).__init__()
        
        self.input_dim = input_dim
        self.d_model = d_model
        self.seq_len = seq_len
        self.pred_len = pred_len
        
        # Input projection
        self.input_projection = nn.Linear(input_dim, d_model)
        
        # Positional encoding
        self.pos_encoding = PositionalEncoding(d_model)
        
        # Informer blocks
        self.encoder_layers = nn.ModuleList([
            InformerBlock(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(d_model * seq_len, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, num_classes)
        )
        
        # Attention weights for interpretability
        self.attention_weights = None
        
    def forward(self, x):
        # Input projection
        x = self.input_projection(x)  # [batch, seq_len, d_model]
        
        # Positional encoding
        x = x.transpose(0, 1)  # [seq_len, batch, d_model]
        x = self.pos_encoding(x)
        x = x.transpose(0, 1)  # [batch, seq_len, d_model]
        
        # Encoder layers
        attn_weights = []
        for layer in self.encoder_layers:
            x, attn = layer(x)
            attn_weights.append(attn)
        
        self.attention_weights = attn_weights
        
        # Global pooling and classification
        x = x.reshape(x.size(0), -1)  # Flatten
        x = self.classifier(x)
        
        return x, attn_weights

def create_sequence_data(features, labels, seq_len=24):
    """Create sequence data for transformer model."""
    sequences = []
    sequence_labels = []
    
    for i in range(len(features) - seq_len + 1):
        seq = features[i:i + seq_len]
        label = labels[i + seq_len - 1]  # Use the last label in sequence
        sequences.append(seq)
        sequence_labels.append(label)
    
    return np.array(sequences), np.array(sequence_labels)

def generate_temporal_malicious_data(benign_data, seq_len=24):
    """Generate temporal malicious data with sophisticated attack patterns."""
    print("Generating temporal malicious data for Informer...")
    
    n_samples = len(benign_data)
    all_attacks = []
    
    # 1. Temporal False Data Injection (FDI)
    fdi_attack = benign_data.copy()
    for i in range(seq_len, len(benign_data)):
        # Add time-varying perturbations
        time_factor = (i - seq_len) / (len(benign_data) - seq_len)
        noise_scale = 0.1 + 0.2 * time_factor
        fdi_attack[i] += np.random.normal(0, noise_scale, benign_data.shape[1])
    all_attacks.append(fdi_attack)
    
    # 2. Coordinated Temporal Attack
    coord_attack = benign_data.copy()
    attack_windows = np.random.choice([0, 1], size=len(benign_data), p=[0.7, 0.3])
    for i in range(seq_len, len(benign_data)):
        if attack_windows[i]:
            # Coordinate attack across multiple time steps
            window_size = min(seq_len, len(benign_data) - i)
            for j in range(window_size):
                coord_attack[i + j] += np.random.normal(0, 0.15, benign_data.shape[1])
    all_attacks.append(coord_attack)
    
    # 3. Stealth Temporal Attack
    stealth_attack = benign_data.copy()
    feature_means = np.mean(benign_data, axis=0)
    feature_stds = np.std(benign_data, axis=0)
    
    for i in range(seq_len, len(benign_data)):
        # Add noise while maintaining temporal consistency
        for j in range(benign_data.shape[1]):
            noise = np.random.normal(0, feature_stds[j] * 0.2)
            stealth_attack[i, j] += noise
            # Maintain temporal smoothness
            if i > 0:
                stealth_attack[i, j] = 0.7 * stealth_attack[i, j] + 0.3 * stealth_attack[i-1, j]
    all_attacks.append(stealth_attack)
    
    # 4. Replay Attack with Temporal Drift
    replay_attack = benign_data.copy()
    for i in range(seq_len, len(benign_data)):
        # Replay previous measurements with temporal drift
        replay_idx = max(0, i - np.random.randint(seq_len, seq_len * 2))
        replay_attack[i] = benign_data[replay_idx]
        # Add temporal drift
        drift = np.random.normal(0, 0.1, benign_data.shape[1])
        replay_attack[i] += drift
    all_attacks.append(replay_attack)
    
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
    
    print(f"Generated {len(malicious_data)} temporal malicious samples")
    return malicious_data

def load_and_preprocess_temporal_data(benign_file, seq_len=24, balance_classes=True):
    """Load and preprocess data for temporal transformer model."""
    print("Loading and preprocessing temporal data for Informer...")
    
    # Load benign data
    benign_df = pd.read_excel(benign_file)
    feature_columns = ['Pd_new', 'Qd_new', 'Vm', 'Va']
    X_benign = benign_df[feature_columns].values
    
    print(f"Benign samples loaded: {len(X_benign)}")
    print(f"Features: {feature_columns}")
    print(f"Sequence length: {seq_len}")
    
    # Generate malicious data
    X_malicious = generate_temporal_malicious_data(X_benign, seq_len)
    
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
    
    # Create sequence data
    X_seq, y_seq = create_sequence_data(X_scaled, y, seq_len)
    
    print(f"Sequence data created: {X_seq.shape}")
    print(f"Sequence labels: {len(y_seq)}")
    
    return X_seq, y_seq, scaler, feature_columns

def train_informer_model(model, train_loader, val_loader, num_epochs=100, device='cuda', class_weights=None):
    """Train the Informer Transformer model."""
    model = model.to(device)
    
    # Loss function with class weights
    if class_weights is not None:
        criterion = nn.CrossEntropyLoss(weight=torch.FloatTensor(class_weights).to(device))
    else:
        criterion = nn.CrossEntropyLoss()
    
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
            
            out, attn_weights = model(batch_x)
            loss = criterion(out, batch_y)
            
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
                out, _ = model(batch_x)
                loss = criterion(out, batch_y)
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
            torch.save(model.state_dict(), 'best_informer_model.pth')
        else:
            patience_counter += 1
        
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
    
    # Load best model
    model.load_state_dict(torch.load('best_informer_model.pth'))
    return train_losses, val_losses

def evaluate_informer_model(model, test_loader, device='cuda'):
    """Comprehensive evaluation of Informer model."""
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []
    all_attention_weights = []
    
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            out, attn_weights = model(batch_x)
            
            probs = F.softmax(out, dim=1)
            preds = out.argmax(dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(batch_y.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            
            if attn_weights and len(attn_weights) > 0:
                all_attention_weights.extend(attn_weights[0].cpu().numpy())
    
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
        'attention_weights': np.array(all_attention_weights) if all_attention_weights else None
    }
    
    return metrics

def plot_informer_results(train_losses, val_losses, metrics, feature_names):
    """Plot comprehensive Informer results."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Training history
    axes[0, 0].plot(train_losses, label='Training Loss', color='blue')
    axes[0, 0].plot(val_losses, label='Validation Loss', color='red')
    axes[0, 0].set_title('Informer Training History')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # Confusion Matrix
    cm = metrics['confusion_matrix']
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0, 1])
    axes[0, 1].set_title('Informer Confusion Matrix')
    axes[0, 1].set_xlabel('Predicted')
    axes[0, 1].set_ylabel('Actual')
    
    # Metrics comparison
    metrics_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']
    metrics_values = [metrics['accuracy'], metrics['precision'], metrics['recall'], 
                     metrics['f1_score'], metrics['roc_auc']]
    
    bars = axes[0, 2].bar(metrics_names, metrics_values, color=['skyblue', 'lightgreen', 'lightcoral', 'lightsalmon', 'lightpink'])
    axes[0, 2].set_title('Informer Performance Metrics')
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
    
    axes[1, 0].set_title('Informer Class-specific Performance')
    axes[1, 0].set_xlabel('Class')
    axes[1, 0].set_ylabel('Score')
    axes[1, 0].set_xticks(x_pos)
    axes[1, 0].set_xticklabels(class_names)
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Attention analysis (if available)
    if metrics['attention_weights'] is not None:
        attention_weights = metrics['attention_weights']
        mean_attention = np.mean(attention_weights, axis=(0, 1))  # Average across batch and heads
        
        axes[1, 1].bar(range(len(mean_attention)), mean_attention, color='orange', alpha=0.7)
        axes[1, 1].set_title('Informer Attention Weights (Temporal)')
        axes[1, 1].set_xlabel('Time Steps')
        axes[1, 1].set_ylabel('Average Attention Weight')
        axes[1, 1].grid(True, alpha=0.3)
    else:
        axes[1, 1].text(0.5, 0.5, 'Attention weights not available', ha='center', va='center', transform=axes[1, 1].transAxes)
        axes[1, 1].set_title('Informer Attention Analysis')
    
    # Attention distribution
    if metrics['attention_weights'] is not None:
        axes[1, 2].hist(attention_weights.flatten(), bins=50, alpha=0.7, color='purple', edgecolor='black')
        axes[1, 2].set_title('Informer Attention Weight Distribution')
        axes[1, 2].set_xlabel('Attention Weight')
        axes[1, 2].set_ylabel('Frequency')
        axes[1, 2].grid(True, alpha=0.3)
    else:
        axes[1, 2].text(0.5, 0.5, 'Attention distribution not available', ha='center', va='center', transform=axes[1, 2].transAxes)
        axes[1, 2].set_title('Informer Attention Distribution')
    
    plt.tight_layout()
    plt.savefig('informer_comprehensive_results.png', dpi=300, bbox_inches='tight')
    plt.show()

def main():
    """Main function to run Informer Transformer attack detection."""
    print("🔋 Informer Transformer Power System Attack Detection")
    print("=" * 60)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Parameters
    seq_len = 24  # Sequence length for temporal modeling
    d_model = 256  # Model dimension
    n_heads = 8    # Number of attention heads
    n_layers = 3   # Number of transformer layers
    
    # Load and preprocess data
    X_seq, y_seq, scaler, feature_names = load_and_preprocess_temporal_data('benign_bus14.xlsx', seq_len, balance_classes=True)
    
    print(f"\nDataset Statistics:")
    print(f"Total sequences: {len(X_seq)}")
    print(f"Sequence shape: {X_seq.shape}")
    print(f"Benign sequences: {np.sum(y_seq == 0)}")
    print(f"Malicious sequences: {np.sum(y_seq == 1)}")
    print(f"Features: {feature_names}")
    
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
    
    # Calculate class weights
    class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
    print(f"Class weights: {class_weights}")
    
    # Create Informer model
    model = InformerTransformer(
        input_dim=len(feature_names),
        d_model=d_model,
        n_heads=n_heads,
        n_layers=n_layers,
        seq_len=seq_len,
        num_classes=2
    )
    
    print(f"\nInformer Model Architecture:")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Sequence length: {seq_len}")
    print(f"Model dimension: {d_model}")
    print(f"Attention heads: {n_heads}")
    print(f"Transformer layers: {n_layers}")
    
    # Train model
    print("\nTraining Informer model...")
    train_losses, val_losses = train_informer_model(
        model, train_loader, val_loader, 
        num_epochs=100, device=device, class_weights=class_weights
    )
    
    # Evaluate model
    print("\nEvaluating Informer model...")
    metrics = evaluate_informer_model(model, test_loader, device=device)
    
    # Print results
    print("\n🏆 Informer Model Performance:")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"F1-Score: {metrics['f1_score']:.4f}")
    print(f"ROC-AUC: {metrics['roc_auc']:.4f}")
    
    print(f"\nClass-specific Performance:")
    print(f"Benign - Precision: {metrics['precision_per_class'][0]:.4f}, Recall: {metrics['recall_per_class'][0]:.4f}, F1: {metrics['f1_per_class'][0]:.4f}")
    print(f"Malicious - Precision: {metrics['precision_per_class'][1]:.4f}, Recall: {metrics['recall_per_class'][1]:.4f}, F1: {metrics['f1_per_class'][1]:.4f}")
    
    # Plot results
    plot_informer_results(train_losses, val_losses, metrics, feature_names)
    
    # Save model and results
    torch.save(model.state_dict(), 'informer_attack_detection_final.pth')
    
    # Save detailed results
    with open('informer_results_report.txt', 'w') as f:
        f.write("Informer Transformer Power System Attack Detection Results\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Dataset: {len(X_seq)} sequences, {X_seq.shape[2]} features, {seq_len} time steps\n")
        f.write(f"Model: Informer Transformer with {sum(p.numel() for p in model.parameters()):,} parameters\n\n")
        f.write("Performance Metrics:\n")
        f.write(f"Accuracy: {metrics['accuracy']:.4f}\n")
        f.write(f"Precision: {metrics['precision']:.4f}\n")
        f.write(f"Recall: {metrics['recall']:.4f}\n")
        f.write(f"F1-Score: {metrics['f1_score']:.4f}\n")
        f.write(f"ROC-AUC: {metrics['roc_auc']:.4f}\n\n")
        f.write("Model Architecture:\n")
        f.write(f"Sequence Length: {seq_len}\n")
        f.write(f"Model Dimension: {d_model}\n")
        f.write(f"Attention Heads: {n_heads}\n")
        f.write(f"Transformer Layers: {n_layers}\n")
    
    print(f"\n✅ Informer implementation complete!")
    print(f"📊 Results saved to 'informer_results_report.txt'")
    print(f"🎯 Model saved to 'informer_attack_detection_final.pth'")
    
    return model, metrics

if __name__ == "__main__":
    main()
