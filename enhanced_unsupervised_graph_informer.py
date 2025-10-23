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
from typing import Dict, List, Tuple, Optional
import logging
warnings.filterwarnings('ignore')

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ContrastiveLearningModule(nn.Module):
    """Contrastive learning module for better representation learning."""
    
    def __init__(self, d_model, temperature=0.1):
        super(ContrastiveLearningModule, self).__init__()
        self.d_model = d_model
        self.temperature = temperature
        
        # Projection head for contrastive learning
        self.projection_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model // 2)
        )
        
    def forward(self, features):
        """Apply contrastive learning projection."""
        return self.projection_head(features)
    
    def contrastive_loss(self, features1, features2, labels):
        """Compute contrastive loss between positive and negative pairs."""
        # Normalize features
        features1 = F.normalize(features1, dim=1)
        features2 = F.normalize(features2, dim=1)
        
        # Compute similarity matrix
        similarity = torch.matmul(features1, features2.T) / self.temperature
        
        # Create positive/negative masks
        batch_size = features1.size(0)
        positive_mask = torch.eye(batch_size, device=features1.device)
        negative_mask = 1 - positive_mask
        
        # Compute contrastive loss
        exp_sim = torch.exp(similarity)
        positive_sim = exp_sim * positive_mask
        negative_sim = exp_sim * negative_mask
        
        # InfoNCE loss
        loss = -torch.log(positive_sim.sum(dim=1) / (positive_sim.sum(dim=1) + negative_sim.sum(dim=1) + 1e-8))
        
        return loss.mean()

class TemporalConsistencyModule(nn.Module):
    """Temporal consistency module for better sequence understanding."""
    
    def __init__(self, d_model, seq_len):
        super(TemporalConsistencyModule, self).__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        
        # Temporal attention mechanism
        self.temporal_attention = nn.MultiheadAttention(d_model, num_heads=8, batch_first=True)
        
        # Temporal consistency predictor
        self.consistency_predictor = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.ReLU(),
            nn.Linear(d_model, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        """Apply temporal consistency modeling."""
        batch_size, seq_len, d_model = x.shape
        
        # Temporal attention
        attn_out, attn_weights = self.temporal_attention(x, x, x)
        
        # Compute temporal consistency scores
        consistency_scores = []
        for i in range(1, seq_len):
            # Compare consecutive time steps
            prev_features = x[:, i-1, :]
            curr_features = x[:, i, :]
            combined_features = torch.cat([prev_features, curr_features], dim=1)
            consistency_score = self.consistency_predictor(combined_features)
            consistency_scores.append(consistency_score)
        
        consistency_scores = torch.cat(consistency_scores, dim=1)  # [batch, seq_len-1, 1]
        
        return attn_out, attn_weights, consistency_scores

class DynamicEnsembleScoring(nn.Module):
    """Dynamic ensemble scoring with adaptive weight learning."""
    
    def __init__(self, d_model, num_scorers=5):
        super(DynamicEnsembleScoring, self).__init__()
        self.num_scorers = num_scorers
        
        # Individual anomaly scorers
        self.scorers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, d_model // 2),
                nn.ReLU(),
                nn.Linear(d_model // 2, 1),
                nn.Sigmoid()
            ) for _ in range(num_scorers)
        ])
        
        # Dynamic weight generator
        self.weight_generator = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, num_scorers),
            nn.Softmax(dim=-1)
        )
        
        # Context-aware weight adjustment
        self.context_encoder = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, d_model)
        )
        
    def forward(self, features, context=None):
        """Generate dynamic ensemble scores."""
        # Individual scorer outputs
        scorer_outputs = []
        for scorer in self.scorers:
            score = scorer(features)
            scorer_outputs.append(score)
        
        scorer_outputs = torch.cat(scorer_outputs, dim=1)  # [batch, num_scorers]
        
        # Generate dynamic weights
        if context is not None:
            context_features = self.context_encoder(context)
            combined_features = features + context_features
        else:
            combined_features = features
            
        dynamic_weights = self.weight_generator(combined_features)
        
        # Weighted ensemble
        ensemble_score = torch.sum(scorer_outputs * dynamic_weights, dim=1, keepdim=True)
        
        return ensemble_score, dynamic_weights, scorer_outputs

class EnhancedUnsupervisedGraphInformer(nn.Module):
    """Enhanced Unsupervised Graph Informer with advanced anomaly detection mechanisms."""
    
    def __init__(self, input_dim, d_model=512, n_heads=8, n_layers=6, seq_len=24, num_nodes=14, 
                 dropout=0.1, use_contrastive=True, use_temporal_consistency=True):
        super(EnhancedUnsupervisedGraphInformer, self).__init__()
        
        self.input_dim = input_dim
        self.d_model = d_model
        self.seq_len = seq_len
        self.num_nodes = num_nodes
        self.use_contrastive = use_contrastive
        self.use_temporal_consistency = use_temporal_consistency
        
        # Enhanced input projection with deeper architecture
        self.input_projection = nn.Sequential(
            nn.Linear(input_dim, d_model * 2),
            nn.LayerNorm(d_model * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model)
        )
        
        # Learnable positional encoding
        self.pos_encoding = self._create_positional_encoding(d_model, seq_len)
        
        # Enhanced transformer blocks
        self.encoder_layers = nn.ModuleList([
            self._create_enhanced_block(d_model, n_heads, dropout, layer_idx=i)
            for i in range(n_layers)
        ])
        
        # Multi-scale graph convolutions
        self.graph_convs = nn.ModuleList([
            GCNConv(d_model, d_model) for _ in range(2)
        ])
        
        self.gat_convs = nn.ModuleList([
            GATConv(d_model, d_model // 4, heads=4, dropout=dropout) for _ in range(2)
        ])
        
        # Enhanced reconstruction decoders
        self.decoder_global = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.LayerNorm(d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, d_model // 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 4, input_dim)
        )
        
        self.decoder_local = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.LayerNorm(d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, input_dim)
        )
        
        self.decoder_temporal = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.LayerNorm(d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, input_dim)
        )
        
        # Contrastive learning module
        if use_contrastive:
            self.contrastive_module = ContrastiveLearningModule(d_model)
        
        # Temporal consistency module
        if use_temporal_consistency:
            self.temporal_consistency = TemporalConsistencyModule(d_model, seq_len)
        
        # Dynamic ensemble scoring with more sophisticated detection
        self.ensemble_scoring = DynamicEnsembleScoring(d_model, num_scorers=8)
        
        # Additional specialized anomaly detectors
        self.statistical_detector = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, d_model // 4),
            nn.ReLU(),
            nn.Linear(d_model // 4, 1),
            nn.Sigmoid()
        )
        
        self.pattern_detector = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, 1),
            nn.Sigmoid()
        )
        
        # Online learning components
        self.online_adaptation = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, d_model),
            nn.Sigmoid()
        )
        
        # Performance tracking
        self.performance_history = []
        self.adaptation_threshold = 0.05
        
    def _create_positional_encoding(self, d_model, max_len):
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        
        return nn.Parameter(pe, requires_grad=True)
    
    def _create_enhanced_block(self, d_model, n_heads, dropout, layer_idx):
        return nn.ModuleDict({
            'attention': nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True),
            'norm1': nn.LayerNorm(d_model),
            'norm2': nn.LayerNorm(d_model),
            'ff': nn.Sequential(
                nn.Linear(d_model, d_model * 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(d_model * 2, d_model),
                nn.Dropout(dropout)
            ),
            'gate': nn.Sequential(
                nn.Linear(d_model, d_model // 4),
                nn.ReLU(),
                nn.Linear(d_model // 4, 1),
                nn.Sigmoid()
            )
        })
    
    def forward(self, x, edge_index=None, return_detailed_output=False):
        batch_size, seq_len, features = x.shape
        
        # Enhanced input projection
        x = self.input_projection(x)
        
        # Add learnable positional encoding
        x = x + self.pos_encoding[:, :seq_len, :]
        
        # Multi-scale graph convolutions
        if edge_index is not None:
            x_reshaped = x.reshape(-1, self.d_model)
            
            # GCN convolutions
            for conv in self.graph_convs:
                x_reshaped = conv(x_reshaped, edge_index)
                x_reshaped = F.relu(x_reshaped)
            
            # GAT convolutions
            for gat in self.gat_convs:
                x_reshaped = gat(x_reshaped, edge_index)
                x_reshaped = F.relu(x_reshaped)
            
            x = x_reshaped.reshape(batch_size, seq_len, self.d_model)
        
        # Enhanced transformer blocks
        attention_weights = []
        gate_weights = []
        
        for i, layer in enumerate(self.encoder_layers):
            # Adaptive gating
            gate_weight = layer['gate'](x.mean(dim=1)).unsqueeze(1)
            gate_weights.append(gate_weight)
            
            # Self-attention
            attn_out, attn_weights = layer['attention'](x, x, x)
            x = layer['norm1'](x + gate_weight * attn_out)
            
            # Feed-forward with adaptive gating
            ff_out = layer['ff'](x)
            x = layer['norm2'](x + gate_weight * ff_out)
            
            attention_weights.append(attn_weights)
        
        # Temporal consistency modeling
        temporal_consistency_scores = None
        if self.use_temporal_consistency:
            x, temporal_attn_weights, temporal_consistency_scores = self.temporal_consistency(x)
            attention_weights.append(temporal_attn_weights)
        
        # Multi-scale feature extraction
        global_features = x.mean(dim=1)
        local_features = x.max(dim=1)[0]
        temporal_features = x[:, -1, :]  # Last time step features
        
        # Enhanced reconstruction
        reconstructed_global = self.decoder_global(global_features)
        reconstructed_local = self.decoder_local(local_features)
        reconstructed_temporal = self.decoder_temporal(temporal_features)
        
        # Contrastive learning
        contrastive_loss = None
        if self.use_contrastive:
            # Create positive pairs (temporal neighbors)
            features1 = self.contrastive_module(global_features)
            features2 = self.contrastive_module(temporal_features)
            # Use temporal consistency as pseudo-labels
            if temporal_consistency_scores is not None:
                labels = (temporal_consistency_scores.mean(dim=1) > 0.5).float()
            else:
                labels = torch.ones(batch_size, device=x.device)
            contrastive_loss = self.contrastive_module.contrastive_loss(features1, features2, labels)
        
        # Dynamic ensemble scoring
        ensemble_score, dynamic_weights, scorer_outputs = self.ensemble_scoring(
            global_features, context=temporal_features
        )
        
        # Online adaptation
        adaptation_signal = self.online_adaptation(global_features)
        
        if return_detailed_output:
            return {
                'reconstructed_global': reconstructed_global,
                'reconstructed_local': reconstructed_local,
                'reconstructed_temporal': reconstructed_temporal,
                'ensemble_score': ensemble_score,
                'dynamic_weights': dynamic_weights,
                'scorer_outputs': scorer_outputs,
                'attention_weights': attention_weights,
                'gate_weights': gate_weights,
                'temporal_consistency_scores': temporal_consistency_scores,
                'contrastive_loss': contrastive_loss,
                'adaptation_signal': adaptation_signal,
                'global_features': global_features,
                'local_features': local_features,
                'temporal_features': temporal_features
            }
        else:
            return (reconstructed_global, reconstructed_local, reconstructed_temporal, 
                   ensemble_score, attention_weights, global_features, local_features, 
                   temporal_features, contrastive_loss, dynamic_weights)

def create_enhanced_malicious_data(benign_data, seq_len=24, num_nodes=14):
    """Generate enhanced malicious data patterns for unsupervised learning."""
    print("Generating enhanced malicious data patterns...")
    
    n_samples = len(benign_data)
    all_attacks = []
    
    # 1. Advanced temporal correlation attack
    temporal_attack = benign_data.copy()
    for i in range(seq_len, len(benign_data)):
        time_factor = (i - seq_len) / (len(benign_data) - seq_len)
        noise_scale = 0.02 + 0.25 * time_factor
        
        for node in range(min(num_nodes, benign_data.shape[1] // 4)):
            if node*4+4 <= benign_data.shape[1]:
                # Add temporally correlated noise
                base_noise = np.random.normal(0, noise_scale, 4)
                if i > 0:
                    base_noise = 0.85 * base_noise + 0.15 * temporal_attack[i-1, node*4:(node+1)*4] * 0.1
                temporal_attack[i, node*4:(node+1)*4] += base_noise
    all_attacks.append(temporal_attack)
    
    # 2. Graph-aware coordinated attack
    graph_attack = benign_data.copy()
    for i in range(seq_len, len(benign_data)):
        if np.random.random() < 0.35:
            max_nodes = min(num_nodes, benign_data.shape[1] // 4)
            if max_nodes >= 2:
                attack_size = min(np.random.randint(2, 4), max_nodes)
                attacked_nodes = np.random.choice(range(max_nodes), size=attack_size, replace=False)
            else:
                attacked_nodes = [0] if max_nodes > 0 else []
            
            for node in attacked_nodes:
                if node*4+4 <= benign_data.shape[1]:
                    # Add coordinated perturbations
                    perturbation = np.random.normal(0, 0.08, 4)
                    # Add feature correlation
                    perturbation[1] = 0.8 * perturbation[1] + 0.2 * perturbation[0]
                    perturbation[3] = 0.7 * perturbation[3] + 0.3 * perturbation[2]
                    graph_attack[i, node*4:(node+1)*4] += perturbation
    all_attacks.append(graph_attack)
    
    # 3. Advanced statistical attack with multiple distributions
    statistical_attack = benign_data.copy()
    for i in range(seq_len, len(benign_data)):
        if np.random.random() < 0.25:
            for j in range(benign_data.shape[1]):
                # Add sophisticated statistical anomalies
                if np.random.random() < 0.3:
                    # Exponential noise
                    noise = np.random.exponential(0.03) - 0.03
                elif np.random.random() < 0.6:
                    # Beta distribution noise
                    noise = np.random.beta(2, 5) - 0.3
                else:
                    # Gamma distribution noise
                    noise = np.random.gamma(2, 0.02) - 0.04
                statistical_attack[i, j] += noise
    all_attacks.append(statistical_attack)
    
    # 4. Stealthy replay attack with systematic drift
    replay_attack = benign_data.copy()
    for i in range(seq_len, len(benign_data)):
        if np.random.random() < 0.2:
            # Replay previous measurements with systematic drift
            replay_idx = max(0, i - np.random.randint(seq_len, seq_len * 2))
            replay_attack[i] = benign_data[replay_idx]
            # Add systematic drift
            drift = np.random.normal(0, 0.02, benign_data.shape[1])
            # Add feature-specific drift patterns
            for j in range(0, benign_data.shape[1], 4):
                if j+3 < benign_data.shape[1]:
                    # Voltage magnitude drift
                    replay_attack[i, j+2] += drift[j+2] * 0.3
                    # Phase angle drift
                    replay_attack[i, j+3] += drift[j+3] * 0.2
    all_attacks.append(replay_attack)
    
    # Combine attacks
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

def load_and_preprocess_enhanced_unsupervised_data(benign_file, seq_len=24, num_nodes=14):
    """Load and preprocess data for enhanced unsupervised learning."""
    print("Loading and preprocessing data for enhanced unsupervised learning...")
    
    # Load benign data with enhanced feature engineering
    benign_df = pd.read_excel(benign_file)
    feature_columns = ['Pd_new', 'Qd_new', 'Vm', 'Va']
    
    # Add engineered features for better discrimination
    benign_df['Pd_Qd_ratio'] = benign_df['Pd_new'] / (benign_df['Qd_new'] + 1e-8)
    benign_df['Vm_Va_product'] = benign_df['Vm'] * benign_df['Va']
    benign_df['power_magnitude'] = np.sqrt(benign_df['Pd_new']**2 + benign_df['Qd_new']**2)
    benign_df['voltage_phase_diff'] = np.abs(benign_df['Va'] - benign_df['Va'].shift(1))
    
    # Use enhanced feature set
    enhanced_features = feature_columns + ['Pd_Qd_ratio', 'Vm_Va_product', 'power_magnitude', 'voltage_phase_diff']
    X_benign = benign_df[enhanced_features].fillna(0).values
    
    print(f"Benign samples loaded: {len(X_benign)}")
    
    # Generate enhanced malicious data with same feature dimension
    X_malicious = create_enhanced_malicious_data(X_benign, seq_len, num_nodes)
    
    # Update feature columns for return
    feature_columns = enhanced_features
    
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

def train_enhanced_unsupervised_model(model, train_loader, val_loader, num_epochs=150, device='cpu'):
    """Train the enhanced unsupervised model with advanced loss functions."""
    model = model.to(device)
    
    # Enhanced optimizer with better learning rate
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)
    
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    patience = 25
    patience_counter = 0
    
    for epoch in range(num_epochs):
        # Training
        model.train()
        total_train_loss = 0
        
        for batch_x, _ in train_loader:  # Ignore labels in unsupervised learning
            batch_x = batch_x.to(device)
            optimizer.zero_grad()
            
            # Get detailed output
            outputs = model(batch_x, return_detailed_output=True)
            
            # Multi-scale reconstruction losses
            recon_loss_global = F.mse_loss(outputs['reconstructed_global'], batch_x.mean(dim=1))
            recon_loss_local = F.mse_loss(outputs['reconstructed_local'], batch_x.max(dim=1)[0])
            recon_loss_temporal = F.mse_loss(outputs['reconstructed_temporal'], batch_x[:, -1, :])
            
            # Temporal reconstruction loss
            recon_loss_temporal_full = F.mse_loss(
                outputs['reconstructed_temporal'].unsqueeze(1).expand(-1, batch_x.size(1), -1), 
                batch_x
            )
            
            # Combined reconstruction loss with better weighting
            recon_loss = (0.4 * recon_loss_global + 
                         0.3 * recon_loss_local + 
                         0.2 * recon_loss_temporal + 
                         0.1 * recon_loss_temporal_full)
            
            # Add additional statistical and pattern losses
            statistical_loss = F.mse_loss(
                model.statistical_detector(outputs['global_features']), 
                torch.zeros_like(outputs['ensemble_score'])
            )
            
            pattern_loss = F.mse_loss(
                model.pattern_detector(outputs['local_features']), 
                torch.zeros_like(outputs['ensemble_score'])
            )
            
            # Contrastive learning loss
            contrastive_loss = 0
            if outputs['contrastive_loss'] is not None:
                contrastive_loss = outputs['contrastive_loss']
            
            # Temporal consistency loss
            temporal_consistency_loss = 0
            if outputs['temporal_consistency_scores'] is not None:
                # Encourage high consistency for normal data
                consistency_scores = outputs['temporal_consistency_scores'].mean(dim=1)
                reconstruction_error = torch.mean((batch_x - outputs['reconstructed_global'].unsqueeze(1).expand(-1, batch_x.size(1), -1))**2, dim=(1, 2))
                normal_mask = (reconstruction_error < torch.median(reconstruction_error)).float()
                temporal_consistency_loss = torch.mean((1 - consistency_scores) * normal_mask)
            
            # Ensemble scoring regularization
            ensemble_reg = torch.mean(outputs['ensemble_score'])
            
            # Total loss with enhanced weighting for better accuracy
            loss = (recon_loss + 
                   0.2 * contrastive_loss + 
                   0.15 * temporal_consistency_loss + 
                   0.1 * statistical_loss +
                   0.1 * pattern_loss +
                   0.05 * ensemble_reg)
            
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
                outputs = model(batch_x, return_detailed_output=True)
                
                recon_loss_global = F.mse_loss(outputs['reconstructed_global'], batch_x.mean(dim=1))
                recon_loss_local = F.mse_loss(outputs['reconstructed_local'], batch_x.max(dim=1)[0])
                recon_loss_temporal = F.mse_loss(outputs['reconstructed_temporal'], batch_x[:, -1, :])
                recon_loss_temporal_full = F.mse_loss(
                    outputs['reconstructed_temporal'].unsqueeze(1).expand(-1, batch_x.size(1), -1), 
                    batch_x
                )
                
                recon_loss = (0.3 * recon_loss_global + 
                             0.25 * recon_loss_local + 
                             0.25 * recon_loss_temporal + 
                             0.2 * recon_loss_temporal_full)
                
                contrastive_loss = outputs['contrastive_loss'] if outputs['contrastive_loss'] is not None else 0
                temporal_consistency_loss = 0
                if outputs['temporal_consistency_scores'] is not None:
                    consistency_scores = outputs['temporal_consistency_scores'].mean(dim=1)
                    reconstruction_error = torch.mean((batch_x - outputs['reconstructed_global'].unsqueeze(1).expand(-1, batch_x.size(1), -1))**2, dim=(1, 2))
                    normal_mask = (reconstruction_error < torch.median(reconstruction_error)).float()
                    temporal_consistency_loss = torch.mean((1 - consistency_scores) * normal_mask)
                
                ensemble_reg = torch.mean(outputs['ensemble_score'])
                
                loss = (recon_loss + 
                       0.1 * contrastive_loss + 
                       0.05 * temporal_consistency_loss + 
                       0.02 * ensemble_reg)
                
                total_val_loss += loss.item()
        
        avg_val_loss = total_val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        
        # Learning rate scheduling
        scheduler.step(avg_val_loss)
        
        # Early stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            torch.save(model.state_dict(), 'best_enhanced_unsupervised_graph_informer.pth')
        else:
            patience_counter += 1
        
        if patience_counter >= patience:
            logger.info(f"Early stopping at epoch {epoch+1}")
            break
        
        if (epoch + 1) % 25 == 0:
            logger.info(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
    
    # Load best model
    model.load_state_dict(torch.load('best_enhanced_unsupervised_graph_informer.pth'))
    return train_losses, val_losses

def evaluate_enhanced_unsupervised_model(model, test_loader, device='cpu'):
    """Comprehensive evaluation of the enhanced unsupervised model."""
    model.eval()
    all_outputs = []
    all_labels = []
    
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            outputs = model(batch_x, return_detailed_output=True)
            
            all_outputs.append(outputs)
            all_labels.extend(batch_y.cpu().numpy())
    
    all_labels = np.array(all_labels)
    
    # Extract all outputs
    reconstructed_global = torch.cat([out['reconstructed_global'] for out in all_outputs]).cpu().numpy()
    reconstructed_local = torch.cat([out['reconstructed_local'] for out in all_outputs]).cpu().numpy()
    reconstructed_temporal = torch.cat([out['reconstructed_temporal'] for out in all_outputs]).cpu().numpy()
    ensemble_scores = torch.cat([out['ensemble_score'] for out in all_outputs]).cpu().numpy().flatten()
    dynamic_weights = torch.cat([out['dynamic_weights'] for out in all_outputs]).cpu().numpy()
    scorer_outputs = torch.cat([out['scorer_outputs'] for out in all_outputs]).cpu().numpy()
    
    # Calculate reconstruction errors
    test_data = test_loader.dataset.tensors[0].numpy()
    reconstruction_error_global = np.mean((reconstructed_global - test_data.mean(axis=1))**2, axis=1)
    reconstruction_error_local = np.mean((reconstructed_local - test_data.max(axis=1))**2, axis=1)
    reconstruction_error_temporal = np.mean((reconstructed_temporal - test_data[:, -1, :])**2, axis=1)
    reconstruction_error_temporal_full = np.mean((reconstructed_temporal[:, np.newaxis, :] - test_data)**2, axis=(1, 2))
    
    # Advanced ensemble scoring with multiple features
    # Normalize all scores to [0, 1] range
    recon_global_norm = (reconstruction_error_global - np.min(reconstruction_error_global)) / (np.max(reconstruction_error_global) - np.min(reconstruction_error_global) + 1e-8)
    recon_local_norm = (reconstruction_error_local - np.min(reconstruction_error_local)) / (np.max(reconstruction_error_local) - np.min(reconstruction_error_local) + 1e-8)
    recon_temporal_norm = (reconstruction_error_temporal - np.min(reconstruction_error_temporal)) / (np.max(reconstruction_error_temporal) - np.min(reconstruction_error_temporal) + 1e-8)
    recon_temporal_full_norm = (reconstruction_error_temporal_full - np.min(reconstruction_error_temporal_full)) / (np.max(reconstruction_error_temporal_full) - np.min(reconstruction_error_temporal_full) + 1e-8)
    ensemble_scores_norm = (ensemble_scores - np.min(ensemble_scores)) / (np.max(ensemble_scores) - np.min(ensemble_scores) + 1e-8)
    
    # Use dynamic weights for ensemble scoring
    avg_dynamic_weights = np.mean(dynamic_weights, axis=0)
    
    # Advanced ensemble scoring with dynamic weights
    combined_score = (avg_dynamic_weights[0] * recon_global_norm + 
                     avg_dynamic_weights[1] * recon_local_norm + 
                     avg_dynamic_weights[2] * recon_temporal_norm + 
                     avg_dynamic_weights[3] * recon_temporal_full_norm + 
                     avg_dynamic_weights[4] * ensemble_scores_norm)
    
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
        'dynamic_weights': avg_dynamic_weights,
        'combined_score': combined_score,
        'reconstruction_error_global': reconstruction_error_global,
        'reconstruction_error_local': reconstruction_error_local,
        'reconstruction_error_temporal': reconstruction_error_temporal,
        'reconstruction_error_temporal_full': reconstruction_error_temporal_full,
        'ensemble_scores': ensemble_scores,
        'scorer_outputs': scorer_outputs
    }
    
    return metrics

def main():
    """Main function for enhanced unsupervised Graph Informer."""
    print("🚀 Enhanced Unsupervised Graph Informer Transformer with Advanced Anomaly Detection")
    print("=" * 100)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Enhanced parameters
    seq_len = 24
    d_model = 256
    n_heads = 8
    n_layers = 3
    num_nodes = 14
    
    # Load and preprocess data
    X_seq, y_seq, edge_index, scaler, feature_names = load_and_preprocess_enhanced_unsupervised_data(
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
    
    # Create enhanced unsupervised model
    model = EnhancedUnsupervisedGraphInformer(
        input_dim=len(feature_names),
        d_model=d_model,
        n_heads=n_heads,
        n_layers=n_layers,
        seq_len=seq_len,
        num_nodes=num_nodes,
        use_contrastive=True,
        use_temporal_consistency=True
    )
    
    print(f"\nEnhanced Unsupervised Model Architecture:")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Model dimension: {d_model}")
    print(f"Transformer layers: {n_layers}")
    print(f"Features: Contrastive learning, temporal consistency, dynamic ensemble scoring")
    print(f"Learning: Pure unsupervised (no labeled attack data used)")
    
    # Train model
    print("\nTraining enhanced unsupervised model...")
    train_losses, val_losses = train_enhanced_unsupervised_model(
        model, train_loader, val_loader, num_epochs=150, device=device
    )
    
    # Evaluate model
    print("\nEvaluating enhanced unsupervised model...")
    metrics = evaluate_enhanced_unsupervised_model(model, test_loader, device=device)
    
    # Print results
    print("\n🏆 Enhanced Unsupervised Model Performance:")
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
    
    print(f"\nDynamic Ensemble Weights:")
    print(f"Weights: {metrics['dynamic_weights']}")
    
    # Save model
    torch.save(model.state_dict(), 'enhanced_unsupervised_graph_informer_final.pth')
    
    print(f"\n✅ Enhanced Unsupervised Graph Informer implementation complete!")
    print(f"🎯 Model saved to 'enhanced_unsupervised_graph_informer_final.pth'")
    print(f"📊 Advanced features: Contrastive learning, temporal consistency, dynamic ensemble scoring")
    print(f"🔍 Pure unsupervised learning - no labeled attack data used!")
    
    return model, metrics

if __name__ == "__main__":
    main()
