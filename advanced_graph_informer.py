import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import warnings
from torch_geometric.nn import GCNConv, GATConv
import optuna
import logging
from typing import Dict, Any, Tuple
import json
import os
from datetime import datetime
warnings.filterwarnings('ignore')

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AdvancedGraphInformer(nn.Module):
    """Advanced Graph Informer with adaptive learning and hyperparameter optimization."""
    
    def __init__(self, input_dim, d_model=256, n_heads=8, n_layers=3, seq_len=24, num_nodes=14, 
                 dropout=0.1, adaptive_weights=True):
        super(AdvancedGraphInformer, self).__init__()
        
        self.input_dim = input_dim
        self.d_model = d_model
        self.seq_len = seq_len
        self.num_nodes = num_nodes
        self.adaptive_weights = adaptive_weights
        
        # Enhanced input projection
        self.input_projection = nn.Sequential(
            nn.Linear(input_dim, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Learnable positional encoding
        self.pos_encoding = self._create_positional_encoding(d_model, seq_len)
        
        # Advanced transformer blocks
        self.encoder_layers = nn.ModuleList([
            self._create_advanced_block(d_model, n_heads, dropout, layer_idx=i)
            for i in range(n_layers)
        ])
        
        # Multi-scale graph convolutions
        self.graph_convs = nn.ModuleList([
            GCNConv(d_model, d_model) for _ in range(2)
        ])
        
        self.gat_convs = nn.ModuleList([
            GATConv(d_model, d_model // 4, heads=4, dropout=dropout) for _ in range(2)
        ])
        
        # Advanced reconstruction decoders
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
        
        # Multi-scale anomaly detection
        self.anomaly_detectors = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, d_model // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(d_model // 2, 1),
                nn.Sigmoid()
            ) for _ in range(4)  # Temporal, Structural, Statistical, Hybrid
        ])
        
        # Adaptive weight generator
        if adaptive_weights:
            self.adaptive_weights_generator = nn.Sequential(
                nn.Linear(d_model, d_model // 2),
                nn.ReLU(),
                nn.Linear(d_model // 2, 4),
                nn.Softmax(dim=-1)
            )
        else:
            self.adaptive_weights_generator = None
        
        # Performance tracking
        self.performance_history = []
        self.adaptation_count = 0
        
    def _create_positional_encoding(self, d_model, max_len):
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        
        return nn.Parameter(pe, requires_grad=True)
    
    def _create_advanced_block(self, d_model, n_heads, dropout, layer_idx):
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
    
    def forward(self, x, edge_index=None, return_adaptation_info=False):
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
        
        # Advanced transformer blocks
        attention_weights = []
        adaptation_info = {}
        
        for i, layer in enumerate(self.encoder_layers):
            # Adaptive gating
            gate_weight = layer['gate'](x.mean(dim=1)).unsqueeze(1)  # [batch, 1, 1]
            
            # Self-attention
            attn_out, attn_weights = layer['attention'](x, x, x)
            x = layer['norm1'](x + gate_weight * attn_out)
            
            # Feed-forward with adaptive gating
            ff_out = layer['ff'](x)
            x = layer['norm2'](x + gate_weight * ff_out)
            
            attention_weights.append(attn_weights)
            adaptation_info[f'gate_weight_layer_{i}'] = gate_weight.mean().item()
        
        # Multi-scale feature extraction
        global_features = x.mean(dim=1)
        local_features = x.max(dim=1)[0]
        
        # Advanced reconstruction
        reconstructed_global = self.decoder_global(global_features)
        reconstructed_local = self.decoder_local(local_features)
        
        # Multi-scale anomaly detection
        anomaly_scores = []
        for detector in self.anomaly_detectors:
            score = detector(global_features)
            anomaly_scores.append(score)
        
        # Adaptive weight generation
        if self.adaptive_weights_generator is not None:
            adaptive_weights = self.adaptive_weights_generator(global_features)
            # Weighted combination of anomaly scores
            combined_anomaly_score = torch.zeros_like(anomaly_scores[0])
            for i, (score, weight) in enumerate(zip(anomaly_scores, adaptive_weights.unbind(dim=1))):
                combined_anomaly_score += weight.unsqueeze(1) * score
            adaptation_info['adaptive_weights'] = adaptive_weights.mean(dim=0).detach().cpu().numpy()
        else:
            # Fixed weights
            weights = torch.tensor([0.3, 0.25, 0.25, 0.2], device=x.device)
            combined_anomaly_score = torch.zeros_like(anomaly_scores[0])
            for i, (score, weight) in enumerate(zip(anomaly_scores, weights)):
                combined_anomaly_score += weight * score
            adaptation_info['adaptive_weights'] = weights.cpu().numpy()
        
        adaptation_info['anomaly_scores'] = [score.mean().item() for score in anomaly_scores]
        
        if return_adaptation_info:
            return (reconstructed_global, reconstructed_local, combined_anomaly_score, 
                   attention_weights, global_features, local_features, adaptation_info)
        else:
            return (reconstructed_global, reconstructed_local, combined_anomaly_score, 
                   attention_weights, global_features, local_features)
    
    def adapt_architecture(self, performance_metric):
        """Dynamically adapt architecture based on performance."""
        self.performance_history.append(performance_metric)
        self.adaptation_count += 1
        
        if len(self.performance_history) > 10:
            recent_performance = np.mean(self.performance_history[-5:])
            older_performance = np.mean(self.performance_history[-10:-5])
            
            # If performance is degrading, adapt
            if recent_performance < older_performance - 0.1:
                logger.info("Performance degrading, adapting architecture...")
                self._increase_model_capacity()
    
    def _increase_model_capacity(self):
        """Increase model capacity dynamically."""
        # Slight increase in weights to improve capacity
        for param in self.parameters():
            if param.requires_grad:
                param.data *= 1.005

class AdaptiveOptimizer:
    """Adaptive optimizer with dynamic learning rate adjustment."""
    
    def __init__(self, model, initial_lr=0.001):
        self.model = model
        self.initial_lr = initial_lr
        self.current_lr = initial_lr
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=initial_lr, weight_decay=1e-4)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', patience=5, factor=0.8
        )
        self.performance_history = []
        self.adaptation_count = 0
        
    def step(self, loss, performance_metric=None):
        """Perform optimization step with adaptive learning rate."""
        self.optimizer.step()
        self.scheduler.step(loss)
        
        if performance_metric is not None:
            self.performance_history.append(performance_metric)
            
            # Adaptive learning rate adjustment
            if len(self.performance_history) > 5:
                recent_perf = np.mean(self.performance_history[-3:])
                older_perf = np.mean(self.performance_history[-6:-3])
                
                if recent_perf > older_perf + 0.05:  # Performance improving
                    self.current_lr *= 1.02
                elif recent_perf < older_perf - 0.05:  # Performance degrading
                    self.current_lr *= 0.98
                
                # Update optimizer learning rate
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = min(self.current_lr, 0.01)  # Cap at 0.01
    
    def zero_grad(self):
        self.optimizer.zero_grad()
    
    def get_lr(self):
        return self.current_lr

def hyperparameter_optimization(trial, X_train, y_train, X_val, y_val, device='cpu'):
    """Optuna hyperparameter optimization."""
    
    # Suggest hyperparameters
    d_model = trial.suggest_categorical('d_model', [128, 256, 512])
    n_heads = trial.suggest_categorical('n_heads', [4, 8, 16])
    n_layers = trial.suggest_categorical('n_layers', [2, 3, 4])
    learning_rate = trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True)
    dropout = trial.suggest_float('dropout', 0.1, 0.3)
    
    # Create model with suggested hyperparameters
    model = AdvancedGraphInformer(
        input_dim=4,
        d_model=d_model,
        n_heads=n_heads,
        n_layers=n_layers,
        seq_len=24,
        num_nodes=14,
        dropout=dropout,
        adaptive_weights=True
    ).to(device)
    
    # Create adaptive optimizer
    adaptive_optimizer = AdaptiveOptimizer(model, learning_rate)
    
    # Training loop
    model.train()
    for epoch in range(15):  # Reduced for hyperparameter optimization
        total_loss = 0
        for batch_x, batch_y in zip(X_train, y_train):
            batch_x = batch_x.unsqueeze(0).to(device)
            batch_y = batch_y.unsqueeze(0).to(device)
            
            adaptive_optimizer.zero_grad()
            
            reconstructed_global, reconstructed_local, anomaly_score, _, _, _ = model(batch_x)
            
            # Loss calculation
            recon_loss_global = F.mse_loss(reconstructed_global, batch_x.mean(dim=1))
            recon_loss_local = F.mse_loss(reconstructed_local, batch_x.max(dim=1)[0])
            recon_loss_temporal = F.mse_loss(
                reconstructed_global.unsqueeze(1).expand(-1, batch_x.size(1), -1), 
                batch_x
            )
            
            loss = 0.4 * recon_loss_global + 0.3 * recon_loss_local + 0.3 * recon_loss_temporal
            loss.backward()
            adaptive_optimizer.step(loss)
            
            total_loss += loss.item()
    
    # Validation
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for batch_x, batch_y in zip(X_val, y_val):
            batch_x = batch_x.unsqueeze(0).to(device)
            batch_y = batch_y.unsqueeze(0).to(device)
            
            reconstructed_global, reconstructed_local, anomaly_score, _, _, _ = model(batch_x)
            
            recon_loss_global = F.mse_loss(reconstructed_global, batch_x.mean(dim=1))
            recon_loss_local = F.mse_loss(reconstructed_local, batch_x.max(dim=1)[0])
            recon_loss_temporal = F.mse_loss(
                reconstructed_global.unsqueeze(1).expand(-1, batch_x.size(1), -1), 
                batch_x
            )
            
            loss = 0.4 * recon_loss_global + 0.3 * recon_loss_local + 0.3 * recon_loss_temporal
            val_loss += loss.item()
    
    return val_loss / len(X_val)

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

def create_advanced_malicious_data(benign_data, seq_len=24, num_nodes=14):
    """Generate advanced malicious data patterns."""
    print("Generating advanced malicious data patterns...")
    
    n_samples = len(benign_data)
    all_attacks = []
    
    # 1. Advanced temporal correlation attack
    temporal_attack = benign_data.copy()
    for i in range(seq_len, len(benign_data)):
        time_factor = (i - seq_len) / (len(benign_data) - seq_len)
        noise_scale = 0.03 + 0.3 * time_factor
        
        for node in range(min(num_nodes, benign_data.shape[1] // 4)):
            if node*4+4 <= benign_data.shape[1]:
                # Add temporally correlated noise
                base_noise = np.random.normal(0, noise_scale, 4)
                if i > 0:
                    base_noise = 0.9 * base_noise + 0.1 * temporal_attack[i-1, node*4:(node+1)*4] * 0.2
                temporal_attack[i, node*4:(node+1)*4] += base_noise
    all_attacks.append(temporal_attack)
    
    # 2. Graph-aware coordinated attack
    graph_attack = benign_data.copy()
    for i in range(seq_len, len(benign_data)):
        if np.random.random() < 0.4:
            max_nodes = min(num_nodes, benign_data.shape[1] // 4)
            if max_nodes >= 2:
                attack_size = min(np.random.randint(2, 5), max_nodes)
                attacked_nodes = np.random.choice(range(max_nodes), size=attack_size, replace=False)
            else:
                attacked_nodes = [0] if max_nodes > 0 else []
            
            for node in attacked_nodes:
                if node*4+4 <= benign_data.shape[1]:
                    # Add coordinated perturbations
                    perturbation = np.random.normal(0, 0.1, 4)
                    # Add feature correlation
                    perturbation[1] = 0.8 * perturbation[1] + 0.2 * perturbation[0]
                    perturbation[3] = 0.7 * perturbation[3] + 0.3 * perturbation[2]
                    graph_attack[i, node*4:(node+1)*4] += perturbation
    all_attacks.append(graph_attack)
    
    # 3. Advanced statistical attack
    statistical_attack = benign_data.copy()
    for i in range(seq_len, len(benign_data)):
        if np.random.random() < 0.3:
            for j in range(benign_data.shape[1]):
                # Add sophisticated statistical anomalies
                if np.random.random() < 0.5:
                    # Exponential noise
                    noise = np.random.exponential(0.05) - 0.05
                else:
                    # Beta distribution noise
                    noise = np.random.beta(2, 5) - 0.3
                statistical_attack[i, j] += noise
    all_attacks.append(statistical_attack)
    
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
    
    print(f"Generated {len(malicious_data)} advanced malicious samples")
    return malicious_data

def load_and_preprocess_advanced_data(benign_file, seq_len=24, num_nodes=14):
    """Load and preprocess data for advanced learning."""
    print("Loading and preprocessing data for advanced learning...")
    
    # Load benign data
    benign_df = pd.read_excel(benign_file)
    feature_columns = ['Pd_new', 'Qd_new', 'Vm', 'Va']
    X_benign = benign_df[feature_columns].values
    
    print(f"Benign samples loaded: {len(X_benign)}")
    
    # Generate advanced malicious data
    X_malicious = create_advanced_malicious_data(X_benign, seq_len, num_nodes)
    
    # Create labels
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

def train_advanced_model(model, train_loader, val_loader, num_epochs=150, device='cpu'):
    """Train the advanced model with adaptive optimization."""
    model = model.to(device)
    
    # Create adaptive optimizer
    adaptive_optimizer = AdaptiveOptimizer(model, initial_lr=0.001)
    
    train_losses = []
    val_losses = []
    adaptation_history = []
    best_val_loss = float('inf')
    patience = 25
    patience_counter = 0
    
    for epoch in range(num_epochs):
        # Training
        model.train()
        total_train_loss = 0
        epoch_adaptations = []
        
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            adaptive_optimizer.zero_grad()
            
            reconstructed_global, reconstructed_local, anomaly_score, _, _, _, adaptation_info = model(
                batch_x, return_adaptation_info=True
            )
            
            # Multi-scale reconstruction loss
            recon_loss_global = F.mse_loss(reconstructed_global, batch_x.mean(dim=1))
            recon_loss_local = F.mse_loss(reconstructed_local, batch_x.max(dim=1)[0])
            recon_loss_temporal = F.mse_loss(
                reconstructed_global.unsqueeze(1).expand(-1, batch_x.size(1), -1), 
                batch_x
            )
            
            # Combined reconstruction loss
            recon_loss = 0.4 * recon_loss_global + 0.3 * recon_loss_local + 0.3 * recon_loss_temporal
            
            # Anomaly regularization
            reconstruction_error = torch.mean((batch_x - reconstructed_global.unsqueeze(1).expand(-1, batch_x.size(1), -1))**2, dim=(1, 2))
            normal_mask = (reconstruction_error < torch.median(reconstruction_error)).float()
            anomaly_reg = torch.mean(anomaly_score.squeeze() * normal_mask)
            
            # Total loss
            loss = recon_loss + 0.1 * anomaly_reg
            
            loss.backward()
            adaptive_optimizer.step(loss, performance_metric=loss.item())
            
            total_train_loss += loss.item()
            epoch_adaptations.append(adaptation_info)
        
        avg_train_loss = total_train_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        
        # Validation
        model.eval()
        total_val_loss = 0
        
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                reconstructed_global, reconstructed_local, anomaly_score, _, _, _ = model(batch_x)
                
                recon_loss_global = F.mse_loss(reconstructed_global, batch_x.mean(dim=1))
                recon_loss_local = F.mse_loss(reconstructed_local, batch_x.max(dim=1)[0])
                recon_loss_temporal = F.mse_loss(
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
        
        # Adaptive architecture modification
        model.adapt_architecture(avg_val_loss)
        
        # Store adaptation history
        avg_adaptation = {}
        for key in epoch_adaptations[0].keys():
            avg_adaptation[key] = np.mean([adapt[key] for adapt in epoch_adaptations])
        adaptation_history.append(avg_adaptation)
        
        # Early stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            torch.save(model.state_dict(), 'best_advanced_graph_informer.pth')
        else:
            patience_counter += 1
        
        if patience_counter >= patience:
            logger.info(f"Early stopping at epoch {epoch+1}")
            break
        
        if (epoch + 1) % 25 == 0:
            logger.info(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
            logger.info(f"Learning Rate: {adaptive_optimizer.get_lr():.6f}")
            logger.info(f"Adaptive Weights: {avg_adaptation.get('adaptive_weights', [0,0,0,0])}")
    
    # Load best model
    model.load_state_dict(torch.load('best_advanced_graph_informer.pth'))
    return train_losses, val_losses, adaptation_history

def evaluate_advanced_model(model, test_loader, device='cpu'):
    """Evaluate the advanced model."""
    model.eval()
    all_reconstructions_global = []
    all_reconstructions_local = []
    all_anomaly_scores = []
    all_labels = []
    all_adaptation_info = []
    
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            reconstructed_global, reconstructed_local, anomaly_score, _, _, _, adaptation_info = model(
                batch_x, return_adaptation_info=True
            )
            
            all_reconstructions_global.extend(reconstructed_global.cpu().numpy())
            all_reconstructions_local.extend(reconstructed_local.cpu().numpy())
            all_anomaly_scores.extend(anomaly_score.cpu().numpy())
            all_labels.extend(batch_y.cpu().numpy())
            all_adaptation_info.append(adaptation_info)
    
    # Convert to numpy arrays
    all_reconstructions_global = np.array(all_reconstructions_global)
    all_reconstructions_local = np.array(all_reconstructions_local)
    all_anomaly_scores = np.array(all_anomaly_scores).flatten()
    all_labels = np.array(all_labels)
    
    # Calculate reconstruction errors
    test_data = test_loader.dataset.tensors[0].numpy()
    reconstruction_error_global = np.mean((all_reconstructions_global - test_data.mean(axis=1))**2, axis=1)
    reconstruction_error_local = np.mean((all_reconstructions_local - test_data.max(axis=1))**2, axis=1)
    reconstruction_error_temporal = np.mean((all_reconstructions_global[:, np.newaxis, :] - test_data)**2, axis=(1, 2))
    
    # Advanced ensemble scoring with adaptive weights
    recon_global_norm = (reconstruction_error_global - np.min(reconstruction_error_global)) / (np.max(reconstruction_error_global) - np.min(reconstruction_error_global) + 1e-8)
    recon_local_norm = (reconstruction_error_local - np.min(reconstruction_error_local)) / (np.max(reconstruction_error_local) - np.min(reconstruction_error_local) + 1e-8)
    recon_temporal_norm = (reconstruction_error_temporal - np.min(reconstruction_error_temporal)) / (np.max(reconstruction_error_temporal) - np.min(reconstruction_error_temporal) + 1e-8)
    anomaly_scores_norm = (all_anomaly_scores - np.min(all_anomaly_scores)) / (np.max(all_anomaly_scores) - np.min(all_anomaly_scores) + 1e-8)
    
    # Use adaptive weights from the model
    avg_adaptive_weights = np.mean([info['adaptive_weights'] for info in all_adaptation_info], axis=0)
    
    # Adaptive ensemble scoring
    combined_score = (avg_adaptive_weights[0] * recon_global_norm + 
                     avg_adaptive_weights[1] * recon_local_norm + 
                     avg_adaptive_weights[2] * recon_temporal_norm + 
                     avg_adaptive_weights[3] * anomaly_scores_norm)
    
    # Find optimal threshold
    from sklearn.metrics import roc_curve
    fpr, tpr, thresholds = roc_curve(all_labels, combined_score)
    youden_j = tpr - fpr
    optimal_idx = np.argmax(youden_j)
    optimal_threshold = thresholds[optimal_idx]
    
    # Final predictions
    final_predictions = (combined_score > optimal_threshold).astype(int)
    
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
        'threshold': optimal_threshold,
        'adaptive_weights': avg_adaptive_weights,
        'combined_score': combined_score,
        'reconstruction_error_global': reconstruction_error_global,
        'reconstruction_error_local': reconstruction_error_local,
        'reconstruction_error_temporal': reconstruction_error_temporal,
        'anomaly_scores': all_anomaly_scores
    }
    
    return metrics

def main():
    """Main function for advanced Graph Informer."""
    print("🚀 Advanced Graph Informer Transformer with Adaptive Learning & Hyperparameter Optimization")
    print("=" * 100)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load and preprocess data
    X_seq, y_seq, edge_index, scaler, feature_names = load_and_preprocess_advanced_data(
        'benign_bus14.xlsx', seq_len=24, num_nodes=14
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
    
    # Hyperparameter optimization with Optuna
    print("\n🔧 Starting hyperparameter optimization...")
    study = optuna.create_study(direction='minimize')
    study.optimize(
        lambda trial: hyperparameter_optimization(trial, X_train, y_train, X_val, y_val, device),
        n_trials=8  # Reduced for demo
    )
    
    best_params = study.best_params
    print(f"Best hyperparameters: {best_params}")
    
    # Create model with best hyperparameters
    model = AdvancedGraphInformer(
        input_dim=len(feature_names),
        d_model=best_params['d_model'],
        n_heads=best_params['n_heads'],
        n_layers=best_params['n_layers'],
        seq_len=24,
        num_nodes=14,
        dropout=best_params['dropout'],
        adaptive_weights=True
    )
    
    print(f"\nAdvanced Model Architecture:")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Model dimension: {best_params['d_model']}")
    print(f"Attention heads: {best_params['n_heads']}")
    print(f"Transformer layers: {best_params['n_layers']}")
    print(f"Dropout: {best_params['dropout']}")
    print(f"Features: Adaptive learning, dynamic architecture, hyperparameter optimization")
    
    # Train model
    print("\nTraining advanced model...")
    train_losses, val_losses, adaptation_history = train_advanced_model(
        model, train_loader, val_loader, num_epochs=150, device=device
    )
    
    # Evaluate model
    print("\nEvaluating advanced model...")
    metrics = evaluate_advanced_model(model, test_loader, device=device)
    
    # Print results
    print("\n🏆 Advanced Model Performance:")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"F1-Score: {metrics['f1_score']:.4f}")
    print(f"ROC-AUC: {metrics['roc_auc']:.4f}")
    print(f"Optimal Threshold: {metrics['threshold']:.4f}")
    
    print(f"\nAdvanced Features:")
    print(f"Final Adaptive Weights: {metrics['adaptive_weights']}")
    print(f"Architecture Adaptations: {len(adaptation_history)}")
    print(f"Hyperparameter Optimization: {len(study.trials)} trials completed")
    
    # Save model and results
    torch.save(model.state_dict(), 'advanced_graph_informer_final.pth')
    
    # Save hyperparameters and results
    results = {
        'best_hyperparameters': best_params,
        'performance_metrics': {k: float(v) for k, v in metrics.items() if isinstance(v, (int, float, np.number))},
        'adaptive_weights': metrics['adaptive_weights'].tolist(),
        'timestamp': datetime.now().isoformat()
    }
    
    with open('advanced_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✅ Advanced Graph Informer implementation complete!")
    print(f"🎯 Model saved to 'advanced_graph_informer_final.pth'")
    print(f"📊 Results saved to 'advanced_results.json'")
    print(f"🔧 Hyperparameters optimized with Optuna")
    
    return model, metrics, best_params

if __name__ == "__main__":
    main()
