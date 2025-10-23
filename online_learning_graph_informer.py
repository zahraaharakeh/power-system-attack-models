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
from torch_geometric.nn import GCNConv, GATConv
from typing import Dict, List, Tuple, Optional
import logging
import queue
import threading
import time
from collections import deque
import json
import os
from datetime import datetime
warnings.filterwarnings('ignore')

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class OnlineLearningBuffer:
    """Buffer for online learning with adaptive sampling."""
    
    def __init__(self, max_size=1000, min_size=100):
        self.max_size = max_size
        self.min_size = min_size
        self.buffer = deque(maxlen=max_size)
        self.anomaly_scores = deque(maxlen=max_size)
        self.timestamps = deque(maxlen=max_size)
        self.adaptation_weights = deque(maxlen=max_size)
        
    def add_sample(self, sample, anomaly_score, timestamp=None, adaptation_weight=1.0):
        """Add a new sample to the buffer."""
        if timestamp is None:
            timestamp = time.time()
            
        self.buffer.append(sample)
        self.anomaly_scores.append(anomaly_score)
        self.timestamps.append(timestamp)
        self.adaptation_weights.append(adaptation_weight)
    
    def get_batch(self, batch_size=32):
        """Get a batch of samples for online learning."""
        if len(self.buffer) < self.min_size:
            return None
            
        # Adaptive sampling based on anomaly scores and recency
        current_time = time.time()
        weights = []
        
        for i, (score, timestamp, adapt_weight) in enumerate(zip(self.anomaly_scores, self.timestamps, self.adaptation_weights)):
            # Recency weight (more recent samples have higher weight)
            recency_weight = 1.0 / (1.0 + (current_time - timestamp) / 3600)  # Decay over hours
            
            # Anomaly weight (higher anomaly scores have higher weight)
            anomaly_weight = 1.0 + score
            
            # Combined weight
            weight = recency_weight * anomaly_weight * adapt_weight
            weights.append(weight)
        
        weights = np.array(weights)
        weights = weights / np.sum(weights)  # Normalize
        
        # Sample indices based on weights
        indices = np.random.choice(len(self.buffer), size=min(batch_size, len(self.buffer)), 
                                 replace=False, p=weights)
        
        batch_samples = [self.buffer[i] for i in indices]
        batch_scores = [self.anomaly_scores[i] for i in indices]
        
        return batch_samples, batch_scores
    
    def get_statistics(self):
        """Get buffer statistics."""
        if not self.buffer:
            return {}
            
        return {
            'size': len(self.buffer),
            'avg_anomaly_score': np.mean(self.anomaly_scores),
            'max_anomaly_score': np.max(self.anomaly_scores),
            'min_anomaly_score': np.min(self.anomaly_scores),
            'recent_samples': len([t for t in self.timestamps if time.time() - t < 3600])
        }

class AdaptiveOnlineOptimizer:
    """Adaptive optimizer for online learning."""
    
    def __init__(self, model, initial_lr=0.001, adaptation_rate=0.01):
        self.model = model
        self.initial_lr = initial_lr
        self.current_lr = initial_lr
        self.adaptation_rate = adaptation_rate
        
        # Separate optimizers for different components
        self.reconstruction_optimizer = torch.optim.AdamW(
            [p for name, p in model.named_parameters() if 'decoder' in name or 'encoder' in name],
            lr=initial_lr, weight_decay=1e-4
        )
        
        self.anomaly_optimizer = torch.optim.AdamW(
            [p for name, p in model.named_parameters() if 'ensemble_scoring' in name or 'anomaly' in name],
            lr=initial_lr * 2, weight_decay=1e-4
        )
        
        self.adaptation_optimizer = torch.optim.AdamW(
            [p for name, p in model.named_parameters() if 'online_adaptation' in name or 'contrastive' in name],
            lr=initial_lr * 0.5, weight_decay=1e-4
        )
        
        # Performance tracking
        self.performance_history = deque(maxlen=100)
        self.adaptation_history = deque(maxlen=50)
        
    def step(self, losses, performance_metric=None):
        """Perform optimization step with adaptive learning rates."""
        # Update reconstruction components
        if 'reconstruction' in losses:
            self.reconstruction_optimizer.zero_grad()
            losses['reconstruction'].backward(retain_graph=True)
            self.reconstruction_optimizer.step()
        
        # Update anomaly detection components
        if 'anomaly' in losses:
            self.anomaly_optimizer.zero_grad()
            losses['anomaly'].backward(retain_graph=True)
            self.anomaly_optimizer.step()
        
        # Update adaptation components
        if 'adaptation' in losses:
            self.adaptation_optimizer.zero_grad()
            losses['adaptation'].backward()
            self.adaptation_optimizer.step()
        
        # Adaptive learning rate adjustment
        if performance_metric is not None:
            self.performance_history.append(performance_metric)
            
            if len(self.performance_history) > 10:
                recent_perf = np.mean(list(self.performance_history)[-5:])
                older_perf = np.mean(list(self.performance_history)[-10:-5])
                
                # Adjust learning rates based on performance
                if recent_perf > older_perf + 0.02:  # Performance improving
                    self.current_lr *= 1.01
                elif recent_perf < older_perf - 0.02:  # Performance degrading
                    self.current_lr *= 0.99
                
                # Update optimizer learning rates
                for optimizer in [self.reconstruction_optimizer, self.anomaly_optimizer, self.adaptation_optimizer]:
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = min(self.current_lr, 0.01)
    
    def get_adaptation_signal(self):
        """Get current adaptation signal."""
        if len(self.performance_history) < 5:
            return 0.0
            
        recent_perf = np.mean(list(self.performance_history)[-3:])
        return recent_perf

class OnlineLearningGraphInformer(nn.Module):
    """Graph Informer with online learning capabilities."""
    
    def __init__(self, input_dim, d_model=256, n_heads=8, n_layers=3, seq_len=24, num_nodes=14, 
                 dropout=0.1, online_buffer_size=1000):
        super(OnlineLearningGraphInformer, self).__init__()
        
        self.input_dim = input_dim
        self.d_model = d_model
        self.seq_len = seq_len
        self.num_nodes = num_nodes
        
        # Base model components (same as enhanced version)
        self.input_projection = nn.Sequential(
            nn.Linear(input_dim, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        self.pos_encoding = self._create_positional_encoding(d_model, seq_len)
        
        self.encoder_layers = nn.ModuleList([
            self._create_enhanced_block(d_model, n_heads, dropout, layer_idx=i)
            for i in range(n_layers)
        ])
        
        self.graph_convs = nn.ModuleList([
            GCNConv(d_model, d_model) for _ in range(2)
        ])
        
        self.gat_convs = nn.ModuleList([
            GATConv(d_model, d_model // 4, heads=4, dropout=dropout) for _ in range(2)
        ])
        
        # Reconstruction decoders
        self.decoder_global = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.LayerNorm(d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, input_dim)
        )
        
        self.decoder_local = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.LayerNorm(d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, input_dim)
        )
        
        # Online learning specific components
        self.online_adaptation = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, d_model),
            nn.Sigmoid()
        )
        
        self.anomaly_detector = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, 1),
            nn.Sigmoid()
        )
        
        # Online learning buffer
        self.online_buffer = OnlineLearningBuffer(max_size=online_buffer_size)
        
        # Performance tracking
        self.performance_history = deque(maxlen=100)
        self.adaptation_count = 0
        
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
    
    def forward(self, x, edge_index=None, return_online_info=False):
        batch_size, seq_len, features = x.shape
        
        # Input projection
        x = self.input_projection(x)
        x = x + self.pos_encoding[:, :seq_len, :]
        
        # Graph convolutions
        if edge_index is not None:
            x_reshaped = x.reshape(-1, self.d_model)
            
            for conv in self.graph_convs:
                x_reshaped = conv(x_reshaped, edge_index)
                x_reshaped = F.relu(x_reshaped)
            
            for gat in self.gat_convs:
                x_reshaped = gat(x_reshaped, edge_index)
                x_reshaped = F.relu(x_reshaped)
            
            x = x_reshaped.reshape(batch_size, seq_len, self.d_model)
        
        # Transformer blocks
        attention_weights = []
        for i, layer in enumerate(self.encoder_layers):
            gate_weight = layer['gate'](x.mean(dim=1)).unsqueeze(1)
            
            attn_out, attn_weights = layer['attention'](x, x, x)
            x = layer['norm1'](x + gate_weight * attn_out)
            
            ff_out = layer['ff'](x)
            x = layer['norm2'](x + gate_weight * ff_out)
            
            attention_weights.append(attn_weights)
        
        # Feature extraction
        global_features = x.mean(dim=1)
        local_features = x.max(dim=1)[0]
        
        # Reconstruction
        reconstructed_global = self.decoder_global(global_features)
        reconstructed_local = self.decoder_local(local_features)
        
        # Anomaly detection
        anomaly_score = self.anomaly_detector(global_features)
        
        # Online adaptation
        adaptation_signal = self.online_adaptation(global_features)
        
        if return_online_info:
            return {
                'reconstructed_global': reconstructed_global,
                'reconstructed_local': reconstructed_local,
                'anomaly_score': anomaly_score,
                'adaptation_signal': adaptation_signal,
                'global_features': global_features,
                'local_features': local_features,
                'attention_weights': attention_weights
            }
        else:
            return (reconstructed_global, reconstructed_local, anomaly_score, 
                   attention_weights, global_features, local_features, adaptation_signal)
    
    def online_update(self, new_sample, anomaly_score, optimizer):
        """Perform online update with new sample."""
        # Add to buffer
        self.online_buffer.add_sample(new_sample, anomaly_score)
        
        # Get batch for online learning
        batch_data = self.online_buffer.get_batch(batch_size=16)
        if batch_data is None:
            return
        
        batch_samples, batch_scores = batch_data
        batch_tensor = torch.stack(batch_samples).to(next(self.parameters()).device)
        
        # Forward pass
        outputs = self.forward(batch_tensor, return_online_info=True)
        
        # Compute losses
        losses = {}
        
        # Reconstruction loss
        recon_loss_global = F.mse_loss(outputs['reconstructed_global'], batch_tensor.mean(dim=1))
        recon_loss_local = F.mse_loss(outputs['reconstructed_local'], batch_tensor.max(dim=1)[0])
        losses['reconstruction'] = 0.6 * recon_loss_global + 0.4 * recon_loss_local
        
        # Anomaly loss (encourage normal data to have low scores)
        normal_mask = torch.tensor([score < 0.5 for score in batch_scores], 
                                 dtype=torch.float, device=batch_tensor.device)
        anomaly_loss = torch.mean(outputs['anomaly_score'].squeeze() * normal_mask)
        losses['anomaly'] = anomaly_loss
        
        # Adaptation loss
        adaptation_loss = torch.mean(outputs['adaptation_signal'])
        losses['adaptation'] = adaptation_loss
        
        # Update model
        optimizer.step(losses, performance_metric=anomaly_loss.item())
        
        # Track performance
        self.performance_history.append(anomaly_loss.item())
        self.adaptation_count += 1

class OnlineLearningManager:
    """Manager for online learning process."""
    
    def __init__(self, model, device='cpu', update_frequency=10):
        self.model = model
        self.device = device
        self.update_frequency = update_frequency
        self.optimizer = AdaptiveOnlineOptimizer(model)
        self.is_running = False
        self.update_thread = None
        self.sample_queue = queue.Queue()
        
    def start_online_learning(self):
        """Start online learning process."""
        self.is_running = True
        self.update_thread = threading.Thread(target=self._online_learning_loop)
        self.update_thread.daemon = True
        self.update_thread.start()
        logger.info("Online learning started")
    
    def stop_online_learning(self):
        """Stop online learning process."""
        self.is_running = False
        if self.update_thread:
            self.update_thread.join()
        logger.info("Online learning stopped")
    
    def add_sample(self, sample, anomaly_score):
        """Add new sample for online learning."""
        self.sample_queue.put((sample, anomaly_score))
    
    def _online_learning_loop(self):
        """Main online learning loop."""
        update_count = 0
        
        while self.is_running:
            try:
                # Get sample from queue (with timeout)
                sample, anomaly_score = self.sample_queue.get(timeout=1.0)
                
                # Perform online update
                self.model.online_update(sample, anomaly_score, self.optimizer)
                update_count += 1
                
                # Log progress
                if update_count % self.update_frequency == 0:
                    buffer_stats = self.model.online_buffer.get_statistics()
                    logger.info(f"Online learning update {update_count}, Buffer stats: {buffer_stats}")
                
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Error in online learning: {e}")
    
    def get_performance_stats(self):
        """Get current performance statistics."""
        return {
            'adaptation_count': self.model.adaptation_count,
            'buffer_stats': self.model.online_buffer.get_statistics(),
            'optimizer_lr': self.optimizer.current_lr,
            'performance_history': list(self.model.performance_history)[-10:] if self.model.performance_history else []
        }

def create_online_learning_malicious_data(benign_data, seq_len=24, num_nodes=14):
    """Generate malicious data for online learning testing."""
    print("Generating malicious data for online learning...")
    
    n_samples = len(benign_data)
    all_attacks = []
    
    # 1. Gradual drift attack
    drift_attack = benign_data.copy()
    for i in range(seq_len, len(benign_data)):
        drift_factor = (i - seq_len) / (len(benign_data) - seq_len)
        noise_scale = 0.01 + 0.15 * drift_factor
        
        for node in range(min(num_nodes, benign_data.shape[1] // 4)):
            if node*4+4 <= benign_data.shape[1]:
                drift = np.random.normal(0, noise_scale, 4)
                drift_attack[i, node*4:(node+1)*4] += drift
    all_attacks.append(drift_attack)
    
    # 2. Intermittent attack
    intermittent_attack = benign_data.copy()
    for i in range(seq_len, len(benign_data)):
        if np.random.random() < 0.1:  # 10% chance of attack
            for node in range(min(num_nodes, benign_data.shape[1] // 4)):
                if node*4+4 <= benign_data.shape[1]:
                    attack_noise = np.random.normal(0, 0.2, 4)
                    intermittent_attack[i, node*4:(node+1)*4] += attack_noise
    all_attacks.append(intermittent_attack)
    
    # 3. Coordinated attack
    coord_attack = benign_data.copy()
    for i in range(seq_len, len(benign_data)):
        if np.random.random() < 0.05:  # 5% chance of coordinated attack
            max_nodes = min(num_nodes, benign_data.shape[1] // 4)
            if max_nodes >= 2:
                attack_size = min(np.random.randint(2, 4), max_nodes)
                attacked_nodes = np.random.choice(range(max_nodes), size=attack_size, replace=False)
                
                for node in attacked_nodes:
                    if node*4+4 <= benign_data.shape[1]:
                        coord_noise = np.random.normal(0, 0.15, 4)
                        coord_attack[i, node*4:(node+1)*4] += coord_noise
    all_attacks.append(coord_attack)
    
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
    
    print(f"Generated {len(malicious_data)} online learning malicious samples")
    return malicious_data

def load_and_preprocess_online_data(benign_file, seq_len=24, num_nodes=14):
    """Load and preprocess data for online learning."""
    print("Loading and preprocessing data for online learning...")
    
    # Load benign data
    benign_df = pd.read_excel(benign_file)
    feature_columns = ['Pd_new', 'Qd_new', 'Vm', 'Va']
    X_benign = benign_df[feature_columns].values
    
    print(f"Benign samples loaded: {len(X_benign)}")
    
    # Generate malicious data
    X_malicious = create_online_learning_malicious_data(X_benign, seq_len, num_nodes)
    
    # Create labels
    y_benign = np.zeros(len(X_benign))
    y_malicious = np.ones(len(X_malicious))
    
    # Combine data
    X = np.vstack([X_benign, X_malicious])
    y = np.concatenate([y_benign, y_malicious])
    
    print(f"Total samples: {len(X)}")
    print(f"Class distribution: Benign={np.sum(y==0)}, Malicious={np.sum(y==1)}")
    
    # Standardize features
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

def train_online_learning_model(model, train_loader, val_loader, num_epochs=100, device='cpu'):
    """Train the online learning model."""
    model = model.to(device)
    
    # Initial training optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)
    
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    patience = 20
    patience_counter = 0
    
    for epoch in range(num_epochs):
        # Training
        model.train()
        total_train_loss = 0
        
        for batch_x, _ in train_loader:
            batch_x = batch_x.to(device)
            optimizer.zero_grad()
            
            reconstructed_global, reconstructed_local, anomaly_score, _, _, _, _ = model(batch_x)
            
            # Reconstruction loss
            recon_loss_global = F.mse_loss(reconstructed_global, batch_x.mean(dim=1))
            recon_loss_local = F.mse_loss(reconstructed_local, batch_x.max(dim=1)[0])
            recon_loss = 0.6 * recon_loss_global + 0.4 * recon_loss_local
            
            # Anomaly regularization
            reconstruction_error = torch.mean((batch_x - reconstructed_global.unsqueeze(1).expand(-1, batch_x.size(1), -1))**2, dim=(1, 2))
            normal_mask = (reconstruction_error < torch.median(reconstruction_error)).float()
            anomaly_reg = torch.mean(anomaly_score.squeeze() * normal_mask)
            
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
                reconstructed_global, reconstructed_local, anomaly_score, _, _, _, _ = model(batch_x)
                
                recon_loss_global = F.mse_loss(reconstructed_global, batch_x.mean(dim=1))
                recon_loss_local = F.mse_loss(reconstructed_local, batch_x.max(dim=1)[0])
                recon_loss = 0.6 * recon_loss_global + 0.4 * recon_loss_local
                
                reconstruction_error = torch.mean((batch_x - reconstructed_global.unsqueeze(1).expand(-1, batch_x.size(1), -1))**2, dim=(1, 2))
                normal_mask = (reconstruction_error < torch.median(reconstruction_error)).float()
                anomaly_reg = torch.mean(anomaly_score.squeeze() * normal_mask)
                
                loss = recon_loss + 0.1 * anomaly_reg
                total_val_loss += loss.item()
        
        avg_val_loss = total_val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        
        scheduler.step(avg_val_loss)
        
        # Early stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            torch.save(model.state_dict(), 'best_online_learning_graph_informer.pth')
        else:
            patience_counter += 1
        
        if patience_counter >= patience:
            logger.info(f"Early stopping at epoch {epoch+1}")
            break
        
        if (epoch + 1) % 20 == 0:
            logger.info(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
    
    # Load best model
    model.load_state_dict(torch.load('best_online_learning_graph_informer.pth'))
    return train_losses, val_losses

def evaluate_online_learning_model(model, test_loader, device='cpu'):
    """Evaluate the online learning model."""
    model.eval()
    all_reconstructions_global = []
    all_reconstructions_local = []
    all_anomaly_scores = []
    all_labels = []
    
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            reconstructed_global, reconstructed_local, anomaly_score, _, _, _, _ = model(batch_x)
            
            all_reconstructions_global.extend(reconstructed_global.cpu().numpy())
            all_reconstructions_local.extend(reconstructed_local.cpu().numpy())
            all_anomaly_scores.extend(anomaly_score.cpu().numpy())
            all_labels.extend(batch_y.cpu().numpy())
    
    # Convert to numpy arrays
    all_reconstructions_global = np.array(all_reconstructions_global)
    all_reconstructions_local = np.array(all_reconstructions_local)
    all_anomaly_scores = np.array(all_anomaly_scores).flatten()
    all_labels = np.array(all_labels)
    
    # Calculate reconstruction errors
    test_data = test_loader.dataset.tensors[0].numpy()
    reconstruction_error_global = np.mean((all_reconstructions_global - test_data.mean(axis=1))**2, axis=1)
    reconstruction_error_local = np.mean((all_reconstructions_local - test_data.max(axis=1))**2, axis=1)
    
    # Ensemble scoring
    recon_global_norm = (reconstruction_error_global - np.min(reconstruction_error_global)) / (np.max(reconstruction_error_global) - np.min(reconstruction_error_global) + 1e-8)
    recon_local_norm = (reconstruction_error_local - np.min(reconstruction_error_local)) / (np.max(reconstruction_error_local) - np.min(reconstruction_error_local) + 1e-8)
    anomaly_scores_norm = (all_anomaly_scores - np.min(all_anomaly_scores)) / (np.max(all_anomaly_scores) - np.min(all_anomaly_scores) + 1e-8)
    
    combined_score = 0.4 * recon_global_norm + 0.3 * recon_local_norm + 0.3 * anomaly_scores_norm
    
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
        'combined_score': combined_score,
        'reconstruction_error_global': reconstruction_error_global,
        'reconstruction_error_local': reconstruction_error_local,
        'anomaly_scores': all_anomaly_scores
    }
    
    return metrics

def simulate_online_learning(model, test_data, test_labels, online_manager, num_samples=100):
    """Simulate online learning with streaming data."""
    print("Simulating online learning with streaming data...")
    
    model.eval()
    predictions = []
    true_labels = []
    anomaly_scores_history = []
    
    # Start online learning
    online_manager.start_online_learning()
    
    for i in range(min(num_samples, len(test_data))):
        sample = test_data[i:i+1]  # Single sample
        true_label = test_labels[i]
        
        # Get prediction
        with torch.no_grad():
            reconstructed_global, reconstructed_local, anomaly_score, _, _, _, _ = model(sample)
            
            # Calculate reconstruction error
            recon_error = F.mse_loss(reconstructed_global, sample.mean(dim=1))
            combined_score = 0.7 * recon_error.item() + 0.3 * anomaly_score.item()
            
            # Make prediction
            prediction = 1 if combined_score > 0.5 else 0
        
        predictions.append(prediction)
        true_labels.append(true_label)
        anomaly_scores_history.append(combined_score)
        
        # Add to online learning
        online_manager.add_sample(sample.squeeze(0), combined_score)
        
        # Log progress
        if (i + 1) % 20 == 0:
            current_accuracy = accuracy_score(true_labels, predictions)
            stats = online_manager.get_performance_stats()
            logger.info(f"Online learning sample {i+1}, Accuracy: {current_accuracy:.4f}, Buffer size: {stats['buffer_stats']['size']}")
    
    # Stop online learning
    online_manager.stop_online_learning()
    
    # Final evaluation
    final_accuracy = accuracy_score(true_labels, predictions)
    final_precision = precision_score(true_labels, predictions, average='weighted')
    final_recall = recall_score(true_labels, predictions, average='weighted')
    final_f1 = f1_score(true_labels, predictions, average='weighted')
    
    return {
        'accuracy': final_accuracy,
        'precision': final_precision,
        'recall': final_recall,
        'f1_score': final_f1,
        'predictions': predictions,
        'true_labels': true_labels,
        'anomaly_scores_history': anomaly_scores_history,
        'online_stats': online_manager.get_performance_stats()
    }

def main():
    """Main function for online learning Graph Informer."""
    print("🚀 Online Learning Graph Informer Transformer for Continuous Adaptation")
    print("=" * 80)
    
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
    X_seq, y_seq, edge_index, scaler, feature_names = load_and_preprocess_online_data(
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
    
    # Create online learning model
    model = OnlineLearningGraphInformer(
        input_dim=len(feature_names),
        d_model=d_model,
        n_heads=n_heads,
        n_layers=n_layers,
        seq_len=seq_len,
        num_nodes=num_nodes,
        online_buffer_size=500
    )
    
    print(f"\nOnline Learning Model Architecture:")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Model dimension: {d_model}")
    print(f"Transformer layers: {n_layers}")
    print(f"Features: Online learning, adaptive optimization, continuous adaptation")
    
    # Train model
    print("\nTraining online learning model...")
    train_losses, val_losses = train_online_learning_model(
        model, train_loader, val_loader, num_epochs=100, device=device
    )
    
    # Evaluate model
    print("\nEvaluating online learning model...")
    metrics = evaluate_online_learning_model(model, test_loader, device=device)
    
    # Print results
    print("\n🏆 Online Learning Model Performance:")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"F1-Score: {metrics['f1_score']:.4f}")
    print(f"ROC-AUC: {metrics['roc_auc']:.4f}")
    print(f"Optimal Threshold: {metrics['threshold']:.4f}")
    
    # Simulate online learning
    print("\nSimulating online learning with streaming data...")
    online_manager = OnlineLearningManager(model, device=device, update_frequency=5)
    
    online_results = simulate_online_learning(
        model, X_test, y_test, online_manager, num_samples=200
    )
    
    print(f"\n🔄 Online Learning Results:")
    print(f"Final Accuracy: {online_results['accuracy']:.4f}")
    print(f"Final Precision: {online_results['precision']:.4f}")
    print(f"Final Recall: {online_results['recall']:.4f}")
    print(f"Final F1-Score: {online_results['f1_score']:.4f}")
    print(f"Adaptation Count: {online_results['online_stats']['adaptation_count']}")
    print(f"Buffer Size: {online_results['online_stats']['buffer_stats']['size']}")
    
    # Save model
    torch.save(model.state_dict(), 'online_learning_graph_informer_final.pth')
    
    print(f"\n✅ Online Learning Graph Informer implementation complete!")
    print(f"🎯 Model saved to 'online_learning_graph_informer_final.pth'")
    print(f"🔄 Features: Continuous adaptation, online learning, adaptive optimization")
    
    return model, metrics, online_results

if __name__ == "__main__":
    main()
