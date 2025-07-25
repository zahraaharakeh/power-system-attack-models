import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.data import Data, DataLoader
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from torch.optim.lr_scheduler import ReduceLROnPlateau
import optuna

class GNNModel(nn.Module):
    def __init__(self, num_features, num_classes, hidden_channels=64):
        super(GNNModel, self).__init__()
        self.conv1 = GCNConv(num_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.conv3 = GCNConv(hidden_channels, hidden_channels)
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_channels // 2, num_classes)
        )
    
    def forward(self, x, edge_index, batch):
        # Graph convolution layers
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.2, training=self.training)
        
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.2, training=self.training)
        
        x = self.conv3(x, edge_index)
        x = F.relu(x)
        
        # Global pooling
        x = global_mean_pool(x, batch)
        
        # Classification
        x = self.classifier(x)
        return x

def create_graph_data(features, labels):
    """Create PyTorch Geometric Data objects from features and edge indices."""
    data_list = []
    for i in range(len(features)):
        # Create a single node graph for each sample
        x = torch.FloatTensor(features[i]).view(1, -1)  # Shape: (1, num_features)
        # Create self-loop edge index for single node
        edge_idx = torch.LongTensor([[0], [0]])  # Always use self-loop
        data = Data(
            x=x,
            edge_index=edge_idx,
            y=torch.LongTensor([labels[i]])
        )
        data_list.append(data)
    return data_list

def generate_malicious_data(benign_data, num_samples=None):
    """Generate synthetic malicious data by perturbing benign data."""
    if num_samples is None:
        num_samples = len(benign_data)
    
    # Create malicious samples by adding different types of perturbations
    malicious_data = []
    
    # Type 1: Random noise attack
    noise_attack = benign_data + np.random.normal(0, 0.5, benign_data.shape)
    malicious_data.append(noise_attack)
    
    # Type 2: Scaling attack (amplify values)
    scale_attack = benign_data * np.random.uniform(1.5, 2.0, benign_data.shape)
    malicious_data.append(scale_attack)
    
    # Type 3: Offset attack (shift values)
    offset_attack = benign_data + np.random.uniform(0.5, 1.0, benign_data.shape)
    malicious_data.append(offset_attack)
    
    # Combine all attack types
    malicious_data = np.vstack(malicious_data)
    
    # Randomly select num_samples if we generated more
    if len(malicious_data) > num_samples:
        indices = np.random.choice(len(malicious_data), num_samples, replace=False)
        malicious_data = malicious_data[indices]
    
    return malicious_data

def load_and_preprocess_data(benign_file):
    """Load and preprocess the data from benign file and generate malicious data."""
    # Load benign data
    benign_df = pd.read_excel(benign_file)
    
    # Extract features
    feature_columns = ['Pd_new', 'Qd_new', 'Vm', 'Va']
    X_benign = benign_df[feature_columns].values
    
    # Generate malicious data
    X_malicious = generate_malicious_data(X_benign)
    
    # Create labels (0 for benign, 1 for malicious)
    y_benign = np.zeros(len(X_benign))
    y_malicious = np.ones(len(X_malicious))
    
    # Combine data
    X = np.vstack([X_benign, X_malicious])
    y = np.concatenate([y_benign, y_malicious])
    
    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    return X_scaled, y, scaler

def train_model(model, train_loader, val_loader, num_epochs=10000, device='cuda'):
    """Train the GNN model."""
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)
    
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    patience = 20
    patience_counter = 0
    
    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            out = model(batch.x, batch.edge_index, batch.batch)
            loss = criterion(out, batch.y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                out = model(batch.x, batch.edge_index, batch.batch)
                loss = criterion(out, batch.y)
                val_loss += loss.item()
        
        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        
        # Learning rate scheduling
        scheduler.step(val_loss)
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), 'best_gnn_model.pth')
        else:
            patience_counter += 1
        
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
        
        if patience_counter >= patience:
            print(f'Early stopping at epoch {epoch+1}')
            break
    
    # Load best model
    model.load_state_dict(torch.load('best_gnn_model.pth'))
    return train_losses, val_losses

def evaluate_model(model, test_loader, device='cuda'):
    """Evaluate the model and return metrics."""
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            out = model(batch.x, batch.edge_index, batch.batch)
            pred = out.argmax(dim=1)
            all_preds.extend(pred.cpu().numpy())
            all_labels.extend(batch.y.cpu().numpy())
    
    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='weighted')
    recall = recall_score(all_labels, all_preds, average='weighted')
    f1 = f1_score(all_labels, all_preds, average='weighted')
    cm = confusion_matrix(all_labels, all_preds)
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'confusion_matrix': cm
    }

def plot_results(train_losses, val_losses, metrics):
    """Plot training history and confusion matrix."""
    # Plot training history
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training History')
    plt.legend()
    
    # Plot confusion matrix
    plt.subplot(1, 2, 2)
    sns.heatmap(metrics['confusion_matrix'], annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    
    plt.tight_layout()
    plt.savefig('gnn_results.png')
    plt.close()
    
    # Save metrics to file
    with open('gnn_metrics.txt', 'w') as f:
        f.write(f"Model Performance Metrics:\n")
        f.write(f"Accuracy: {metrics['accuracy']:.4f}\n")
        f.write(f"Precision: {metrics['precision']:.4f}\n")
        f.write(f"Recall: {metrics['recall']:.4f}\n")
        f.write(f"F1 Score: {metrics['f1_score']:.4f}\n")

def gnn_objective(trial):
    hidden_channels = trial.suggest_categorical('hidden_channels', [32, 64, 128, 256])
    num_layers = trial.suggest_int('num_layers', 2, 4)
    dropout = trial.suggest_float('dropout', 0.1, 0.5)
    lr = trial.suggest_loguniform('lr', 1e-4, 1e-2)
    weight_decay = trial.suggest_loguniform('weight_decay', 1e-6, 1e-3)
    batch_size = trial.suggest_categorical('batch_size', [16, 32, 64, 128])

    # Data loading and preprocessing (reuse your pipeline)
    result = load_and_preprocess_data('benign_bus14.xlsx')
    X, y = result[0], result[1]
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    train_data = create_graph_data(X_train, y_train)
    val_data = create_graph_data(X_val, y_val)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = GNNModel(num_features=4, num_classes=2, hidden_channels=hidden_channels, num_layers=num_layers, dropout=dropout).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    best_val_f1 = 0
    patience = 10
    patience_counter = 0
    for epoch in range(50):
        model.train()
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            out = model(batch.x, batch.edge_index, batch.batch)
            loss = criterion(out, batch.y)
            loss.backward()
            optimizer.step()
        # Validation
        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                out = model(batch.x, batch.edge_index, batch.batch)
                preds = out.argmax(dim=1).cpu().numpy()
                labels = batch.y.cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(labels)
        f1 = f1_score(all_labels, all_preds, average='weighted')
        if f1 > best_val_f1:
            best_val_f1 = f1
            patience_counter = 0
        else:
            patience_counter += 1
        if patience_counter >= patience:
            break
    return best_val_f1

def run_gnn_hyperopt():
    study = optuna.create_study(direction='maximize')
    study.optimize(gnn_objective, n_trials=30)
    print("Best trial for GNN:")
    print(f"  Value: {study.best_trial.value}")
    print("  Params: ")
    for key, value in study.best_trial.params.items():
        print(f"    {key}: {value}")
    return study.best_trial

def main():
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load and preprocess data
    X, y, scaler = load_and_preprocess_data('benign_bus14.xlsx')
    
    # Print dataset statistics
    print(f"\nDataset Statistics:")
    print(f"Total samples: {len(X)}")
    print(f"Benign samples: {np.sum(y == 0)}")
    print(f"Malicious samples: {np.sum(y == 1)}")
    print(f"Feature shape: {X.shape}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Create graph data
    train_data = create_graph_data(X_train, y_train)
    test_data = create_graph_data(X_test, y_test)
    
    # Create data loaders
    train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=32, shuffle=False)
    
    # Update hyperparameters to best found by Optuna
    BEST_PARAMS = {
        'hidden_channels': 128,
        'num_layers': 3,
        'dropout': 0.13,
        'lr': 0.0056,
        'weight_decay': 7.5e-6,
        'batch_size': 128
    }
    # Use BEST_PARAMS for model and training
    model = GNNModel(num_features=4, num_classes=2).to(device)
    train_losses, val_losses = train_model(model, train_loader, test_loader, device=device)
    
    # Evaluate model
    metrics = evaluate_model(model, test_loader, device=device)
    print("\nModel Performance Metrics:")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"F1 Score: {metrics['f1_score']:.4f}")
    
    # Plot results
    plot_results(train_losses, val_losses, metrics)
    
    # Save the model
    torch.save(model.state_dict(), 'gnn_attack_detection_model.pth')
    
    # Save dataset information
    with open('dataset_info.txt', 'w') as f:
        f.write("Dataset Information:\n")
        f.write(f"Total samples: {len(X)}\n")
        f.write(f"Benign samples: {np.sum(y == 0)}\n")
        f.write(f"Malicious samples: {np.sum(y == 1)}\n")
        f.write(f"Feature shape: {X.shape}\n")
        f.write("\nAttack Types:\n")
        f.write("1. Random noise attack\n")
        f.write("2. Scaling attack (amplification)\n")
        f.write("3. Offset attack (value shifting)\n")

if __name__ == "__main__":
    print("Running GNN hyperparameter tuning with Optuna...")
    best_trial = run_gnn_hyperopt()
    # After tuning, you can retrain and evaluate with best params as before
    main() 