import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.data import Data, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import optuna
import warnings
warnings.filterwarnings('ignore')

class GNNModel(nn.Module):
    def __init__(self, num_features, num_classes, hidden_channels=64, num_layers=3, dropout=0.2):
        super(GNNModel, self).__init__()
        self.layers = nn.ModuleList()
        self.layers.append(GCNConv(num_features, hidden_channels))
        for _ in range(num_layers - 1):
            self.layers.append(GCNConv(hidden_channels, hidden_channels))
        self.classifier = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels // 2, num_classes)
        )
        self.dropout = dropout
    def forward(self, x, edge_index, batch):
        for conv in self.layers:
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = global_mean_pool(x, batch)
        x = self.classifier(x)
        return x

def create_graph_data(features, labels):
    data_list = []
    for i in range(len(features)):
        x = torch.FloatTensor(features[i]).view(1, -1)
        edge_idx = torch.LongTensor([[0], [0]])
        data = Data(x=x, edge_index=edge_idx, y=torch.LongTensor([labels[i]]))
        data_list.append(data)
    return data_list

def load_and_preprocess_data(benign_file):
    benign_df = pd.read_excel(benign_file)
    feature_columns = ['Pd_new', 'Qd_new', 'Vm', 'Va']
    X_benign = benign_df[feature_columns].values
    X_malicious = generate_malicious_data(X_benign)
    y_benign = np.zeros(len(X_benign))
    y_malicious = np.ones(len(X_malicious))
    X = np.vstack([X_benign, X_malicious])
    y = np.concatenate([y_benign, y_malicious])
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, y

def generate_malicious_data(benign_data):
    noise_attack = benign_data + np.random.normal(0, 0.1, benign_data.shape)
    scaling_attack = benign_data * np.random.uniform(1.5, 2.0, benign_data.shape)
    offset_attack = benign_data + np.random.uniform(0.5, 1.0, benign_data.shape)
    malicious_data = np.vstack([noise_attack, scaling_attack, offset_attack])
    return malicious_data

def objective(trial):
    # Hyperparameters to tune
    hidden_channels = trial.suggest_categorical('hidden_channels', [32, 64, 128, 256])
    num_layers = trial.suggest_int('num_layers', 2, 4)
    dropout = trial.suggest_float('dropout', 0.1, 0.5)
    lr = trial.suggest_loguniform('lr', 1e-4, 1e-2)
    weight_decay = trial.suggest_loguniform('weight_decay', 1e-6, 1e-3)
    batch_size = trial.suggest_categorical('batch_size', [16, 32, 64, 128])

    X, y = load_and_preprocess_data('benign_bus14.xlsx')
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

def main():
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=30)
    print("Best trial:")
    print(f"  Value: {study.best_trial.value}")
    print("  Params: ")
    for key, value in study.best_trial.params.items():
        print(f"    {key}: {value}")

if __name__ == "__main__":
    main() 