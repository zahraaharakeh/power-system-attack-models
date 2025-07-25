import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report, roc_auc_score
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

class PowerSystemDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

def load_and_preprocess_data(benign_file):
    print("Loading and preprocessing data...")
    benign_df = pd.read_excel(benign_file)
    feature_columns = ['Pd_new', 'Qd_new', 'Vm', 'Va']
    X_benign = benign_df[feature_columns].values
    print(f"Benign samples loaded: {len(X_benign)}")
    print(f"Features: {feature_columns}")
    X_malicious = generate_malicious_data(X_benign)
    y_benign = np.zeros(len(X_benign))
    y_malicious = np.ones(len(X_malicious))
    X = np.vstack([X_benign, X_malicious])
    y = np.concatenate([y_benign, y_malicious])
    print(f"Total samples: {len(X)}")
    print(f"Benign samples: {len(X_benign)}")
    print(f"Malicious samples: {len(X_malicious)}")
    print(f"Class distribution: {np.bincount(y.astype(int))}")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, y, scaler, feature_columns

def generate_malicious_data(benign_data):
    print("Generating malicious data...")
    n_samples = len(benign_data)
    noise_attack = benign_data + np.random.normal(0, 0.1, benign_data.shape)
    scaling_attack = benign_data * np.random.uniform(1.5, 2.0, benign_data.shape)
    offset_attack = benign_data + np.random.uniform(0.5, 1.0, benign_data.shape)
    malicious_data = np.vstack([noise_attack, scaling_attack, offset_attack])
    print(f"Generated {len(malicious_data)} malicious samples")
    return malicious_data

def handle_class_imbalance(y):
    print("Handling class imbalance...")
    class_weights = compute_class_weight('balanced', classes=np.unique(y), y=y)
    weight_tensor = torch.tensor(class_weights, dtype=torch.float32)
    print(f"Class weights: {dict(zip(np.unique(y), class_weights))}")
    return weight_tensor

class FeedforwardNN(nn.Module):
    def __init__(self, input_size, hidden_sizes=[32, 16]):
        super(FeedforwardNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_sizes[0])
        self.bn1 = nn.BatchNorm1d(hidden_sizes[0])
        self.fc2 = nn.Linear(hidden_sizes[0], hidden_sizes[1])
        self.bn2 = nn.BatchNorm1d(hidden_sizes[1])
        self.fc3 = nn.Linear(hidden_sizes[1], 1)
        self.dropout = nn.Dropout(0.3)
    def forward(self, x):
        x = self.dropout(torch.relu(self.bn1(self.fc1(x))))
        x = self.dropout(torch.relu(self.bn2(self.fc2(x))))
        x = self.fc3(x)
        return x.squeeze(-1)

def train_fnn(X, y, class_weights, epochs=30, batch_size=32, lr=0.001):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    train_dataset = PowerSystemDataset(X_train, y_train)
    test_dataset = PowerSystemDataset(X_test, y_test)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = FeedforwardNN(input_size=X.shape[1]).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=class_weights[1].to(device))
    optimizer = optim.Adam(model.parameters(), lr=lr)
    train_losses = []
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * xb.size(0)
        train_losses.append(epoch_loss / len(train_loader.dataset))
        if (epoch+1) % 5 == 0:
            print(f"Epoch {epoch+1}/{epochs}, Loss: {train_losses[-1]:.4f}")
    # Evaluate
    model.eval()
    all_logits, all_labels = [], []
    with torch.no_grad():
        for xb, yb in test_loader:
            xb = xb.to(device)
            logits = model(xb).cpu()
            all_logits.append(logits)
            all_labels.append(yb)
    all_logits = torch.cat(all_logits)
    all_labels = torch.cat(all_labels)
    y_pred = (torch.sigmoid(all_logits) > 0.5).int().numpy()
    y_true = all_labels.int().numpy()
    y_pred_proba = torch.sigmoid(all_logits).numpy()
    return model, X_train, X_test, y_train, y_test, y_pred, y_pred_proba, y_true

def evaluate_model(y_true, y_pred, y_pred_proba, model_name="FNN"):
    print(f"\n{model_name} Model Evaluation:")
    print("=" * 50)
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted')
    recall = recall_score(y_true, y_pred, average='weighted')
    f1 = f1_score(y_true, y_pred, average='weighted')
    roc_auc = roc_auc_score(y_true, y_pred_proba)
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"ROC AUC: {roc_auc:.4f}")
    cm = confusion_matrix(y_true, y_pred)
    print(f"\nConfusion Matrix:")
    print(cm)
    print(f"\nClassification Report:")
    print(classification_report(y_true, y_pred))
    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'roc_auc': roc_auc,
        'confusion_matrix': cm
    }
    return metrics

def plot_results(y_true, y_pred, y_pred_proba, metrics):
    print("Creating visualizations...")
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    # 1. Confusion Matrix Heatmap
    cm = metrics['confusion_matrix']
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0])
    axes[0].set_title('Confusion Matrix')
    axes[0].set_xlabel('Predicted')
    axes[0].set_ylabel('Actual')
    # 2. ROC Curve
    from sklearn.metrics import roc_curve
    fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
    axes[1].plot(fpr, tpr, label=f'ROC Curve (AUC = {metrics["roc_auc"]:.3f})')
    axes[1].plot([0, 1], [0, 1], 'k--', label='Random')
    axes[1].set_xlabel('False Positive Rate')
    axes[1].set_ylabel('True Positive Rate')
    axes[1].set_title('ROC Curve')
    axes[1].legend()
    axes[1].grid(True)
    plt.tight_layout()
    plt.savefig('fnn_results.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("Visualizations saved as 'fnn_results.png'")

def save_results(metrics):
    print("Saving results...")
    with open('fnn_results.txt', 'w') as f:
        f.write("FNN Model Results\n")
        f.write("=" * 50 + "\n\n")
        f.write("Model Performance Metrics:\n")
        f.write(f"Accuracy: {metrics['accuracy']:.4f}\n")
        f.write(f"Precision: {metrics['precision']:.4f}\n")
        f.write(f"Recall: {metrics['recall']:.4f}\n")
        f.write(f"F1 Score: {metrics['f1_score']:.4f}\n")
        f.write(f"ROC AUC: {metrics['roc_auc']:.4f}\n\n")
        f.write(f"\nConfusion Matrix:\n{metrics['confusion_matrix']}\n")
    print("Results saved to 'fnn_results.txt'")

def main():
    print("FNN Model for Power System Attack Detection")
    print("=" * 60)
    X, y, scaler, feature_names = load_and_preprocess_data('benign_bus14.xlsx')
    class_weights = handle_class_imbalance(y)
    model, X_train, X_test, y_train, y_test, y_pred, y_pred_proba, y_true = train_fnn(X, y, class_weights)
    metrics = evaluate_model(y_true, y_pred, y_pred_proba)
    plot_results(y_true, y_pred, y_pred_proba, metrics)
    save_results(metrics)
    print("\n" + "="*60)
    print("FNN ANALYSIS COMPLETE!")
    print("="*60)
    print("Files generated:")
    print("- fnn_results.png (visualizations)")
    print("- fnn_results.txt (detailed results)")
    return model, metrics

if __name__ == "__main__":
    model, metrics = main() 