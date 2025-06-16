import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import Dataset, DataLoader
import random

class PowerSystemDataset(Dataset):
    def __init__(self, features, labels, transform=None):
        self.features = features
        self.labels = labels
        self.transform = transform
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        feature = self.features[idx]
        label = self.labels[idx]
        
        if self.transform:
            feature = self.transform(feature)
        
        return feature, label

class DataAugmentation:
    def __init__(self, noise_level=0.1, scale_range=(0.9, 1.1)):
        self.noise_level = noise_level
        self.scale_range = scale_range
    
    def __call__(self, x):
        # Random noise
        if random.random() < 0.5:
            x = x + torch.randn_like(x) * self.noise_level
        
        # Random scaling
        if random.random() < 0.5:
            scale = random.uniform(*self.scale_range)
            x = x * scale
        
        return x

class SupervisedCNN(nn.Module):
    def __init__(self, num_classes=2):
        super(SupervisedCNN, self).__init__()
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(1, 32, kernel_size=2, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=2, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=2, padding=1)
        
        # Calculate the size of flattened features
        # Input shape is (batch_size, 1, 2, 2)
        # After conv1: (batch_size, 32, 3, 3)
        # After maxpool1: (batch_size, 32, 1, 1)
        # After conv2: (batch_size, 64, 2, 2)
        # After maxpool2: (batch_size, 64, 1, 1)
        # After conv3: (batch_size, 128, 2, 2)
        # After maxpool3: (batch_size, 128, 1, 1)
        self.flatten_size = 128 * 1 * 1
        
        # Fully connected layers
        self.fc1 = nn.Linear(self.flatten_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, num_classes)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(0.5)
        
        # Batch normalization
        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
    
    def forward(self, x):
        # Ensure input shape is correct
        if len(x.shape) == 3:
            x = x.unsqueeze(0)  # Add batch dimension if missing
        
        # Print shapes for debugging
        # print(f"Input shape: {x.shape}")
        
        # Convolutional layers with ReLU and max pooling
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout(x)
        # print(f"After conv1: {x.shape}")
        
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout(x)
        # print(f"After conv2: {x.shape}")
        
        x = self.conv3(x)
        x = self.bn3(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout(x)
        # print(f"After conv3: {x.shape}")
        
        # Flatten the output
        x = x.view(x.size(0), -1)  # Use batch size from input
        # print(f"After flatten: {x.shape}")
        
        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        # print(f"Final output: {x.shape}")
        
        return x

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
    
    # Reshape for CNN (samples, channels, height, width)
    X_reshaped = X_scaled.reshape(-1, 1, 2, 2)
    
    return X_reshaped, y, scaler

def generate_malicious_data(benign_data):
    """Generate synthetic malicious data by perturbing benign data."""
    # Create three types of attacks
    n_samples = len(benign_data)
    
    # Random noise attack
    noise_attack = benign_data + np.random.normal(0, 0.1, benign_data.shape)
    
    # Scaling attack
    scaling_attack = benign_data * np.random.uniform(1.5, 2.0, benign_data.shape)
    
    # Offset attack
    offset_attack = benign_data + np.random.uniform(0.5, 1.0, benign_data.shape)
    
    # Combine all attacks
    malicious_data = np.vstack([noise_attack, scaling_attack, offset_attack])
    
    return malicious_data

def train_model(model, train_loader, val_loader, num_epochs=10000, device='cuda'):
    """Train the CNN model with curriculum learning."""
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)
    
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    patience = 20
    patience_counter = 0
    
    # Curriculum learning parameters
    curriculum_epochs = [0, 1000, 2000]  # Epochs to increase difficulty
    noise_levels = [0.05, 0.1, 0.15]    # Increasing noise levels
    
    for epoch in range(num_epochs):
        # Update curriculum learning parameters
        current_noise = noise_levels[min(len(noise_levels)-1, 
                                       sum(1 for e in curriculum_epochs if epoch >= e))]
        
        # Training
        model.train()
        train_loss = 0
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            
            # Print shapes for debugging
            if epoch == 0:
                print(f"Batch shapes - Input: {batch_x.shape}, Target: {batch_y.shape}")
            
            # Apply curriculum learning
            if random.random() < 0.3:  # 30% chance to apply noise
                batch_x = batch_x + torch.randn_like(batch_x) * current_noise
            
            optimizer.zero_grad()
            out = model(batch_x)
            loss = criterion(out, batch_y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                out = model(batch_x)
                loss = criterion(out, batch_y)
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
            torch.save(model.state_dict(), 'best_cnn_model.pth')
        else:
            patience_counter += 1
        
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
        
        if patience_counter >= patience:
            print(f'Early stopping at epoch {epoch+1}')
            break
    
    # Load best model
    model.load_state_dict(torch.load('best_cnn_model.pth'))
    return train_losses, val_losses

def evaluate_model(model, test_loader, device='cuda'):
    """Evaluate the model and return metrics."""
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            out = model(batch_x)
            pred = out.argmax(dim=1)
            all_preds.extend(pred.cpu().numpy())
            all_labels.extend(batch_y.cpu().numpy())
    
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
    # Create figure with subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot training history
    ax1.plot(train_losses, label='Training Loss')
    ax1.plot(val_losses, label='Validation Loss')
    ax1.set_title('Training History')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    
    # Plot confusion matrix
    sns.heatmap(metrics['confusion_matrix'], annot=True, fmt='d', cmap='Blues', ax=ax2)
    ax2.set_title('Confusion Matrix')
    ax2.set_xlabel('Predicted')
    ax2.set_ylabel('True')
    
    plt.tight_layout()
    plt.savefig('cnn_results.png')
    plt.close()

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
    
    # Convert to PyTorch tensors
    X_train = torch.FloatTensor(X_train)
    y_train = torch.LongTensor(y_train)
    X_test = torch.FloatTensor(X_test)
    y_test = torch.LongTensor(y_test)
    
    # Create datasets with data augmentation
    train_dataset = PowerSystemDataset(X_train, y_train, transform=DataAugmentation())
    test_dataset = PowerSystemDataset(X_test, y_test)
    
    # Create data loaders with consistent batch sizes
    batch_size = 32
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=True, num_workers=0)
    
    # Create and train model
    model = SupervisedCNN().to(device)
    
    # Print model summary
    print("\nModel Architecture:")
    print(model)
    
    # Print batch sizes and test forward pass
    for batch_x, batch_y in train_loader:
        print(f"\nTraining batch shapes:")
        print(f"Input shape: {batch_x.shape}")
        print(f"Target shape: {batch_y.shape}")
        
        # Test forward pass
        with torch.no_grad():
            test_output = model(batch_x)
            print(f"Model output shape: {test_output.shape}")
        break
    
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
    torch.save(model.state_dict(), 'supervised_cnn_model.pth')

if __name__ == "__main__":
    main() 