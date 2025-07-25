import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

print("Testing data loading...")
try:
    # Load benign data
    benign_df = pd.read_excel('benign_bus14.xlsx')
    print(f"Data loaded successfully: {benign_df.shape}")
    
    # Extract features
    feature_columns = ['Pd_new', 'Qd_new', 'Vm', 'Va']
    X_benign = benign_df[feature_columns].values
    print(f"Features extracted: {X_benign.shape}")
    
    # Generate malicious data
    print("Generating malicious data...")
    n_samples = len(X_benign)
    
    # Simple malicious data generation
    fdi_attack = X_benign + np.random.normal(0, 0.1, X_benign.shape)
    replay_attack = X_benign + np.random.normal(0, 0.1, X_benign.shape)
    covert_attack = X_benign + np.random.normal(0, 0.1, X_benign.shape)
    realistic_noise = X_benign + np.random.normal(0, 0.1, X_benign.shape)
    
    malicious_data = np.vstack([fdi_attack, replay_attack, covert_attack, realistic_noise])
    print(f"Malicious data generated: {malicious_data.shape}")
    
    # Create labels
    y_benign = np.zeros(len(X_benign))
    y_malicious = np.ones(len(malicious_data))
    
    # Combine data
    X = np.vstack([X_benign, malicious_data])
    y = np.concatenate([y_benign, y_malicious])
    
    print(f"Total data: {X.shape}")
    print(f"Class distribution: {np.bincount(y.astype(int))}")
    
    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    print("Data preprocessing completed successfully!")
    
    # Test data splitting
    X_temp, X_test, y_temp, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.2, random_state=42, stratify=y_temp
    )
    
    print(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
    
    # Test DataLoader creation
    train_dataset = TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train))
    val_dataset = TensorDataset(torch.FloatTensor(X_val), torch.FloatTensor(y_val))
    test_dataset = TensorDataset(torch.FloatTensor(X_test), torch.FloatTensor(y_test))
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    print("DataLoaders created successfully!")
    print("All tests passed!")
    
except Exception as e:
    print(f"Error occurred: {e}")
    import traceback
    traceback.print_exc() 