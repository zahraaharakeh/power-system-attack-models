import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

print("All imports successful!")
print("Testing basic functionality...")

# Test data loading
try:
    df = pd.read_excel('benign_bus14.xlsx')
    print(f"Data loaded successfully: {df.shape}")
except Exception as e:
    print(f"Error loading data: {e}")

print("Test completed!") 