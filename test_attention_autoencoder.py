import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

def test_attention_autoencoder_results():
    """Test and display the attention autoencoder results"""
    
    print("Testing Attention Autoencoder Results")
    print("=" * 50)
    
    # Check if results file exists
    try:
        with open('attention_autoencoder_results.txt', 'r') as f:
            results_content = f.read()
            print("Results file found!")
            print("\n" + results_content)
    except FileNotFoundError:
        print("Results file not found yet. Model may still be training...")
        return
    
    # Check if model file exists
    try:
        model_state = torch.load('attention_autoencoder_model.pth', map_location='cpu')
        print(f"\nModel file found! Model parameters: {len(model_state)} layers")
    except FileNotFoundError:
        print("Model file not found yet. Model may still be training...")
        return
    
    # Check if visualization exists
    try:
        import matplotlib.image as mpimg
        img = mpimg.imread('attention_autoencoder_results.png')
        print(f"\nVisualization found! Image shape: {img.shape}")
        
        # Display the results
        plt.figure(figsize=(15, 10))
        plt.imshow(img)
        plt.axis('off')
        plt.title('Attention Autoencoder Results', fontsize=16, pad=20)
        plt.tight_layout()
        plt.show()
        
    except FileNotFoundError:
        print("Visualization file not found yet. Model may still be training...")
    
    print("\n" + "="*50)
    print("ATTENTION AUTOENCODER TEST COMPLETE!")
    print("="*50)

if __name__ == "__main__":
    test_attention_autoencoder_results() 