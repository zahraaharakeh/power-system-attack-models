import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    confusion_matrix, classification_report, roc_auc_score
)
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.utils.class_weight import compute_class_weight
import warnings
warnings.filterwarnings('ignore')

def load_and_preprocess_data(benign_file):
    """Load and preprocess the data from benign file and generate malicious data."""
    print("Loading and preprocessing data...")
    
    # Load benign data
    benign_df = pd.read_excel(benign_file)
    
    # Extract features
    feature_columns = ['Pd_new', 'Qd_new', 'Vm', 'Va']
    X_benign = benign_df[feature_columns].values
    
    print(f"Benign samples loaded: {len(X_benign)}")
    print(f"Features: {feature_columns}")
    
    # Generate malicious data
    X_malicious = generate_malicious_data(X_benign)
    
    # Create labels (0 for benign, 1 for malicious)
    y_benign = np.zeros(len(X_benign))
    y_malicious = np.ones(len(X_malicious))
    
    # Combine data
    X = np.vstack([X_benign, X_malicious])
    y = np.concatenate([y_benign, y_malicious])
    
    print(f"Total samples: {len(X)}")
    print(f"Benign samples: {len(X_benign)}")
    print(f"Malicious samples: {len(X_malicious)}")
    print(f"Class distribution: {np.bincount(y.astype(int))}")
    
    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    return X_scaled, y, scaler, feature_columns

def generate_malicious_data(benign_data):
    """Generate synthetic malicious data by perturbing benign data."""
    print("Generating malicious data...")
    
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
    
    print(f"Generated {len(malicious_data)} malicious samples")
    return malicious_data

def handle_class_imbalance(X, y):
    """Handle class imbalance using class weights."""
    print("Handling class imbalance...")
    
    # Compute class weights
    class_weights = compute_class_weight(
        'balanced', 
        classes=np.unique(y), 
        y=y
    )
    
    # Create weight dictionary
    weight_dict = dict(zip(np.unique(y), class_weights))
    print(f"Class weights: {weight_dict}")
    
    return weight_dict

def train_random_forest(X, y, class_weights=None):
    """Train Random Forest model with optional hyperparameter tuning."""
    print("Training Random Forest model...")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"Training set size: {len(X_train)}")
    print(f"Test set size: {len(X_test)}")
    
    # Initialize Random Forest with balanced class weights
    rf_model = RandomForestClassifier(
        n_estimators=100,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        max_features='sqrt',
        random_state=42,
        n_jobs=-1,
        class_weight=class_weights if class_weights else 'balanced'
    )
    
    # Train the model
    rf_model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = rf_model.predict(X_test)
    y_pred_proba = rf_model.predict_proba(X_test)[:, 1]
    
    return rf_model, X_train, X_test, y_train, y_test, y_pred, y_pred_proba

def evaluate_model(y_true, y_pred, y_pred_proba, model_name="Random Forest"):
    """Evaluate the model and return comprehensive metrics."""
    print(f"\n{model_name} Model Evaluation:")
    print("=" * 50)
    
    # Calculate metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted')
    recall = recall_score(y_true, y_pred, average='weighted')
    f1 = f1_score(y_true, y_pred, average='weighted')
    roc_auc = roc_auc_score(y_true, y_pred_proba)
    
    # Print metrics
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"ROC AUC: {roc_auc:.4f}")
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    print(f"\nConfusion Matrix:")
    print(cm)
    
    # Classification report
    print(f"\nClassification Report:")
    print(classification_report(y_true, y_pred))
    
    # Store metrics
    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'roc_auc': roc_auc,
        'confusion_matrix': cm
    }
    
    return metrics

def plot_results(model, X_test, y_test, y_pred, feature_names, metrics):
    """Plot various results and visualizations."""
    print("Creating visualizations...")
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. Feature Importance
    feature_importance = model.feature_importances_
    sorted_idx = np.argsort(feature_importance)[::-1]
    
    axes[0, 0].bar(range(len(feature_importance)), feature_importance[sorted_idx])
    axes[0, 0].set_xticks(range(len(feature_importance)))
    axes[0, 0].set_xticklabels([feature_names[i] for i in sorted_idx], rotation=45)
    axes[0, 0].set_title('Feature Importance')
    axes[0, 0].set_ylabel('Importance')
    
    # 2. Confusion Matrix Heatmap
    cm = metrics['confusion_matrix']
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0, 1])
    axes[0, 1].set_title('Confusion Matrix')
    axes[0, 1].set_xlabel('Predicted')
    axes[0, 1].set_ylabel('Actual')
    
    # 3. ROC Curve
    from sklearn.metrics import roc_curve
    fpr, tpr, _ = roc_curve(y_test, model.predict_proba(X_test)[:, 1])
    axes[1, 0].plot(fpr, tpr, label=f'ROC Curve (AUC = {metrics["roc_auc"]:.3f})')
    axes[1, 0].plot([0, 1], [0, 1], 'k--', label='Random')
    axes[1, 0].set_xlabel('False Positive Rate')
    axes[1, 0].set_ylabel('True Positive Rate')
    axes[1, 0].set_title('ROC Curve')
    axes[1, 0].legend()
    axes[1, 0].grid(True)
    
    # 4. Metrics Bar Plot
    metric_names = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
    metric_values = [metrics['accuracy'], metrics['precision'], 
                    metrics['recall'], metrics['f1_score']]
    
    bars = axes[1, 1].bar(metric_names, metric_values, color=['blue', 'green', 'orange', 'red'])
    axes[1, 1].set_title('Model Performance Metrics')
    axes[1, 1].set_ylabel('Score')
    axes[1, 1].set_ylim(0, 1)
    
    # Add value labels on bars
    for bar, value in zip(bars, metric_values):
        axes[1, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                       f'{value:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('random_forest_results.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("Visualizations saved as 'random_forest_results.png'")

def perform_cross_validation(X, y, cv=5):
    """Perform cross-validation to assess model stability."""
    print(f"\nPerforming {cv}-fold Cross-Validation...")
    
    rf_cv = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        class_weight='balanced',
        n_jobs=-1
    )
    
    # Cross-validation scores
    cv_scores = cross_val_score(rf_cv, X, y, cv=cv, scoring='f1_weighted')
    
    print(f"Cross-validation F1 scores: {cv_scores}")
    print(f"Mean CV F1 score: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
    
    return cv_scores

def hyperparameter_tuning(X, y):
    """Perform hyperparameter tuning using GridSearchCV."""
    print("\nPerforming Hyperparameter Tuning...")
    
    # Define parameter grid
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['sqrt', 'log2']
    }
    
    # Initialize model
    rf_tune = RandomForestClassifier(
        random_state=42,
        class_weight='balanced',
        n_jobs=-1
    )
    
    # Grid search with F1 score
    grid_search = GridSearchCV(
        rf_tune, 
        param_grid, 
        cv=3, 
        scoring='f1_weighted',
        n_jobs=-1,
        verbose=1
    )
    
    # Fit grid search
    grid_search.fit(X, y)
    
    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Best F1 score: {grid_search.best_score_:.4f}")
    
    return grid_search.best_estimator_, grid_search.best_params_

def save_results(metrics, feature_names, model):
    """Save results to text file."""
    print("Saving results...")
    
    with open('random_forest_results.txt', 'w') as f:
        f.write("Random Forest Model Results\n")
        f.write("=" * 50 + "\n\n")
        
        f.write("Model Performance Metrics:\n")
        f.write(f"Accuracy: {metrics['accuracy']:.4f}\n")
        f.write(f"Precision: {metrics['precision']:.4f}\n")
        f.write(f"Recall: {metrics['recall']:.4f}\n")
        f.write(f"F1 Score: {metrics['f1_score']:.4f}\n")
        f.write(f"ROC AUC: {metrics['roc_auc']:.4f}\n\n")
        
        f.write("Feature Importance:\n")
        feature_importance = model.feature_importances_
        for i, (feature, importance) in enumerate(zip(feature_names, feature_importance)):
            f.write(f"{feature}: {importance:.4f}\n")
        
        f.write(f"\nConfusion Matrix:\n{metrics['confusion_matrix']}\n")
    
    print("Results saved to 'random_forest_results.txt'")

def main():
    """Main function to run the complete Random Forest analysis."""
    print("Random Forest Model for Power System Attack Detection")
    print("=" * 60)
    
    # Load and preprocess data
    X, y, scaler, feature_names = load_and_preprocess_data('benign_bus14.xlsx')
    
    # Handle class imbalance
    class_weights = handle_class_imbalance(X, y)
    
    # Train Random Forest model
    rf_model, X_train, X_test, y_train, y_test, y_pred, y_pred_proba = train_random_forest(X, y, class_weights)
    
    # Evaluate model
    metrics = evaluate_model(y_test, y_pred, y_pred_proba)
    
    # Perform cross-validation
    cv_scores = perform_cross_validation(X, y)
    
    # Hyperparameter tuning (optional - can be commented out for faster execution)
    print("\n" + "="*60)
    print("HYPERPARAMETER TUNING (This may take several minutes...)")
    print("="*60)
    
    try:
        best_model, best_params = hyperparameter_tuning(X, y)
        
        # Train and evaluate best model
        best_model.fit(X_train, y_train)
        y_pred_best = best_model.predict(X_test)
        y_pred_proba_best = best_model.predict_proba(X_test)[:, 1]
        
        print("\nBest Model Evaluation:")
        best_metrics = evaluate_model(y_test, y_pred_best, y_pred_proba_best, "Optimized Random Forest")
        
        # Use best model for final results
        rf_model = best_model
        metrics = best_metrics
        y_pred = y_pred_best
        y_pred_proba = y_pred_proba_best
        
    except Exception as e:
        print(f"Hyperparameter tuning failed: {e}")
        print("Using default model parameters.")
    
    # Plot results
    plot_results(rf_model, X_test, y_test, y_pred, feature_names, metrics)
    
    # Save results
    save_results(metrics, feature_names, rf_model)
    
    print("\n" + "="*60)
    print("RANDOM FOREST ANALYSIS COMPLETE!")
    print("="*60)
    print("Files generated:")
    print("- random_forest_results.png (visualizations)")
    print("- random_forest_results.txt (detailed results)")
    
    return rf_model, metrics

if __name__ == "__main__":
    rf_model, metrics = main() 