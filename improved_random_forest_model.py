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
    """Load and preprocess the data from benign file and generate realistic malicious data."""
    print("Loading and preprocessing data...")
    
    # Load benign data
    benign_df = pd.read_excel(benign_file)
    
    # Extract features
    feature_columns = ['Pd_new', 'Qd_new', 'Vm', 'Va']
    X_benign = benign_df[feature_columns].values
    
    print(f"Benign samples loaded: {len(X_benign)}")
    print(f"Features: {feature_columns}")
    
    # Generate realistic malicious data
    X_malicious = generate_realistic_malicious_data(X_benign)
    
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

def generate_realistic_malicious_data(benign_data):
    """Generate realistic malicious data using power system attack models."""
    print("Generating realistic malicious data...")
    
    n_samples = len(benign_data)
    malicious_samples = []
    
    # 1. False Data Injection (FDI) Attack - Subtle manipulation
    # FDI attacks try to maintain system observability while injecting false data
    n_fdi = n_samples // 3
    fdi_indices = np.random.choice(n_samples, n_fdi, replace=False)
    
    for idx in fdi_indices:
        original_sample = benign_data[idx].copy()
        
        # Select random features to manipulate (1-2 features)
        n_features_to_attack = np.random.randint(1, 3)
        attack_features = np.random.choice(4, n_features_to_attack, replace=False)
        
        for feat in attack_features:
            # Subtle manipulation that maintains realistic ranges
            if feat in [0, 1]:  # Pd_new, Qd_new (power values)
                # Small percentage change (±5-15%)
                change_factor = np.random.uniform(0.85, 1.15)
                original_sample[feat] *= change_factor
            elif feat == 2:  # Vm (voltage magnitude)
                # Small voltage deviation (±2-8%)
                change_factor = np.random.uniform(0.92, 1.08)
                original_sample[feat] *= change_factor
            else:  # Va (voltage angle)
                # Small angle deviation (±1-3 degrees)
                angle_change = np.random.uniform(-3, 3)
                original_sample[feat] += np.radians(angle_change)
        
        malicious_samples.append(original_sample)
    
    # 2. Replay Attack - Reuse of previous measurements
    n_replay = n_samples // 4
    replay_indices = np.random.choice(n_samples, n_replay, replace=False)
    
    for idx in replay_indices:
        # Replay a previous measurement with slight noise
        replay_idx = np.random.choice(n_samples)
        replayed_sample = benign_data[replay_idx].copy()
        
        # Add small noise to make it slightly different
        noise = np.random.normal(0, 0.02, replayed_sample.shape)
        replayed_sample += noise * np.abs(replayed_sample)  # Proportional noise
        
        malicious_samples.append(replayed_sample)
    
    # 3. Covert Attack - Maintains system observability
    n_covert = n_samples // 4
    covert_indices = np.random.choice(n_samples, n_covert, replace=False)
    
    for idx in covert_indices:
        original_sample = benign_data[idx].copy()
        
        # Covert attack: manipulate multiple measurements to maintain consistency
        # This is more sophisticated and harder to detect
        
        # Manipulate power measurements while maintaining power balance
        power_change = np.random.uniform(-0.1, 0.1)
        original_sample[0] += power_change  # Pd_new
        original_sample[1] -= power_change * 0.8  # Qd_new (related but not exactly opposite)
        
        # Add small random perturbations to voltage measurements
        original_sample[2] += np.random.normal(0, 0.01)  # Vm
        original_sample[3] += np.random.normal(0, 0.005)  # Va
        
        malicious_samples.append(original_sample)
    
    # 4. Random Noise Attack - But with realistic constraints
    n_noise = n_samples - len(malicious_samples)
    noise_indices = np.random.choice(n_samples, n_noise, replace=False)
    
    for idx in noise_indices:
        original_sample = benign_data[idx].copy()
        
        # Add realistic noise based on measurement characteristics
        for i in range(4):
            if i in [0, 1]:  # Power measurements
                noise_level = 0.03  # 3% noise
            elif i == 2:  # Voltage magnitude
                noise_level = 0.01  # 1% noise
            else:  # Voltage angle
                noise_level = 0.005  # 0.5% noise (angles are more precise)
            
            noise = np.random.normal(0, noise_level)
            original_sample[i] += noise * np.abs(original_sample[i])
        
        malicious_samples.append(original_sample)
    
    malicious_data = np.array(malicious_samples)
    print(f"Generated {len(malicious_data)} realistic malicious samples")
    print(f"Attack types: FDI ({n_fdi}), Replay ({n_replay}), Covert ({n_covert}), Noise ({n_noise})")
    
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
    plt.savefig('improved_random_forest_results.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("Visualizations saved as 'improved_random_forest_results.png'")

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

def save_results(metrics, feature_names, model):
    """Save results to text file."""
    print("Saving results...")
    
    with open('improved_random_forest_results.txt', 'w') as f:
        f.write("Improved Random Forest Model Results\n")
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
        
        f.write("\nAttack Types Implemented:\n")
        f.write("1. False Data Injection (FDI) - Subtle manipulation of measurements\n")
        f.write("2. Replay Attack - Reuse of previous measurements with noise\n")
        f.write("3. Covert Attack - Maintains system observability\n")
        f.write("4. Realistic Noise Attack - Proportional noise based on measurement type\n")
    
    print("Results saved to 'improved_random_forest_results.txt'")

def main():
    """Main function to run the improved Random Forest analysis."""
    print("Improved Random Forest Model for Power System Attack Detection")
    print("=" * 70)
    print("Using realistic attack models instead of simple synthetic data")
    print("=" * 70)
    
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
    
    # Plot results
    plot_results(rf_model, X_test, y_test, y_pred, feature_names, metrics)
    
    # Save results
    save_results(metrics, feature_names, rf_model)
    
    print("\n" + "="*70)
    print("IMPROVED RANDOM FOREST ANALYSIS COMPLETE!")
    print("="*70)
    print("Files generated:")
    print("- improved_random_forest_results.png (visualizations)")
    print("- improved_random_forest_results.txt (detailed results)")
    print("\nExpected: Lower accuracy due to realistic attack models")
    
    return rf_model, metrics

if __name__ == "__main__":
    rf_model, metrics = main() 