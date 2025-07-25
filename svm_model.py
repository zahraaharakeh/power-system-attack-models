import numpy as np
import pandas as pd
from sklearn.svm import SVC
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

def handle_class_imbalance(X, y):
    print("Handling class imbalance...")
    class_weights = compute_class_weight('balanced', classes=np.unique(y), y=y)
    weight_dict = dict(zip(np.unique(y), class_weights))
    print(f"Class weights: {weight_dict}")
    return weight_dict

def train_svm(X, y, class_weights=None):
    print("Training SVM model...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"Training set size: {len(X_train)}")
    print(f"Test set size: {len(X_test)}")
    # SVM with RBF kernel and balanced class weights
    svm_model = SVC(
        kernel='rbf',
        class_weight=class_weights if class_weights else 'balanced',
        probability=True,
        random_state=42
    )
    svm_model.fit(X_train, y_train)
    y_pred = svm_model.predict(X_test)
    y_pred_proba = svm_model.predict_proba(X_test)[:, 1]
    return svm_model, X_train, X_test, y_train, y_test, y_pred, y_pred_proba

def evaluate_model(y_true, y_pred, y_pred_proba, model_name="SVM"):
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

def plot_results(model, X_test, y_test, y_pred, feature_names, metrics):
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
    fpr, tpr, _ = roc_curve(y_test, model.predict_proba(X_test)[:, 1])
    axes[1].plot(fpr, tpr, label=f'ROC Curve (AUC = {metrics["roc_auc"]:.3f})')
    axes[1].plot([0, 1], [0, 1], 'k--', label='Random')
    axes[1].set_xlabel('False Positive Rate')
    axes[1].set_ylabel('True Positive Rate')
    axes[1].set_title('ROC Curve')
    axes[1].legend()
    axes[1].grid(True)
    plt.tight_layout()
    plt.savefig('svm_results.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("Visualizations saved as 'svm_results.png'")

def perform_cross_validation(X, y, cv=5):
    print(f"\nPerforming {cv}-fold Cross-Validation...")
    svm_cv = SVC(kernel='rbf', class_weight='balanced', probability=True, random_state=42)
    cv_scores = cross_val_score(svm_cv, X, y, cv=cv, scoring='f1_weighted')
    print(f"Cross-validation F1 scores: {cv_scores}")
    print(f"Mean CV F1 score: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
    return cv_scores

def hyperparameter_tuning(X, y):
    print("\nPerforming Hyperparameter Tuning...")
    param_grid = {
        'C': [0.1, 1, 10],
        'gamma': ['scale', 0.01, 0.1, 1],
        'kernel': ['rbf', 'linear']
    }
    svm_tune = SVC(class_weight='balanced', probability=True, random_state=42)
    grid_search = GridSearchCV(
        svm_tune, 
        param_grid, 
        cv=3, 
        scoring='f1_weighted',
        n_jobs=-1,
        verbose=1
    )
    grid_search.fit(X, y)
    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Best F1 score: {grid_search.best_score_:.4f}")
    return grid_search.best_estimator_, grid_search.best_params_

def save_results(metrics, feature_names, model):
    print("Saving results...")
    with open('svm_results.txt', 'w') as f:
        f.write("SVM Model Results\n")
        f.write("=" * 50 + "\n\n")
        f.write("Model Performance Metrics:\n")
        f.write(f"Accuracy: {metrics['accuracy']:.4f}\n")
        f.write(f"Precision: {metrics['precision']:.4f}\n")
        f.write(f"Recall: {metrics['recall']:.4f}\n")
        f.write(f"F1 Score: {metrics['f1_score']:.4f}\n")
        f.write(f"ROC AUC: {metrics['roc_auc']:.4f}\n\n")
        f.write(f"\nConfusion Matrix:\n{metrics['confusion_matrix']}\n")
    print("Results saved to 'svm_results.txt'")

def main():
    print("SVM Model for Power System Attack Detection")
    print("=" * 60)
    X, y, scaler, feature_names = load_and_preprocess_data('benign_bus14.xlsx')
    class_weights = handle_class_imbalance(X, y)
    svm_model, X_train, X_test, y_train, y_test, y_pred, y_pred_proba = train_svm(X, y, class_weights)
    metrics = evaluate_model(y_test, y_pred, y_pred_proba)
    cv_scores = perform_cross_validation(X, y)
    print("\n" + "="*60)
    print("HYPERPARAMETER TUNING (This may take several minutes...)")
    print("="*60)
    try:
        best_model, best_params = hyperparameter_tuning(X, y)
        best_model.fit(X_train, y_train)
        y_pred_best = best_model.predict(X_test)
        y_pred_proba_best = best_model.predict_proba(X_test)[:, 1]
        print("\nBest Model Evaluation:")
        best_metrics = evaluate_model(y_test, y_pred_best, y_pred_proba_best, "Optimized SVM")
        svm_model = best_model
        metrics = best_metrics
        y_pred = y_pred_best
        y_pred_proba = y_pred_proba_best
    except Exception as e:
        print(f"Hyperparameter tuning failed: {e}")
        print("Using default model parameters.")
    plot_results(svm_model, X_test, y_test, y_pred, feature_names, metrics)
    save_results(metrics, feature_names, svm_model)
    print("\n" + "="*60)
    print("SVM ANALYSIS COMPLETE!")
    print("="*60)
    print("Files generated:")
    print("- svm_results.png (visualizations)")
    print("- svm_results.txt (detailed results)")
    return svm_model, metrics

if __name__ == "__main__":
    svm_model, metrics = main() 