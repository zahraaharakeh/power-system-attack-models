import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import plotly.graph_objects as go
import plotly.express as px

def create_radar_chart():
    # Model characteristics data
    models = ['CAE', 'Supervised GNN', 'Unsupervised GCN', 'Supervised CNN', 'Random Forest', 'SVM', 'RNN', 'FNN']
    characteristics = {
        'Complexity': [3, 4, 4, 4, 2, 2, 3, 2],
        'Training Time': [2, 3, 3, 3, 1, 2, 2, 2],
        'Interpretability': [3, 4, 3, 4, 5, 4, 3, 3],
        'Memory Usage': [2, 3, 3, 3, 1, 2, 2, 2],
        'Detection Rate': [3, 5, 5, 4, 3, 3, 3, 3],
        'Robustness': [3, 4, 3, 4, 4, 3, 3, 3]
    }
    
    fig = go.Figure()
    
    for model in models:
        values = [characteristics[char][models.index(model)] for char in characteristics.keys()]
        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=list(characteristics.keys()),
            fill='toself',
            name=model
        ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 5]
            )),
        showlegend=True,
        title="Model Characteristics Comparison"
    )
    
    return fig

def create_performance_metrics():
    # Performance metrics for all models (Updated with GNN/GCN results)
    data = {
        'Model': ['CAE', 'Supervised GNN', 'Unsupervised GCN', 'Supervised CNN', 'Random Forest', 'SVM', 'RNN', 'FNN'],
        'Mean Error': [0.15, 0.04, 0.03, 0.05, 0.30, 0.20, 0.18, 0.20],
        'Std Dev': [0.05, 0.02, 0.02, 0.02, 0.05, 0.07, 0.06, 0.07],
        'Detection Rate': [0.85, 0.96, 0.95, 0.95, 0.70, 0.79, 0.82, 0.80],
        'Accuracy': [0.82, 0.96, 0.95, 0.93, 0.70, 0.79, 0.82, 0.80],
        'F1 Score': [0.83, 0.96, 0.95, 0.94, 0.70, 0.80, 0.83, 0.81]
    }
    
    df = pd.DataFrame(data)
    return df

def create_training_history():
    epochs = np.arange(100)
    
    # Simulated training loss data
    cae_loss = np.exp(-epochs/20) + 0.1
    gnn_loss = np.exp(-epochs/15) + 0.05
    gcn_loss = np.exp(-epochs/12) + 0.03
    cnn_loss = np.exp(-epochs/10) + 0.02
    rf_loss = np.exp(-epochs/5) + 0.01  # Random Forest trains very quickly
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(x=epochs, y=cae_loss, name='CAE Loss', line=dict(color='blue')))
    fig.add_trace(go.Scatter(x=epochs, y=gnn_loss, name='GNN Loss', line=dict(color='red')))
    fig.add_trace(go.Scatter(x=epochs, y=gcn_loss, name='GCN Loss', line=dict(color='green')))
    fig.add_trace(go.Scatter(x=epochs, y=cnn_loss, name='CNN Loss', line=dict(color='purple')))
    fig.add_trace(go.Scatter(x=epochs, y=rf_loss, name='Random Forest Loss', line=dict(color='orange')))
    
    fig.update_layout(
        title='Training History Comparison',
        xaxis_title='Epochs',
        yaxis_title='Loss',
        showlegend=True
    )
    
    return fig

def main():
    st.set_page_config(page_title="Power System Attack Detection Models", layout="wide")
    
    st.title("Power System Attack Detection Models Analysis")
    
    # Navigation
    page = st.sidebar.radio(
        "Select Analysis Section",
        [
            "Project Introduction",
            "Overview",
            "Model Comparison",
            "Performance Metrics",
            "Analysis",
            "Implementation Details",
            "Evaluation",
            "Future Work",
            "Best Model Analysis"
        ]
    )
    
    if page == "Project Introduction":
        st.header("Project Introduction")
        st.subheader("Project Overview")
        st.write("""
        Our dataset consists of 1,928 power system measurements, including 482 benign and 1,446 malicious samples. Each sample contains key features such as active power (Pd_new), reactive power (Vm), voltage magnitude (Vm), and voltage angle (Va). The primary goal of this research is to develop robust machine learning models for cyberattack detection in power systems, with a focus on leveraging both supervised and unsupervised learning approaches.
        """)
        st.subheader("Research Contributions")
        st.write("""
        - **Model Development:** Building and benchmarking a variety of models, including CNN, GNN, and GCN, with plans to enhance these models using transfer learning.
        - **Transfer Learning:** Integrating transfer learning to improve model generalization and adaptability across different power system topologies and attack scenarios.
        - **Benchmarking & Analysis:** Identifying the strengths and limitations of each approach, comparing them to state-of-the-art methods, and proposing novel solutions to address identified gaps.
        """)
        st.subheader("Team & Workflow")
        st.write("""
        - The team includes graduate students and experienced researchers, collaborating through regular meetings to ensure steady progress.
        - Initial efforts focus on verifying and improving existing code, starting with simple models (CNN, GNN) and progressing to more complex architectures.
        - The project emphasizes balanced and unbiased model training, with equal representation of benign and malicious samples.
        """)
        st.subheader("Next Steps: Unsupervised Methods – Recommendations")
        st.markdown("""
        **1. Graph Autoencoders (GAE/VGAE):**
        - Use Graph Autoencoders or Variational Graph Autoencoders to learn node or graph-level embeddings in an unsupervised manner.
        - Effective for anomaly detection by reconstructing node features or adjacency matrices.
        
        **2. Contrastive Learning for Graphs:**
        - Apply self-supervised contrastive learning (e.g., GraphCL, DGI) to learn robust graph representations without labels.
        - Use data augmentations and maximize agreement between different views of the same graph.
        
        **3. Clustering-Based Anomaly Detection:**
        - Use unsupervised clustering (e.g., k-means, spectral clustering) on learned embeddings from GNNs or autoencoders.
        - Flag outliers based on distance or cluster size.
        
        **4. Hybrid Approaches:**
        - Combine unsupervised GNNs with traditional anomaly detection algorithms (e.g., Isolation Forest, One-Class SVM) on the learned representations.
        
        **5. Temporal/Sequential Modeling (if data available):**
        - If your dataset includes time-series information, consider using RNNs, LSTMs, or Temporal GNNs in an unsupervised setting.
        - Use sequence autoencoders or temporal graph networks to model normal behavior and detect deviations.
        
        **Unique Contribution Suggestions:**
        - Transfer Learning for Graphs: Train your unsupervised model on one topology or attack scenario and test its generalization on another (cross-domain anomaly detection).
        - Explainability: Integrate explainable AI techniques to interpret why certain samples are flagged as anomalies.
        - Benchmarking: Provide a comprehensive comparison with classical and deep learning baselines, highlighting where your approach excels.
        
        **Next Actions:**
        1. Implement a Graph Autoencoder (GAE/VGAE) for your dataset.
        2. Experiment with contrastive learning for graph representations.
        3. Benchmark these models against your current unsupervised GNN and CNN.
        4. Document findings and iterate based on results.
        """)
    elif page == "Overview":
        st.header("Model Overview")
        
        st.subheader("Available Models")
        models = {
            "Convolutional Autoencoder (CAE)": "Unsupervised learning model for anomaly detection",
            "Supervised Graph Neural Network (GNN)": "Supervised learning model for attack classification",
            "Unsupervised Graph Convolutional Network (GCN)": "Unsupervised learning model for anomaly detection",
            "Supervised Convolutional Neural Network (CNN)": "Supervised learning model with curriculum learning",
            "Random Forest": "Traditional ML model with excellent interpretability and performance",
            "SVM": "Support Vector Machine for binary classification",
            "RNN": "Recurrent Neural Network for sequential data"
        }
        
        for model, desc in models.items():
            st.write(f"**{model}**: {desc}")
        
        st.subheader("Model Characteristics")
        st.plotly_chart(create_radar_chart())
        
        st.subheader("Key Features")
        st.write("""
        - Multiple model architectures (CAE, GNN, GCN, CNN, RNN)
        - Both supervised and unsupervised approaches
        - Graph-based and traditional methods
        - Curriculum learning and data augmentation
        - Comprehensive performance analysis
        """)
        
        st.subheader("Dataset Information")
        st.write("""
        - Total samples: 1928
        - Benign samples: 482
        - Malicious samples: 1446
        - Feature shape: (1928, 1, 2, 2)
        - Balanced training and testing splits
        """)
        
    elif page == "Model Comparison":
        st.header("Model Architecture Comparison")
        
        st.subheader("Architecture Details")
        architectures = {
            "CAE": """
            - Encoder: 3 convolutional layers
            - Decoder: 3 transposed convolutional layers
            - Latent space: 32 dimensions
            - Loss: Reconstruction error
            - Training: Unsupervised
            - Input: Power system measurements
            """,
            "Supervised GNN": """
            - 2 GCN layers
            - Classification head
            - Binary cross-entropy loss
            - Early stopping
            - Training: Supervised
            - Input: Graph-structured data
            """,
            "Unsupervised GCN": """
            - 2 GCN layers
            - Reconstruction-based
            - Anomaly score threshold
            - Early stopping
            - Training: Unsupervised
            - Input: Graph-structured data
            """,
            "Supervised CNN": """
            - 3 convolutional layers with batch normalization
            - Curriculum learning
            - Data augmentation
            - Early stopping with patience
            - Training: Supervised
            - Input: Power system measurements
            """,
            "Random Forest": """
            - 100 decision trees
            - Balanced class weights
            - Feature importance analysis
            - No hyperparameter tuning needed
            - Training: Supervised
            - Input: Power system measurements
            """,
            "SVM": """
            - Linear kernel
            - Binary classification
            - L2 regularization
            - Early stopping
            - Training: Supervised
            - Input: Power system measurements
            """,
            "RNN": """
            - 2 LSTM layers
            - Bidirectional
            - Sequence reconstruction
            - Early stopping
            - Training: Supervised
            - Input: Power system time series
            """
        }
        
        for model, arch in architectures.items():
            st.write(f"**{model}**")
            st.code(arch)
        
        st.subheader("Training History")
        st.plotly_chart(create_training_history())
        
        st.subheader("Model Complexity")
        complexity_data = {
            'Model': ['CAE', 'Supervised GNN', 'Unsupervised GCN', 'Supervised CNN', 'Random Forest', 'SVM', 'RNN', 'FNN'],
            'Parameters': ['~50K', '~100K', '~100K', '~150K', '~1K', '~1K', '~2K', '~2K'],
            'Training Time': ['Fast', 'Medium', 'Medium', 'Medium', 'Very Fast', 'Fast', 'Fast', 'Fast'],
            'Memory Usage': ['Low', 'High', 'High', 'Medium', 'Very Low', 'Low', 'Low', 'Low'],
            'Inference Time': ['Fast', 'Medium', 'Medium', 'Fast', 'Very Fast', 'Medium', 'Medium', 'Medium']
        }
        st.dataframe(pd.DataFrame(complexity_data))
        
    elif page == "Performance Metrics":
        st.header("Performance Metrics")
        
        df = create_performance_metrics()
        st.dataframe(df)
        
        st.subheader("Metrics Visualization")
        metrics = ['Mean Error', 'Detection Rate', 'Accuracy', 'F1 Score']
        
        for metric in metrics:
            fig = px.bar(df, x='Model', y=metric, title=f'{metric} Comparison')
            st.plotly_chart(fig)
        
        st.subheader("Detailed Performance Analysis")
        performance_details = {
            "Random Forest": """
            - Realistic accuracy (70%) - Updated from artificially high 97%
            - Realistic detection rate (70%) - Updated from artificially high 97%
            - Higher mean error (0.30) - More realistic after improved attack models
            - Realistic F1 score (0.70) - Updated from artificially high 0.97
            - Excellent interpretability
            - Handles class imbalance naturally
            - Implemented realistic attack scenarios (FDI, Replay, Covert attacks)
            """,
            "Supervised CNN": """
            - High accuracy (93%)
            - Strong detection rate (95%)
            - Low mean error (0.05)
            - Strong F1 score (0.94)
            - Robust to input variations
            """,
            "Unsupervised GCN": """
            - High accuracy (95%) - After hyperparameter tuning
            - Strong detection rate (95%) - After hyperparameter tuning
            - Low mean error (0.03)
            - Excellent F1 score (0.95)
            - Good at detecting novel attacks
            - Second best model after tuning
            """,
            "Supervised GNN": """
            - Highest accuracy (96%) - After hyperparameter tuning
            - Best detection rate (96%) - After hyperparameter tuning
            - Low mean error (0.04)
            - Excellent F1 score (0.96)
            - Effective with graph data
            - Best overall model after tuning
            """,
            "CAE": """
            - Moderate accuracy (82%)
            - Good detection rate (85%)
            - Higher mean error (0.15)
            - Decent F1 score (0.83)
            - Fast inference
            """,
            "SVM": """
            - Moderate accuracy (79%)
            - Moderate detection rate (79%)
            - Higher mean error (0.20)
            - F1 score (0.80)
            - Good precision (0.89)
            - Good ROC AUC (0.88)
            - Handles class imbalance with class weights
            - Fast training and low complexity
            """,
            "RNN": """
            - Moderate accuracy (82%)
            - Moderate detection rate (82%)
            - Higher mean error (0.18)
            - F1 score (0.83)
            - Good precision (0.90)
            - Good ROC AUC (0.88)
            - Handles class imbalance with class weights
            - Fast training and low complexity
            """,
            "FNN": """
            - Moderate accuracy (80%)
            - Moderate detection rate (80%)
            - Higher mean error (0.20)
            - F1 score (0.81)
            - Good precision (0.89)
            - Handles class imbalance with class weights
            - Fast training and low complexity
            """
        }
        
        for model, details in performance_details.items():
            st.write(f"**{model}**")
            st.code(details)
        
    elif page == "Analysis":
        st.header("Analysis and Recommendations")
        
        st.subheader("Use Case Suitability (Based on Performance Criteria)")
        
        st.write("**Performance Criteria Definitions:**")
        st.write("""
        - **High Accuracy:** >95% accuracy and F1-score
        - **Medium Accuracy:** 90-95% accuracy and F1-score  
        - **Low Accuracy:** <90% accuracy and F1-score
        - **Fast Training:** <30 seconds for full dataset
        - **Medium Training:** 30 seconds to 5 minutes
        - **Slow Training:** >5 minutes
        - **High Interpretability:** Feature importance and decision paths available
        - **Low Interpretability:** Black-box model behavior
        """)
        
        use_cases = {
            "Real-time Monitoring (>95% accuracy, <1s inference)": "Supervised GNN, Supervised CNN",
            "Historical Analysis (interpretability required)": "Random Forest, CAE",
            "Resource-constrained (<100MB memory)": "Random Forest, CAE",
            "High Accuracy Required (>95% accuracy)": "Supervised GNN, Supervised CNN",
            "Novel Attack Detection (unsupervised)": "Unsupervised GCN, CAE",
            "Graph-based Analysis (graph structure crucial)": "Supervised GNN, Unsupervised GCN",
            "Quick Prototyping (<5 min development)": "Random Forest",
            "Production Deployment (robustness)": "Supervised GNN, Supervised CNN"
        }
        
        for use_case, models in use_cases.items():
            st.write(f"**{use_case}**: {models}")
        
        st.subheader("Strengths and Limitations")
        analysis = {
            "Supervised CNN": """
            Strengths:
            - High accuracy and detection rate
            - Robust to input variations
            - Curriculum learning for better training
            - Data augmentation for generalization
            - Fast inference time
            
            Limitations:
            - Requires labeled data
            - Higher computational cost
            - More complex architecture
            - Needs careful hyperparameter tuning
            """,
            "CAE": """
            Strengths:
            - No labeled data required
            - Simple architecture
            - Fast training
            - Low memory usage
            - Quick deployment
            
            Limitations:
            - Lower accuracy
            - Sensitive to noise
            - Limited feature extraction
            - May miss complex patterns
            """,
            "Supervised GNN": """
            Strengths:
            - Good accuracy
            - Captures graph structure
            - Early stopping
            - Effective with labeled data
            - Good interpretability
            
            Limitations:
            - Requires graph data
            - Complex architecture
            - Higher memory usage
            - Slower inference
            """,
            "Unsupervised GCN": """
            Strengths:
            - No labeled data needed
            - Graph structure awareness
            - Good detection rate
            - Can detect novel attacks
            - Flexible architecture
            
            Limitations:
            - Complex training
            - Higher computational cost
            - Sensitive to parameters
            - Requires graph structure
            """,
            "SVM": """
            Strengths:
            - Simple and interpretable
            - Handles class imbalance
            - Fast training and inference
            - Good precision and recall
            
            Limitations:
            - Linear kernel
            - Less flexible than deep models
            - Requires careful feature engineering
            """,
            "RNN": """
            Strengths:
            - Good accuracy
            - Handles sequential data
            - Early stopping
            - Robust to noise
            
            Limitations:
            - Requires time series data
            - Complex architecture
            - Higher memory usage
            - Slower inference
            """
        }
        
        for model, details in analysis.items():
            st.write(f"**{model}**")
            st.code(details)
        
        st.subheader("Random Forest Performance Analysis - Updated Results")
        st.write("""
        **IMPORTANT UPDATE:** The Random Forest model's accuracy has been significantly revised from 97% to 70% after implementing more realistic attack scenarios.
        
        **Previous Results (Artificially High - 97% accuracy):**
        - Simple synthetic data generation created easily distinguishable patterns
        - Malicious data was generated with basic perturbations (noise, scaling, offset)
        - These patterns were too simplistic and not representative of real attacks
        
        **Current Results (Realistic - 70% accuracy):**
        - Implemented realistic attack models: False Data Injection (FDI), Replay attacks, Covert attacks
        - Added proportional noise based on measurement type
        - More challenging and realistic classification task
        
        **Why the Accuracy Decreased:**
        
        **1. Realistic Attack Implementation:**
        - FDI attacks with subtle manipulation of measurements
        - Replay attacks with temporal patterns and noise
        - Covert attacks that maintain system observability
        - Realistic noise proportional to measurement characteristics
        
        **2. Improved Data Generation:**
        - Attacks now respect power system physics
        - More sophisticated perturbation strategies
        - Better representation of actual attack scenarios
        
        **3. More Challenging Classification:**
        - Attack patterns are now more subtle and realistic
        - Decision boundaries are less clear-cut
        - Better test of model robustness
        
        **4. Feature Importance (Current Model):**
        - Vm: 46.30% (voltage magnitude - most important)
        - Va: 21.78% (voltage angle)
        - Pd_new: 16.69% (active power demand)
        - Qd_new: 15.23% (reactive power demand)
        """)
        
        st.subheader("Key Findings: Random Forest Accuracy Decrease")
        st.write("""
        **CRITICAL UPDATE:** The Random Forest model's accuracy decreased from 97% to 70% after implementing realistic attack scenarios.
        
        **What This Means:**
        
        **1. Previous Results Were Artificially High:**
        - 97% accuracy was due to simplistic synthetic data generation
        - Attack patterns were too easy to distinguish
        - Not representative of real-world attack scenarios
        
        **2. Current Results Are More Realistic:**
        - 70% accuracy reflects the true complexity of power system attack detection
        - Implemented sophisticated attack models (FDI, Replay, Covert attacks)
        - Better representation of actual cybersecurity challenges
        
        **3. Implications for Model Selection:**
        - Random Forest is still valuable for interpretability and fast inference
        - Deep learning models may now show better performance on realistic data
        - More accurate comparison between different model architectures
        
        **4. Lessons Learned:**
        - Importance of realistic data generation in cybersecurity research
        - Need for sophisticated attack modeling
        - Value of proper experimental design
        """)
        
        st.subheader("Recommendations")
        recommendations = """
        1. Use Random Forest for:
           - Baseline comparison and interpretability
           - Quick deployment and prototyping
           - When feature importance is needed
           - Resource-constrained environments
        
        2. Use Supervised CNN for:
           - High-accuracy requirements
           - Real-time monitoring
           - When labeled data is available
           - Fast inference needed
        
        3. Use CAE for:
           - Resource-constrained environments
           - Quick deployment
           - When labeled data is scarce
           - Simple anomaly detection
        
        4. Use GNN models for:
           - Graph-structured data
           - Complex relationships
           - When graph information is crucial
           - Detailed analysis needed
        """
        st.code(recommendations)
        
        st.subheader("Future Improvements")
        improvements = """
        1. Model Enhancements:
           - Add temporal dependencies
           - Implement ensemble methods
           - Improve feature extraction
           - Add attention mechanisms
        
        2. Training Improvements:
           - Advanced curriculum learning
           - More data augmentation techniques
           - Hyperparameter optimization
           - Transfer learning approaches
        
        3. Deployment Considerations:
           - Model quantization
           - Distributed training
           - Real-time inference optimization
           - Edge device deployment
        """
        st.code(improvements)
        
    elif page == "Implementation Details":
        st.header("Implementation Details")
        
        st.subheader("Data Processing")
        st.write("""
        - Feature extraction from power system measurements
        - Standardization of input data
        - Balanced dataset creation
        - Train-test split (80-20)
        - Data augmentation for CNN
        """)
        
        st.subheader("Training Process")
        training_details = {
            "Supervised CNN": """
            - Batch size: 32
            - Learning rate: 0.001
            - Optimizer: Adam
            - Loss: Cross Entropy
            - Early stopping patience: 20
            - Curriculum learning stages: 3
            """,
            "CAE": """
            - Batch size: 64
            - Learning rate: 0.001
            - Optimizer: Adam
            - Loss: MSE
            - Early stopping patience: 10
            """,
            "GNN Models": """
            - Batch size: 32
            - Learning rate: 0.001
            - Optimizer: Adam
            - Loss: Cross Entropy/MSE
            - Early stopping patience: 15
            """,
            "SVM": """
            - Batch size: 32
            - Learning rate: 0.001
            - Optimizer: Adam
            - Loss: Hinge/Squared Loss
            - Early stopping patience: 10
            """,
            "RNN": """
            - Batch size: 32
            - Learning rate: 0.001
            - Optimizer: Adam
            - Loss: Cross Entropy
            - Early stopping patience: 10
            """
        }
        
        for model, details in training_details.items():
            st.write(f"**{model}**")
            st.code(details)
        
        st.subheader("Model Architecture Details")
        st.write("""
        Common Components:
        - Batch Normalization
        - Dropout layers
        - ReLU activation
        - Max pooling
        
        Model-specific:
        - CNN: Curriculum learning, data augmentation
        - GNN: Graph convolution layers
        - CAE: Encoder-decoder structure
        """)
        
    elif page == "Evaluation":
        st.header("Evaluation Metrics and Results")
        
        st.subheader("Quantitative Metrics")
        st.write("""
        Primary Metrics:
        - Accuracy
        - Detection Rate
        - F1 Score
        - Mean Error
        - Standard Deviation
        
        Additional Metrics:
        - Training Time
        - Inference Time
        - Memory Usage
        - Model Size
        """)
        
        st.subheader("Qualitative Analysis")
        st.write("""
        Model Behavior:
        - CNN: Best for known attack patterns
        - GNN: Effective with graph relationships
        - GCN: Good at detecting novel attacks
        - CAE: Simple but effective baseline
        
        Use Case Performance:
        - Real-time: CNN > GNN > GCN > CAE
        - Accuracy: CNN > GCN > GNN > CAE
        - Resource Usage: CAE < CNN < GNN ≈ GCN
        """)
        
        st.subheader("Comparative Analysis")
        st.write("""
        Performance Trade-offs:
        - Accuracy vs. Speed
        - Complexity vs. Interpretability
        - Resource Usage vs. Performance
        - Training Time vs. Inference Time
        """)
        
    elif page == "Best Model Analysis":
        st.header("Best Model Analysis and Comparison")
        
        st.subheader("Overall Best Model: Supervised GNN (After Hyperparameter Tuning)")
        st.write("""
        The Supervised GNN model emerges as the best performer overall after hyperparameter tuning with Optuna:
        
        1. Performance Metrics:
           - Highest accuracy (96%) - After hyperparameter tuning
           - Best detection rate (96%)
           - Low mean error (0.04)
           - Excellent F1 score (0.96)
        
        2. Technical Advantages:
           - Excellent performance on graph-structured data
           - Captures complex spatial relationships
           - Robust to input variations
           - Handles power system topology effectively
        """)
        
        st.subheader("Random Forest Performance Analysis - Updated Assessment")
        st.write("""
        **IMPORTANT UPDATE:** Random Forest performance has been revised from 97% to 70% accuracy after implementing realistic attack scenarios.
        
        **Previous Assessment (Artificially High Performance):**
        - Simple synthetic data generation created easily distinguishable patterns
        - Decision trees excelled at finding clear decision boundaries
        - Deep learning models appeared overkill for the simplistic task
        
        **Current Assessment (Realistic Performance):**
        
        **1. Realistic Attack Implementation:**
        - Implemented sophisticated attack models (FDI, Replay, Covert attacks)
        - Attack patterns are now more subtle and realistic
        - Decision boundaries are less clear-cut, making the task more challenging
        
        **2. Improved Data Generation:**
        - Attacks now respect power system physics
        - More sophisticated perturbation strategies
        - Better representation of actual attack scenarios
        
        **3. More Challenging Classification:**
        - Attack patterns are now more subtle and realistic
        - Decision boundaries are less clear-cut
        - Better test of model robustness
        
        **4. Model Characteristics:**
        - Still maintains excellent interpretability
        - Handles class imbalance naturally
        - Fast training and inference times
        - Feature importance analysis shows Vm (voltage magnitude) as most important (46.30%)
        """)
        
        st.subheader("GNN Hyperparameter Tuning Comparison")
        st.write("""
| Model                | Accuracy | F1 Score |
|----------------------|----------|----------|
| GNN (Default)        | 0.90     | 0.91     |
| GNN (Tuned/Optuna)   | 0.96     | 0.96     |
""")
        st.markdown("**Hyperparameter tuning with Optuna improved both accuracy and F1 score of the GNN model from 0.90/0.91 to 0.96/0.96.** This demonstrates the significant impact of tuning on model performance.")
        
        st.subheader("GCN Hyperparameter Tuning Comparison")
        st.write("""
| Model                | Accuracy | F1 Score |
|----------------------|----------|----------|
| GCN (Default)        | 0.92     | 0.93     |
| GCN (Tuned/Optuna)   | 0.96     | 0.96     |
""")
        st.markdown("**Hyperparameter tuning with Optuna improved both accuracy and F1 score of the GCN model from 0.92/0.93 to 0.96/0.96.** This demonstrates the significant impact of tuning on model performance.")
        
        st.subheader("Why GNN is Now the Best Model")
        st.write("""
        **CRITICAL UPDATE:** With Random Forest accuracy dropping to 70% after realistic attack implementation, the Supervised GNN emerges as the best model with 96% accuracy after hyperparameter tuning.
        
        **Key Factors Making GNN the Best:**
        
        **1. Superior Performance:**
        - 96% accuracy vs 70% for Random Forest
        - 96% detection rate vs 70% for Random Forest
        - Excellent F1 score of 0.96
        
        **2. Graph Structure Advantage:**
        - Power systems are inherently graph-structured
        - GNN can capture spatial relationships between nodes
        - Better representation of power system topology
        
        **3. Hyperparameter Tuning Impact:**
        - Optuna optimization improved GNN from 90% to 96% accuracy
        - Demonstrates the importance of proper tuning
        - Shows model's potential with optimal parameters
        
        **4. Robustness:**
        - Handles complex attack patterns better than tree-based models
        - More sophisticated feature learning
        - Better generalization to unseen attack types
        """)
        
        st.subheader("Comparison with Other Models")
        comparison_data = {
            "Metric": [
                "Accuracy",
                "Detection Rate",
                "Training Time",
                "Inference Time",
                "Memory Usage",
                "Complexity"
            ],
            "Supervised GNN": [
                "96% (Highest) - After tuning",
                "96% (Highest) - After tuning",
                "60s (Medium)",
                "10ms (Medium)",
                "200MB (High)",
                "100K (High)"
            ],
            "Random Forest": [
                "70% (Medium) - Updated",
                "70% (Medium) - Updated",
                "0.5s (Very Fast)",
                "2ms (Very Fast)",
                "10MB (Very Low)",
                "1K (Low)"
            ],
            "Supervised CNN": [
                "93% (High)",
                "95% (High)",
                "30s (Medium)",
                "5ms (Fast)",
                "100MB (Medium)",
                "150K (High)"
            ],
            "Unsupervised GCN": [
                "95% (High) - After tuning",
                "95% (High) - After tuning",
                "60s (Medium)",
                "10ms (Medium)",
                "200MB (High)",
                "100K (High)"
            ],
            "Supervised GNN": [
                "96% (High) - After tuning",
                "96% (High) - After tuning",
                "60s (Medium)",
                "10ms (Medium)",
                "200MB (High)",
                "100K (High)"
            ],
            "CAE": [
                "82% (Low)",
                "85% (Low)",
                "10s (Fast)",
                "3ms (Fast)",
                "20MB (Low)",
                "50K (Low)"
            ],
            "SVM": [
                "79% (Low)",
                "79% (Low)",
                "20s (Fast)",
                "8ms (Medium)",
                "50MB (Low)",
                "1K (Low)"
            ],
            "RNN": [
                "82% (Low)",
                "82% (Low)",
                "25s (Fast)",
                "7ms (Medium)",
                "30MB (Low)",
                "2K (Low)"
            ],
            "FNN": [
                "80% (Low)",
                "80% (Low)",
                "20s (Fast)",
                "7ms (Medium)",
                "30MB (Low)",
                "2K (Low)"
            ]
        }
        st.dataframe(pd.DataFrame(comparison_data))
        st.markdown("""
        **Legend:**
        - **Accuracy/Detection Rate:** (High: ≥95%, Medium: 90-94%, Low: <90%)
        - **Training Time:** (Very Fast: <1s, Fast: 1-10s, Medium: 10-60s, Slow: >60s)
        - **Inference Time:** (Very Fast: <3ms, Fast: 3-5ms, Medium: 6-15ms, Slow: >15ms)
        - **Memory Usage:** (Very Low: <20MB, Low: 20-50MB, Medium: 51-150MB, High: >150MB)
        - **Complexity:** (Low: <60K, Medium: 60K-120K, High: >120K parameters)
        """)
        
        st.subheader("Why Supervised GNN is the Best Model")
        st.write("""
        1. Performance Advantages:
           - Highest accuracy (96%) after hyperparameter tuning
           - Best detection rate (96%) among all models
           - Excellent F1 score (0.96)
           - Superior performance on graph-structured data
        
        2. Technical Advantages:
           - Captures complex spatial relationships in power systems
           - Handles graph topology effectively
           - Robust to input variations
           - Better generalization to unseen attack patterns
        
        3. Hyperparameter Tuning Impact:
           - Improved from 90% to 96% accuracy with Optuna
           - Demonstrates significant optimization potential
           - Shows model's capability with proper tuning
        
        4. Graph Structure Benefits:
           - Power systems are inherently graph-structured
           - GNN can model node relationships and dependencies
           - Better representation of power system topology
           - More sophisticated feature learning than tree-based models
        """)
        
        st.subheader("Trade-offs and Considerations")
        st.write("""
        While Supervised GNN is the best overall, each model has its niche:
        
        1. Supervised GNN Best For:
           - Highest accuracy requirements (>95%)
           - Graph-structured data analysis
           - Complex spatial relationships
           - When hyperparameter tuning is possible
        
        2. Unsupervised GCN Best For:
           - Novel attack detection
           - When labeled data is scarce
           - Graph-structured data without labels
           - Anomaly detection scenarios
        
        3. Supervised CNN Best For:
           - Complex feature interactions
           - When deep feature extraction is needed
           - Real-time monitoring with GPU acceleration
           - When labeled data is abundant
        
        4. Random Forest Best For:
           - Quick prototyping and deployment
           - When interpretability is crucial
           - Resource-constrained environments
           - Fast inference requirements
        
        5. CAE Best For:
           - Unsupervised learning scenarios
           - When labeled data is scarce
           - Anomaly detection without labels
        """)
        
        st.subheader("Performance Comparison")
        performance_comparison = {
            "Aspect": [
                "Accuracy",
                "Detection Rate",
                "Training Speed",
                "Inference Speed",
                "Memory Usage",
                "Data Requirements",
                "Implementation Complexity",
                "Deployment Ease",
                "Interpretability"
            ],
            "Supervised GNN": [
                "Best (96%)",
                "Best (96%)",
                "Medium",
                "Medium",
                "High",
                "High (labeled)",
                "High",
                "Medium",
                "Medium"
            ],
            "Supervised CNN": [
                "Very Good (93%)",
                "Very Good (95%)",
                "Medium",
                "Very Good",
                "Medium",
                "High (labeled)",
                "Medium",
                "Easy",
                "Low"
            ],
            "Random Forest": [
                "Good (70%)",
                "Good (70%)",
                "Best",
                "Best",
                "Best",
                "Medium (labeled)",
                "Best",
                "Best",
                "Best"
            ],
            "Unsupervised GCN": [
                "Very Good (95%)",
                "Very Good (95%)",
                "Medium",
                "Medium",
                "High",
                "Low (unlabeled)",
                "High",
                "Medium",
                "Medium"
            ],
            "Supervised GNN": [
                "Best (96%)",
                "Best (96%)",
                "Medium",
                "Medium",
                "High",
                "High (labeled)",
                "High",
                "Medium",
                "Medium"
            ],
            "CAE": [
                "Basic (82%)",
                "Basic (85%)",
                "Very Good",
                "Very Good",
                "Very Good",
                "Low (unlabeled)",
                "Very Good",
                "Very Good",
                "Low"
            ],
            "SVM": [
                "Basic (79%)",
                "Basic (79%)",
                "Fast",
                "Fast",
                "Low",
                "Low (labeled)",
                "Low",
                "Easy",
                "Low"
            ],
            "RNN": [
                "Basic (82%)",
                "Basic (82%)",
                "Fast",
                "Fast",
                "Low",
                "Low (labeled)",
                "Low",
                "Easy",
                "Low"
            ],
            "FNN": [
                "Basic (80%)",
                "Basic (80%)",
                "Fast",
                "Fast",
                "Low",
                "Low (labeled)",
                "Low",
                "Easy",
                "Low"
            ]
        }
        st.dataframe(pd.DataFrame(performance_comparison))
        
        st.subheader("Conclusion")
        st.write("""
        The Supervised GNN model is the best choice for most power system attack detection scenarios because:
        
        1. Superior Performance:
           - Highest accuracy and detection rates (96%)
           - Best performance on graph-structured data
           - Excellent after hyperparameter tuning
        
        2. Technical Advantages:
           - Captures complex spatial relationships
           - Handles power system topology effectively
           - Better generalization to unseen attacks
        
        3. Graph Structure Benefits:
           - Power systems are inherently graph-structured
           - GNN can model node relationships and dependencies
           - More sophisticated feature learning than tree-based models
        
        However, the choice of model should still consider specific requirements:
        - Use Supervised GNN for highest accuracy requirements
        - Use Unsupervised GCN for novel attack detection
        - Use Random Forest for quick prototyping and interpretability
        - Use CNN for complex feature interactions
        - Use CAE for unsupervised scenarios
        """)
        
    else:  # Future Work
        st.header("Future Work and Research Directions")
        
        st.subheader("Model Improvements")
        st.write("""
        1. Architecture Enhancements:
           - Attention mechanisms
           - Transformer-based models
           - Hybrid architectures
           - Dynamic graph structures
        
        2. Training Advancements:
           - Self-supervised learning
           - Meta-learning approaches
           - Few-shot learning
           - Active learning
        """)
        
        st.subheader("Application Extensions")
        st.write("""
        1. Domain-specific:
           - Real-time monitoring systems
           - Edge device deployment
           - Distributed detection
           - Multi-modal analysis
        
        2. Integration:
           - With existing security systems
           - Cloud-based deployment
           - IoT device integration
           - Real-time alert systems
        """)
        
        st.subheader("Research Directions")
        st.write("""
        1. Theoretical:
           - Explainable AI approaches
           - Uncertainty quantification
           - Robustness analysis
           - Theoretical guarantees
        
        2. Practical:
           - Deployment optimization
           - Resource efficiency
           - Scalability improvements
           - Real-world validation
        """)

if __name__ == "__main__":
    main()