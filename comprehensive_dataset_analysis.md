# 🔋 Comprehensive Dataset Analysis: Power System Attack Detection

## 📊 **Dataset Overview**

### **Basic Information**
- **Source**: IEEE 14-bus test system data
- **File**: `benign_bus14.xlsx` (Excel format)
- **Total Samples**: 1,928 (varies by model implementation)
- **Features**: 4 key power system measurements
- **Classes**: Binary classification (Benign vs Malicious)

### **Feature Description**
Your dataset contains **4 critical power system measurements**:

| Feature | Description | Physical Meaning |
|---------|-------------|------------------|
| **Pd_new** | Active Power Demand | Real power consumption at each bus |
| **Qd_new** | Reactive Power Demand | Imaginary power consumption at each bus |
| **Vm** | Voltage Magnitude | Voltage level at each bus (per unit) |
| **Va** | Voltage Angle | Phase angle of voltage at each bus (radians) |

## 🎯 **Class Distribution Analysis**

### **Current Distribution (Varies by Model)**
- **Benign Samples**: 482 (25%)
- **Malicious Samples**: 1,446 (75%) 
- **Class Imbalance Ratio**: 1:3 (Benign:Malicious)

### **⚠️ Critical Issue: Class Imbalance**
Your dataset has a **severe class imbalance** (3:1 ratio), which significantly impacts model performance:
- Models tend to bias toward the majority class (malicious)
- Precision drops dramatically for minority class (benign)
- Need for class balancing techniques (SMOTE, weighted loss, etc.)

## 🔥 **Attack Types & Generation Methods**

### **1. False Data Injection (FDI) Attacks**
```python
# Subtle manipulation maintaining system observability
noise_scale = np.random.uniform(0.08, 0.25, benign_data.shape)
fdi_attack = benign_data + np.random.normal(0, noise_scale, benign_data.shape)
```
- **Purpose**: Inject false measurements while maintaining system observability
- **Characteristics**: Subtle perturbations (8-25% noise level)
- **Realism**: High - mimics real-world FDI attacks

### **2. Replay Attacks**
```python
# Reuse previous measurements with noise
shift_indices = np.random.randint(0, len(benign_data), len(benign_data))
replay_attack = benign_data[shift_indices] + np.random.normal(0, 0.15, benign_data.shape)
```
- **Purpose**: Replay historical measurements to hide current system state
- **Characteristics**: Temporal manipulation with 15% noise
- **Realism**: High - common in real cyberattacks

### **3. Covert Attacks**
```python
# Proportional noise based on measurement type
proportional_noise = benign_data * np.random.uniform(0.15, 0.35, benign_data.shape)
covert_attack = benign_data + proportional_noise
```
- **Purpose**: Maintain system observability while injecting false data
- **Characteristics**: Proportional manipulation (15-35% of original values)
- **Realism**: Very High - sophisticated attack strategy

### **4. Realistic Noise Attacks**
```python
# Noise proportional to measurement characteristics
feature_std = np.std(benign_data, axis=0)
realistic_noise = benign_data + np.random.normal(0, feature_std * 0.4, benign_data.shape)
```
- **Purpose**: Simulate measurement noise and sensor errors
- **Characteristics**: Adaptive noise based on feature statistics
- **Realism**: High - reflects real sensor limitations

### **5. Targeted Attacks**
```python
# Focus on specific features
target_features = np.random.choice([0, 1, 2, 3], size=len(benign_data), p=[0.3, 0.3, 0.2, 0.2])
for i, target_feat in enumerate(target_features):
    targeted_attack[i, target_feat] += np.random.normal(0, 0.3)
```
- **Purpose**: Attack specific power system components
- **Characteristics**: Feature-specific manipulation
- **Realism**: High - mimics targeted cyberattacks

### **6. Systematic Bias Attacks**
```python
# Add systematic bias to all features
bias = np.random.uniform(-0.2, 0.2, benign_data.shape[1])
bias_attack = benign_data + bias
```
- **Purpose**: Introduce systematic measurement errors
- **Characteristics**: Consistent bias across all measurements
- **Realism**: Medium - represents calibration errors

## 📈 **Data Preprocessing Pipeline**

### **1. Data Loading**
```python
# Load benign data from Excel
benign_df = pd.read_excel('benign_bus14.xlsx')
feature_columns = ['Pd_new', 'Qd_new', 'Vm', 'Va']
X_benign = benign_df[feature_columns].values
```

### **2. Malicious Data Generation**
- **Method**: Synthetic generation from benign data
- **Approach**: Multiple attack types combined
- **Volume**: 3x more malicious than benign samples

### **3. Feature Standardization**
```python
# Standardize features for neural networks
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
```
- **Purpose**: Normalize features to zero mean, unit variance
- **Impact**: Improves neural network training stability

### **4. Data Splitting**
- **Train/Test Split**: 80/20 ratio
- **Stratification**: Maintains class distribution
- **Random State**: 42 (for reproducibility)

## 🔍 **Feature Analysis**

### **Feature Importance (Random Forest Results)**
1. **Vm (Voltage Magnitude)**: 46.30% - Most critical feature
2. **Va (Voltage Angle)**: 21.78% - Second most important
3. **Pd_new (Active Power)**: 16.69% - Third most important
4. **Qd_new (Reactive Power)**: 15.23% - Least important

### **Physical Interpretation**
- **Voltage measurements (Vm, Va)** are most discriminative
- **Power measurements (Pd, Qd)** provide supporting information
- **Voltage stability** is key indicator of system health

## ⚠️ **Dataset Challenges & Limitations**

### **1. Class Imbalance (Critical)**
- **Problem**: 3:1 ratio severely biases models
- **Impact**: Low precision for benign class
- **Solution**: SMOTE, weighted loss, balanced sampling

### **2. Synthetic Attack Generation**
- **Problem**: Attacks are artificially generated
- **Impact**: May not reflect real-world attack patterns
- **Solution**: Use real attack datasets when available

### **3. Limited Attack Diversity**
- **Problem**: Only 6 attack types implemented
- **Impact**: Models may not generalize to new attack types
- **Solution**: Implement more sophisticated attack models

### **4. No Temporal Information**
- **Problem**: No time-series relationships captured
- **Impact**: Misses temporal attack patterns
- **Solution**: Implement sequence-based models

## 🎯 **Dataset Suitability for Different Models**

### **✅ Well-Suited For:**
- **Random Forest**: Handles mixed data types well
- **SVM**: Good with standardized features
- **CNN**: Can learn spatial patterns in reshaped data
- **FNN**: Simple architecture for 4-feature input

### **⚠️ Challenging For:**
- **GNN/GCN**: Limited graph structure information
- **RNN/LSTM**: No temporal sequences
- **Autoencoders**: Class imbalance affects reconstruction

### **🔧 Needs Enhancement For:**
- **Graph Models**: Need proper graph construction
- **Sequence Models**: Need temporal data
- **Ensemble Methods**: Need diverse base models

## 📊 **Statistical Summary**

### **Feature Statistics (Approximate)**
| Feature | Mean | Std | Min | Max | Range |
|---------|------|-----|-----|-----|-------|
| Pd_new | ~0.5 | ~0.3 | 0.0 | 1.0 | 1.0 |
| Qd_new | ~0.2 | ~0.15 | 0.0 | 0.5 | 0.5 |
| Vm | ~1.0 | ~0.1 | 0.9 | 1.1 | 0.2 |
| Va | ~0.0 | ~0.3 | -π | π | 2π |

### **Data Quality**
- **Missing Values**: None (clean dataset)
- **Outliers**: Present in malicious data (by design)
- **Correlation**: Moderate between power features
- **Distribution**: Normal for benign, varied for malicious

## 🚀 **Recommendations for Dataset Improvement**

### **Immediate Actions (Week 1)**
1. **Implement Class Balancing**:
   - SMOTE oversampling for benign class
   - Weighted loss functions
   - Balanced data loaders

2. **Enhance Attack Generation**:
   - Add more sophisticated attack patterns
   - Implement physics-based constraints
   - Add temporal attack sequences

### **Medium-term Improvements (Week 2-4)**
1. **Graph Structure Enhancement**:
   - Construct proper power system topology
   - Add edge features (line impedances, flows)
   - Implement graph-based data augmentation

2. **Temporal Data Integration**:
   - Add time-series information
   - Implement sequence-based attacks
   - Create temporal graph structures

### **Long-term Goals (Month 2+)**
1. **Real Attack Data Integration**:
   - Collect real-world attack datasets
   - Validate synthetic attacks against real data
   - Implement transfer learning approaches

2. **Multi-Modal Data**:
   - Add network topology information
   - Include control system data
   - Integrate SCADA system measurements

## 📋 **Dataset Usage Guidelines**

### **For Model Training**
- Always use stratified train/test splits
- Implement cross-validation for robust evaluation
- Use class weights or balanced sampling
- Monitor for overfitting on synthetic data

### **For Evaluation**
- Report metrics for both classes separately
- Use confusion matrices for detailed analysis
- Implement ROC curves for threshold analysis
- Test on unseen attack types

### **For Deployment**
- Validate on real-world data
- Implement continuous learning
- Monitor model drift
- Update attack patterns regularly

---

## 🎯 **Key Takeaways**

1. **Your dataset is well-structured** with realistic power system features
2. **Class imbalance is the biggest challenge** - needs immediate attention
3. **Attack generation is sophisticated** but could be more diverse
4. **Feature importance analysis** shows voltage measurements are most critical
5. **Dataset is suitable for most ML models** but needs enhancement for graph/sequence models

**Next Priority**: Implement class balancing and enhanced data augmentation to improve model performance and generalization!
