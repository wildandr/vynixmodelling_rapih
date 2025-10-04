# Dokumentasi Training Script XGB-HMM-LSTM

## Daftar Isi
1. [Gambaran Umum](#gambaran-umum)
2. [Arsitektur Sistem](#arsitektur-sistem)
3. [Kelas dan Komponen Utama](#kelas-dan-komponen-utama)
4. [Alur Data Pipeline](#alur-data-pipeline)
5. [Hyperparameter Tuning](#hyperparameter-tuning)
6. [Proses Training Model](#proses-training-model)
7. [Evaluasi dan Metrik](#evaluasi-dan-metrik)
8. [Output dan Logging](#output-dan-logging)

## Gambaran Umum

Script `xgb_hmm_lstm.py` mengimplementasikan sistem hybrid machine learning yang menggabungkan tiga algoritma utama:
- **XGBoost**: Gradient boosting untuk prediksi dasar
- **Hidden Markov Model (HMM)**: Modeling state transitions dan temporal dependencies
- **LSTM**: Deep learning untuk sequence modeling (opsional, tergantung ketersediaan TensorFlow)

### Tujuan Utama
- Prediksi pergerakan harga saham menggunakan pendekatan ensemble hybrid
- Implementasi Triple Barrier Method untuk labeling data
- Optimasi hyperparameter menggunakan Optuna
- Feature selection dan engineering yang komprehensif

### Dependencies
```python
# Core ML Libraries
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from hmmlearn.hmm import GaussianHMM

# Hyperparameter Optimization
import optuna
from optuna import create_study
from optuna.storages import InMemoryStorage
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner

# Deep Learning (Optional)
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False

# Custom Utilities
from utils.fundamental_feature_engineering import apply_feature_engineering
from utils.technical_fundamental_preprocessing import preprocess_technical_fundamental_data
from triple_barrier.triplebarrier import apply_triple_barrier_labeling
```

## Arsitektur Sistem

### Model Hierarchy
```
XGB-HMM-LSTM System
├── Data Pipeline
│   ├── Technical Data Processing
│   ├── Fundamental Data Processing
│   ├── Feature Engineering
│   └── Triple Barrier Labeling
├── Feature Selection
│   ├── Univariate Selection
│   ├── Mutual Information
│   ├── Recursive Feature Elimination (RFE)
│   └── Feature Importance
├── Model Components
│   ├── LSTMTimeSeriesModel (Optional)
│   ├── XGBHMMModel (Dual Hybrid)
│   └── XGBHMMLSTMModel (Triple Hybrid)
└── Optimization & Evaluation
    ├── Optuna Hyperparameter Tuning
    ├── Cross-validation
    └── Performance Metrics
```

## Kelas dan Komponen Utama

### 1. LSTMTimeSeriesModel
**Lokasi**: Lines 44-210
**Tujuan**: Implementasi LSTM untuk time series prediction

#### Atribut Utama:
```python
def __init__(self, sequence_length=20, lstm_units=50, dropout_rate=0.2, 
             learning_rate=0.001, batch_size=32, epochs=50, random_state=42)
```

#### Metode Utama:
- `_create_sequences(X, y=None)`: Membuat sequences untuk LSTM input
- `_build_model(input_shape, n_classes)`: Membangun arsitektur LSTM
- `fit(X, y)`: Training model LSTM
- `predict(X)`: Prediksi kelas
- `predict_proba(X)`: Prediksi probabilitas

#### Arsitektur LSTM:
```python
model = Sequential([
    LSTM(units=lstm_units, return_sequences=True, input_shape=input_shape),
    Dropout(dropout_rate),
    BatchNormalization(),
    LSTM(units=lstm_units//2, return_sequences=False),
    Dropout(dropout_rate),
    BatchNormalization(),
    Dense(units=lstm_units//4, activation='relu'),
    Dropout(dropout_rate),
    Dense(units=n_classes, activation='softmax')
])
```

### 2. XGBHMMModel
**Lokasi**: Lines 211-564
**Tujuan**: Dual hybrid model menggabungkan XGBoost dan HMM

#### Atribut Utama:
```python
def __init__(self, n_states=3, max_iter=50, tol=1e-4, random_state=42)
```

#### Metode Utama:
- `_initialize_gmm_hmm(X, y)`: Inisialisasi HMM menggunakan GMM
- `_train_xgb_emission_model(X, y, gamma)`: Training XGBoost untuk emission probabilities
- `_compute_emission_probabilities(X)`: Komputasi emission probabilities
- `_forward_backward_algorithm(X)`: Implementasi algoritma Forward-Backward
- `_update_transition_matrix(X, gamma)`: Update transition matrix
- `fit(X, y)`: Training menggunakan EM algorithm
- `predict(X)` & `predict_proba(X)`: Prediksi menggunakan Viterbi algorithm

#### Algoritma EM:
1. **E-step**: Komputasi posterior probabilities menggunakan forward-backward
2. **M-step**: Update parameters (transition matrix, XGBoost model)
3. **Convergence**: Check log-likelihood improvement

### 3. XGBHMMLSTMModel
**Lokasi**: Lines 565-930
**Tujuan**: Triple hybrid model menggabungkan XGBoost, HMM, dan LSTM

#### Ensemble Strategy:
```python
def __init__(self, ensemble_weights=None):
    # Default weights: [XGB, HMM, LSTM]
    self.ensemble_weights = ensemble_weights or [0.4, 0.3, 0.3]
```

#### Prediksi Ensemble:
```python
def predict_proba(self, X):
    xgb_proba = self._compute_emission_probabilities(X)
    hmm_proba = self._forward_backward_algorithm(X)[1]
    lstm_proba = self.lstm_model.predict_proba(X)
    
    # Weighted ensemble
    ensemble_proba = (self.ensemble_weights[0] * xgb_proba + 
                     self.ensemble_weights[1] * hmm_proba + 
                     self.ensemble_weights[2] * lstm_proba)
    return ensemble_proba
```

### 4. FeatureSelector
**Lokasi**: Lines 931-1125
**Tujuan**: Comprehensive feature selection menggunakan multiple methods

#### Metode Selection:
1. **Univariate Selection**: Statistical tests (f_classif, chi2)
2. **Mutual Information**: Information-theoretic approach
3. **Recursive Feature Elimination (RFE)**: Iterative feature removal
4. **Feature Importance**: XGBoost-based importance scores
5. **Ensemble Selection**: Voting mechanism across methods

#### Implementasi Ensemble:
```python
def ensemble_selection(self, X, y, methods=['univariate', 'mutual_info', 'rfe', 'feature_importance'], 
                      min_votes=2):
    # Combine multiple selection methods
    # Features selected by >= min_votes methods are retained
```

### 5. EnhancedProgressCallback
**Lokasi**: Lines 1128-1156
**Tujuan**: Advanced callback untuk Optuna optimization dengan early stopping

#### Features:
- **Patience mechanism**: Stop jika tidak ada improvement
- **Minimum improvement threshold**: Avoid overfitting pada small improvements
- **Maximum trial time**: Time-based constraints
- **Progress tracking**: Detailed logging

## Alur Data Pipeline

### 1. Data Sources

#### 1.1 Technical Data
- **Source File**: `datasets/technical/TSLA_original.csv`
- **Asset**: Tesla Inc. (TSLA) stock data
- **Data Range**: 2015-05-13 to 2025-01-17 (approximately 10 years)
- **Total Records**: 3,817 daily observations
- **Features**: 
  - Basic OHLCV data (Open, High, Low, Close, Volume)
  - Technical indicators (RSI, MACD, Bollinger Bands, etc.)
  - Timestamp in Unix format

#### 1.2 Fundamental Data
- **Source File**: `datasets/fundamental/TSLA_enhanced_features.csv`
- **Data Type**: Quarterly financial statements and ratios
- **Coverage Period**: 2012-Q2 to 2025-Q2 (52 quarters)
- **Conversion Method**: Quarterly data converted to daily using quarter-shift mapping
- **Default Date Range for Daily Conversion**: 2012-04-01 to 2025-06-30
- **Features**: Enhanced fundamental metrics including:
  - Financial ratios
  - Growth metrics
  - Profitability indicators
  - Liquidity measures
  - Leverage ratios

### 2. Data Loading dan Preprocessing
**Lokasi**: Lines 1202-1237

#### 2.1 Technical Data Processing
```python
# Load original technical data
df = original_df.copy()
non_null_df = df.dropna()
```
1. **Timestamp Conversion**: Unix timestamps converted to readable dates
2. **Missing Value Handling**: Rows with NaN values removed
3. **Feature Scaling**: Applied where necessary for model compatibility
4. **Date Alignment**: Synchronized with fundamental data timeline

#### 2.2 Fundamental Data Processing
```python
# Process fundamental data
pivoted_df = process_fundamental_data_local("TSLA")
enhanced_fundamental_df = apply_feature_engineering(pivoted_df, "TSLA")
```
1. **Quarterly to Daily Conversion**:
   - Uses quarter-shift mapping methodology
   - Q4 data applied to Q1 of next year (Jan-Mar)
   - Q1 data applied to Q2 of same year (Apr-Jun)
   - Q2 data applied to Q3 of same year (Jul-Sep)
   - Q3 data applied to Q4 of same year (Oct-Dec)
2. **Forward Fill**: Missing values filled using forward propagation
3. **Feature Engineering**: Enhanced metrics calculated using `apply_feature_engineering()`
4. **Data Alignment**: Merged with technical data on date basis

#### 2.3 Data Integration
```python
# Merge technical and fundamental data
filtered_df = preprocess_technical_fundamental_data(
    technical_df=non_null_df,
    fundamental_df=enhanced_fundamental_df,
    output_path='datasets/processed/main_processed_data.csv',
    start_period='2012-Q2',
    end_period='2025-Q2'
)
```
- **Preprocessing Function**: `preprocess_technical_fundamental_data()`
- **Period Filter**: Data filtered from 2012-Q2 to 2025-Q2
- **Output**: Combined dataset saved to `datasets/processed/main_processed_data.csv`
- **Final Shape**: Varies based on available features and date range

### 3. Triple Barrier Labeling
**Lokasi**: Lines 1239-1280

The system uses triple barrier method for generating trading labels:
- **Upper Barrier**: Profit-taking level
- **Lower Barrier**: Stop-loss level  
- **Time Barrier**: Maximum holding period

#### Parameters:
```python
TRIPLE_BARRIER_PARAMS = {
    'volatility_window': 20,
    'upper_barrier_multiplier': 1.0,
    'lower_barrier_multiplier': 1.0,
    'time_barrier_days': 15,
    'verbose': True
}
```

#### Process:
1. **Volatility Calculation**: Rolling window volatility
2. **Barrier Setting**: Upper/lower price barriers
3. **Time Barrier**: Maximum holding period
4. **Label Assignment**: 
   - `1`: Upper barrier touched (profit)
   - `-1`: Lower barrier touched (loss)
   - `0`: Time barrier reached (neutral)

**Output Files**:
- Triple barrier results: `logs/triple_barrier/triple_barrier_results.csv`
- Merged labeled data: `datasets/processed/merged_labeled_data.csv`

### 4. Data Splitting Strategy
**Lokasi**: Lines 1300-1350

#### 4.1 Time-Based Split Methodology
The system uses time-based splitting to maintain temporal order and prevent data leakage:

1. **Test Set Definition**:
   - **Cutoff Date**: July 1, 2024 (`q3_2024_start = pd.Timestamp('2024-07-01')`)
   - **Rationale**: Represents Q3 2024 onwards for out-of-sample testing
   - **Purpose**: Evaluate model performance on most recent market conditions

2. **Train/Validation Split**:
   - **Data**: All observations before July 1, 2024
   - **Split Ratio**: 80% training, 20% validation
   - **Method**: `train_test_split()` with `test_size=0.20`
   - **Random State**: 42 (for reproducibility)
   - **Stratification**: Applied based on target labels to maintain class distribution

```python
# Time-based split
q3_2024_start = pd.Timestamp('2024-07-01')
test_mask = merged_df['decision_date'] >= q3_2024_start
train_val_mask = ~test_mask

# Further split train_val into train and validation
X_train, X_val, y_train, y_val = train_test_split(
    X_train_val, y_train_val, 
    test_size=0.2, 
    random_state=42, 
    stratify=y_train_val
)
```

#### 4.2 Data Distribution
- **Training Set**: ~64% of total data (80% of pre-2024 data)
- **Validation Set**: ~16% of total data (20% of pre-2024 data)  
- **Test Set**: ~20% of total data (Q3 2024 onwards)

#### 4.3 Temporal Considerations
- **No Future Information**: Strict temporal ordering maintained
- **Market Regime Changes**: Test set captures recent market dynamics
- **Seasonal Effects**: All seasons represented across splits
- **Data Integrity**: No overlap between train/validation/test sets

## Hyperparameter Tuning

### 1. Objective Functions

#### XGB-HMM Objective (`combined_xgb_hmm_objective`)
**Lokasi**: Lines 1480-1596

```python
def combined_xgb_hmm_objective(trial):
    # HMM Parameters
    hmm_params = {
        'n_states': trial.suggest_int('n_states', 2, 5),
        'max_iter': trial.suggest_int('max_iter', 20, 1000, step=10),
        'tol': trial.suggest_float('tol', 1e-6, 1e-2, log=True)
    }
    
    # XGBoost Parameters
    xgb_params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 500, step=50),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0, step=0.1),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0, step=0.1),
        'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 10.0, step=0.1),
        'reg_lambda': trial.suggest_float('reg_lambda', 1.0, 10.0, step=0.1)
    }
    
    # Triple Barrier Parameters
    barrier_params = {
        'volatility_window': trial.suggest_int('volatility_window', 5, 100, step=3),
        'upper_barrier_multiplier': trial.suggest_float('barrier_multiplier', 0.5, 3.0, step=0.1),
        'lower_barrier_multiplier': trial.suggest_float('barrier_multiplier', 0.5, 3.0, step=0.1),
        'time_barrier_days': trial.suggest_int('time_barrier_days', 3, 35)
    }
```

#### XGB-HMM-LSTM Objective (`combined_xgb_hmm_lstm_objective`)
**Lokasi**: Lines 1600-1750

```python
# Additional LSTM Parameters (when TensorFlow available)
if TENSORFLOW_AVAILABLE:
    lstm_params = {
        'sequence_length': trial.suggest_int('sequence_length', 5, 500, step=5),
        'lstm_units': trial.suggest_int('lstm_units', 16, 612, step=8),
        'dropout_rate': trial.suggest_float('dropout_rate', 0.1, 0.5, step=0.001),
        'learning_rate': trial.suggest_float('lstm_learning_rate', 0.0001, 0.1, log=True)
    }
    
    # Ensemble Weights
    xgb_weight = trial.suggest_float('xgb_weight', 0.2, 0.6, step=0.01)
    hmm_weight = trial.suggest_float('hmm_weight', 0.1, 0.5, step=0.01)
    lstm_weight = 1.0 - xgb_weight - hmm_weight
    
    # Ensure minimum weight for LSTM
    if lstm_weight < 0.1:
        lstm_weight = 0.1
        total = xgb_weight + hmm_weight + lstm_weight
        xgb_weight /= total
        hmm_weight /= total
        lstm_weight /= total
```

### 2. Optuna Study Configuration
**Lokasi**: Lines 1752-1850

```python
def create_study_with_storage(study_name, direction='maximize'):
    storage = InMemoryStorage()
    sampler = TPESampler(seed=42, n_startup_trials=10, n_ei_candidates=24)
    pruner = MedianPruner(n_startup_trials=5, n_warmup_steps=10, interval_steps=1)
    
    study = create_study(
        study_name=study_name,
        direction=direction,
        storage=storage,
        sampler=sampler,
        pruner=pruner
    )
    return study

# Study execution
study = create_study_with_storage("XGB_HMM_LSTM_Optimization")
callback = EnhancedProgressCallback("XGB_HMM_LSTM", patience=10, min_improvement=0.001)

study.optimize(
    objective_function,
    n_trials=100,
    callbacks=[callback],
    show_progress_bar=True
)
```

## Proses Training Model

### 1. Feature Selection Pipeline
**Lokasi**: Lines 1900-2000

```python
# Initialize feature selector
feature_selector = FeatureSelector(random_state=42)

# Apply ensemble feature selection
feature_summary = feature_selector.ensemble_selection(
    X_train, y_train,
    methods=['univariate', 'mutual_info', 'rfe', 'feature_importance'],
    k_univariate=100,
    k_mutual_info=100,
    n_rfe=100,
    n_importance=100,
    min_votes=2
)

# Transform datasets
X_train_selected = feature_selector.transform(X_train)
X_val_selected = feature_selector.transform(X_val)
X_test_selected = feature_selector.transform(X_test)
```

### 2. Model Selection dan Training
**Lokasi**: Lines 2020-2090

```python
# Conditional model selection
if TENSORFLOW_AVAILABLE and final_lstm_params is not None:
    # Triple Hybrid Model
    logging.info("Training XGB-HMM-LSTM Triple Hybrid Model")
    model_to_evaluate = XGBHMMLSTMModel(
        n_states=final_hmm_params['n_states'],
        max_iter=final_hmm_params['max_iter'],
        tol=final_hmm_params['tol'],
        sequence_length=final_lstm_params['sequence_length'],
        lstm_units=final_lstm_params['lstm_units'],
        dropout_rate=final_lstm_params['dropout_rate'],
        learning_rate=final_lstm_params['learning_rate'],
        ensemble_weights=final_ensemble_weights,
        random_state=42
    )
    model_name = "XGB-HMM-LSTM"
else:
    # Dual Hybrid Model
    logging.info("Training XGB-HMM Dual Hybrid Model")
    model_to_evaluate = XGBHMMModel(
        n_states=final_hmm_params['n_states'],
        max_iter=final_hmm_params['max_iter'],
        tol=final_hmm_params['tol'],
        random_state=42
    )
    model_name = "XGB-HMM"

# Training
start_time = time.time()
model_to_evaluate.fit(X_train_selected, y_train_mapped)
training_time = time.time() - start_time
```

### 3. EM Algorithm Implementation (XGBHMMModel)
**Lokasi**: Lines 409-455

```python
def fit(self, X, y):
    # Initialize HMM
    self._initialize_gmm_hmm(X, y)
    
    prev_log_likelihood = -np.inf
    self.log_likelihood_history_ = []
    
    for iteration in range(self.max_iter):
        # E-step: Forward-backward algorithm
        log_alpha, log_beta = self._forward_backward_algorithm(X)
        
        # Compute gamma (posterior probabilities)
        log_gamma = log_alpha + log_beta
        gamma = np.exp(log_gamma - logsumexp(log_gamma, axis=1, keepdims=True))
        
        # M-step: Update parameters
        # 1. Update transition matrix
        self._update_transition_matrix(X, gamma)
        
        # 2. Update XGBoost emission model
        self._train_xgb_emission_model(X, y, gamma)
        
        # Check convergence
        current_log_likelihood = self._compute_log_likelihood(X)
        self.log_likelihood_history_.append(current_log_likelihood)
        
        if abs(current_log_likelihood - prev_log_likelihood) < self.tol:
            self.converged_ = True
            break
            
        prev_log_likelihood = current_log_likelihood
```

## Evaluasi dan Metrik

### 1. Metrik Evaluasi Komprehensif
**Lokasi**: Lines 2119-2190

```python
# Training Set Evaluation
y_train_pred = model.predict(X_train_selected)
y_train_proba = model.predict_proba(X_train_selected)

train_accuracy = accuracy_score(y_train_mapped, y_train_pred)
train_precision = precision_score(y_train_mapped, y_train_pred, average='weighted')
train_recall = recall_score(y_train_mapped, y_train_pred, average='weighted')
train_f1 = f1_score(y_train_mapped, y_train_pred, average='weighted')

# ROC AUC handling for multi-class
if num_classes == 2:
    train_roc_auc = roc_auc_score(y_train_mapped, y_train_proba[:, 1])
else:
    train_roc_auc = roc_auc_score(y_train_mapped, y_train_proba, multi_class='ovr')
```

### 2. Performance Comparison
**Lokasi**: Lines 2209-2220

```python
performance_comparison = pd.DataFrame({
    'Dataset': ['Training', 'Validation', 'Test'],
    'Accuracy': [train_accuracy, val_accuracy, test_accuracy],
    'Precision': [train_precision, val_precision, test_precision],
    'Recall': [train_recall, val_recall, test_recall],
    'F1-Score': [train_f1, val_f1, test_f1],
    'ROC AUC': [train_roc_auc, val_roc_auc, test_roc_auc]
})
```

### 3. Model Analysis
**Lokasi**: Lines 2248-2290

#### Hyperparameter Summary:
- Model parameters (n_states, max_iter, tolerance)
- Training time dan convergence status
- Final log-likelihood

#### Performance Analysis:
- Comparison dengan baseline XGBoost
- Generalization analysis (train-val-test gaps)
- Feature selection impact analysis

#### Feature Importance Analysis:
```python
if 'feature_importance' in feature_summary:
    importance_scores = feature_summary['feature_importance']['scores']
    top_indices = sorted(range(len(importance_scores)), 
                        key=lambda i: importance_scores[i], reverse=True)[:10]
    top_features = [selected_feature_names[i] for i in top_indices]
```

## Output dan Logging

### 1. Logging Configuration
**Lokasi**: Lines 1175-1200

```python
# Create log directories
os.makedirs('logs/main', exist_ok=True)
os.makedirs('logs/xgb_results', exist_ok=True)

# Setup logging
current_datetime = datetime.now().strftime('%Y%m%d_%H%M%S')
log_filename = f'logs/main/main_{current_datetime}.log'

# File and console handlers
file_handler = logging.FileHandler(log_filename)
console_handler = logging.StreamHandler()

# Formatters
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
```

### 2. Output Files Generated

#### Model dan Results:
- `logs/xgb_models/{model_name}_model.pkl`: Trained model
- `logs/xgb_results/final_performance_comparison.csv`: Performance metrics
- `logs/xgb_results/final_test_predictions.csv`: Test predictions dengan probabilities

#### Data dan Visualizations:
- `datasets/processed/main_processed_data.csv`: Processed dataset
- `logs/triple_barrier/triple_barrier_results.csv`: Triple barrier labels
- `logs/visualization/`: Triple barrier visualizations

#### Logs:
- `logs/main/main_{timestamp}.log`: Comprehensive training logs
- Real-time console output dengan progress tracking

### 3. Detailed Logging Information

#### Data Processing Logs:
```python
logging.info(f"Original dataframe shape: {df.shape}")
logging.info(f"Final processed data shape: {filtered_df.shape}")
logging.info(f"Date range: {filtered_df['date'].min()} to {filtered_df['date'].max()}")
logging.info(f"Data completeness: {completeness:.2f}%")
```

#### Training Progress Logs:
```python
logging.info(f"Training {model_name} model...")
logging.info(f"Training time: {training_time:.2f} seconds")
logging.info(f"Convergence achieved: {model.converged_}")
logging.info(f"Final log-likelihood: {model.log_likelihood_history_[-1]}")
```

#### Performance Logs:
```python
logging.info(f"Test Set Performance:")
logging.info(f"Accuracy: {test_accuracy:.4f}")
logging.info(f"Precision: {test_precision:.4f}")
logging.info(f"Recall: {test_recall:.4f}")
logging.info(f"F1-Score: {test_f1:.4f}")
logging.info(f"ROC AUC: {test_roc_auc:.4f}")
```

## Kesimpulan

Script `xgb_hmm_lstm.py` mengimplementasikan sistem machine learning hybrid yang sophisticated dengan:

1. **Modular Architecture**: Kelas-kelas terpisah untuk setiap komponen
2. **Flexible Model Selection**: Conditional usage berdasarkan dependency availability
3. **Comprehensive Feature Engineering**: Multiple selection methods dengan ensemble approach
4. **Advanced Optimization**: Optuna-based hyperparameter tuning dengan early stopping
5. **Robust Evaluation**: Multi-metric evaluation dengan detailed analysis
6. **Production-Ready Logging**: Comprehensive logging dan output management

Sistem ini dirancang untuk memberikan prediksi yang akurat pada financial time series dengan menggabungkan kekuatan dari gradient boosting, probabilistic modeling, dan deep learning dalam satu framework yang terintegrasi.