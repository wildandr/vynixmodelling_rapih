logging.info("8. HYPERPARAMETER TUNING")
logging.info("="*80)

from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import make_scorer
import optuna
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner
import time
import os
from datetime import datetime
try:
    import plotly.graph_objects as go
    import plotly.express as px
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    logging.info("Warning: Plotly not available. Visualization will be skipped.")
import warnings
warnings.filterwarnings('ignore')

class EnhancedProgressCallback:
    def __init__(self, study_name, patience=10, min_improvement=0.001, max_trial_time=300):
        self.study_name = study_name
        self.patience = patience
        self.min_improvement = min_improvement
        self.max_trial_time = max_trial_time
        self.best_score = -float('inf')
        self.trials_without_improvement = 0
        self.trial_start_times = {}
        
    def __call__(self, study, trial):
        if trial.number not in self.trial_start_times:
            self.trial_start_times[trial.number] = time.time()
        
        elapsed_time = time.time() - self.trial_start_times[trial.number]
        if elapsed_time > self.max_trial_time:
            logging.warning(f"Trial {trial.number} exceeded {self.max_trial_time}s timeout, pruning...")
            raise optuna.TrialPruned()
        
        if trial.value is not None:
            if trial.value > self.best_score + self.min_improvement:
                self.best_score = trial.value
                self.trials_without_improvement = 0
                logging.info(f"{self.study_name} - Trial {trial.number}: New best score {trial.value:.4f}")
            else:
                self.trials_without_improvement += 1
                
            if self.trials_without_improvement >= self.patience:
                logging.info(f"{self.study_name} - Early stopping after {self.patience} trials without improvement")
                study.stop()

def create_study_with_storage(study_name, direction='maximize'):
    import os
    os.makedirs('logs/optuna_studies', exist_ok=True)
    
    db_path = f'logs/optuna_studies/{study_name}.db'
    storage = f'sqlite:///{db_path}'
    
    study = optuna.create_study(
        study_name=study_name,
        storage=storage,
        direction=direction,
        load_if_exists=True
    )
    
    logging.info(f"Study '{study_name}' created/loaded with storage: {db_path}")
    return study

logging.info("\nCombined HMM, XGBoost, and Barrier Parameters Tuning with Optuna")
logging.info("-" * 50)

def combined_xgb_hmm_objective(trial):
    try:
        hmm_params = {
            'n_states': trial.suggest_int('n_states', 2, 5),
            'max_iter': trial.suggest_int('max_iter', 20, 100, step=10),
            'tol': trial.suggest_float('tol', 1e-6, 1e-2, log=True)
        }
        
        xgb_params = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 2000, step=50),
            'max_depth': trial.suggest_int('max_depth', 2, 100),
            'learning_rate': trial.suggest_float('learning_rate', 0.0001, 0.3, log=True),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0, step=0.001),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0, step=0.001),
            'colsample_bylevel': trial.suggest_float('colsample_bylevel', 0.6, 1.0, step=0.001),
            'colsample_bynode': trial.suggest_float('colsample_bynode', 0.6, 1.0, step=0.001),
            'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 10.0, step=0.01),
            'reg_lambda': trial.suggest_float('reg_lambda', 1.0, 10.0, step=0.01),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
            'gamma': trial.suggest_float('gamma', 0.0, 5.0, step=0.01)
        }
        
        barrier_multiplier = trial.suggest_float('barrier_multiplier', 0.5, 10.0, step=0.01)
        
        barrier_params = {
            'volatility_window': trial.suggest_int('volatility_window', 5, 70, step=5),
            'upper_barrier_multiplier': barrier_multiplier,
            'lower_barrier_multiplier': barrier_multiplier,
            'time_barrier_days': trial.suggest_int('time_barrier_days', 3, 35),
            'verbose': False
        }
        
        temp_triple_barrier_df = apply_triple_barrier_labeling(
            data=filtered_df,
            **barrier_params
        )
        
        temp_triple_barrier_df['decision_date'] = pd.to_datetime(temp_triple_barrier_df['decision_date'])
        temp_merged_df = temp_triple_barrier_df.merge(
            filtered_df, 
            left_on='decision_date', 
            right_on='date', 
            how='left'
        )
        
        if 'date' in temp_merged_df.columns:
            temp_merged_df = temp_merged_df.drop('date', axis=1)
        
        temp_columns_to_drop = [
            'decision_date', 'entry_date', 'end_date', 'end_price',
            'return', 'barrier_touched', 'value_at_barrier_touched',
            'time_converted', 'datetime', 'time', 'time_barrier'
        ]
        temp_existing_columns_to_drop = [col for col in temp_columns_to_drop if col in temp_merged_df.columns]
        
        if temp_existing_columns_to_drop:
            df_with_labels_new = temp_merged_df.drop(columns=temp_existing_columns_to_drop)
        else:
            df_with_labels_new = temp_merged_df.copy()
        
        df_with_labels_new = df_with_labels_new.dropna(subset=['label'])
        
        if len(df_with_labels_new) < 100:
            return 0.0
        
        merged_new_df = df_with_labels_new.copy()
        merged_new_df['decision_date'] = temp_triple_barrier_df['decision_date']
        
        q3_2024_start_new = pd.Timestamp('2024-07-01')
        
        test_mask_new = merged_new_df['decision_date'] >= q3_2024_start_new
        train_val_mask_new = ~test_mask_new
        
        train_val_data_new = merged_new_df[train_val_mask_new].copy()
        
        X_train_val_new = train_val_data_new.drop(['label', 'decision_date'], axis=1)
        y_train_val_new = train_val_data_new['label']
        
        X_train_new, X_val_new, y_train_new, y_val_new = train_test_split(
            X_train_val_new, y_train_val_new, test_size=0.20, random_state=42
        )
        
        unique_labels_new = sorted(y_train_new.unique())
        num_classes_new = len(unique_labels_new)
        
        if num_classes_new == 2:
            label_mapping_new = {unique_labels_new[0]: 0, unique_labels_new[1]: 1}
        else:
            label_mapping_new = {label: idx for idx, label in enumerate(unique_labels_new)}
        
        y_train_mapped_new = y_train_new.map(label_mapping_new)
        y_val_mapped_new = y_val_new.map(label_mapping_new)
        
        suggested_n_states = hmm_params['n_states']
        actual_n_states = max(num_classes_new, suggested_n_states)
        
        temp_xgb_hmm_model = XGBHMMModel(
            n_states=actual_n_states,
            max_iter=hmm_params['max_iter'],
            tol=hmm_params['tol'],
            random_state=42
        )
        
        xgb_params['verbosity'] = 0
        temp_xgb_hmm_model.xgb_params = xgb_params
        
        temp_xgb_hmm_model.fit(X_train_new, y_train_mapped_new)
        
        y_val_pred_new = temp_xgb_hmm_model.predict(X_val_new)
        val_accuracy_new = accuracy_score(y_val_mapped_new, y_val_pred_new)
        
        return val_accuracy_new
        
    except Exception as e:
        logging.warning(f"Trial failed with error: {str(e)}")
        return 0.0

def combined_xgb_hmm_lstm_objective(trial):
    try:
        hmm_params = {
            'n_states': trial.suggest_int('n_states', 2, 5),
            'max_iter': trial.suggest_int('max_iter', 20, 1000, step=10),
            'tol': trial.suggest_float('tol', 1e-6, 1e-2, log=True)
        }
        
        xgb_params = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 500, step=50),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0, step=0.1),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0, step=0.1),
            'colsample_bylevel': trial.suggest_float('colsample_bylevel', 0.6, 1.0, step=0.01),
            'colsample_bynode': trial.suggest_float('colsample_bynode', 0.6, 1.0, step=0.01),
            'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 10.0, step=0.1),
            'reg_lambda': trial.suggest_float('reg_lambda', 1.0, 10.0, step=0.1),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
            'gamma': trial.suggest_float('gamma', 0.0, 5.0, step=0.1)
        }
        
        if TENSORFLOW_AVAILABLE:
            lstm_params = {
                'sequence_length': trial.suggest_int('sequence_length', 5, 500, step=5),
                'lstm_units': trial.suggest_int('lstm_units', 16, 612, step=8),
                'dropout_rate': trial.suggest_float('dropout_rate', 0.1, 0.5, step=0.001),
                'learning_rate': trial.suggest_float('lstm_learning_rate', 0.0001, 0.1, log=True)
            }
            
            xgb_weight = trial.suggest_float('xgb_weight', 0.2, 0.6, step=0.01)
            hmm_weight = trial.suggest_float('hmm_weight', 0.1, 0.5, step=0.01)
            lstm_weight = 1.0 - xgb_weight - hmm_weight
            
            if lstm_weight < 0.1:
                lstm_weight = 0.1
                total = xgb_weight + hmm_weight + lstm_weight
                xgb_weight /= total
                hmm_weight /= total
                lstm_weight /= total
            
            ensemble_weights = [xgb_weight, hmm_weight, lstm_weight]
        else:
            lstm_params = None
            ensemble_weights = [0.6, 0.4, 0.0]
        
        barrier_multiplier = trial.suggest_float('barrier_multiplier', 0.5, 3.0, step=0.1)
        
        barrier_params = {
            'volatility_window': trial.suggest_int('volatility_window', 5, 100, step=3),
            'upper_barrier_multiplier': barrier_multiplier,
            'lower_barrier_multiplier': barrier_multiplier,
            'time_barrier_days': trial.suggest_int('time_barrier_days', 3, 35),
            'verbose': False
        }
        
        temp_triple_barrier_df = apply_triple_barrier_labeling(
            data=filtered_df,
            **barrier_params
        )
        
        temp_triple_barrier_df['decision_date'] = pd.to_datetime(temp_triple_barrier_df['decision_date'])
        temp_merged_df = temp_triple_barrier_df.merge(
            filtered_df, 
            left_on='decision_date', 
            right_on='date', 
            how='left'
        )
        
        if 'date' in temp_merged_df.columns:
            temp_merged_df = temp_merged_df.drop('date', axis=1)
        
        temp_columns_to_drop = [
            'decision_date', 'entry_date', 'end_date', 'end_price',
            'return', 'barrier_touched', 'value_at_barrier_touched',
            'time_converted', 'datetime', 'time', 'time_barrier'
        ]
        temp_existing_columns_to_drop = [col for col in temp_columns_to_drop if col in temp_merged_df.columns]
        
        if temp_existing_columns_to_drop:
            df_with_labels_new = temp_merged_df.drop(columns=temp_existing_columns_to_drop)
        else:
            df_with_labels_new = temp_merged_df.copy()
        
        df_with_labels_new = df_with_labels_new.dropna(subset=['label'])
        
        if len(df_with_labels_new) < 100:
            return 0.0
        
        merged_new_df = df_with_labels_new.copy()
        merged_new_df['decision_date'] = temp_triple_barrier_df['decision_date']
        
        q3_2024_start_new = pd.Timestamp('2024-07-01')
        
        test_mask_new = merged_new_df['decision_date'] >= q3_2024_start_new
        train_val_mask_new = ~test_mask_new
        
        train_val_data_new = merged_new_df[train_val_mask_new].copy()
        
        X_train_val_new = train_val_data_new.drop(['label', 'decision_date'], axis=1)
        y_train_val_new = train_val_data_new['label']
        
        X_train_new, X_val_new, y_train_new, y_val_new = train_test_split(
            X_train_val_new, y_train_val_new, test_size=0.20, random_state=42
        )
        
        unique_labels_new = sorted(y_train_new.unique())
        num_classes_new = len(unique_labels_new)
        
        if num_classes_new == 2:
            label_mapping_new = {unique_labels_new[0]: 0, unique_labels_new[1]: 1}
        else:
            label_mapping_new = {label: idx for idx, label in enumerate(unique_labels_new)}
        
        y_train_mapped_new = y_train_new.map(label_mapping_new)
        y_val_mapped_new = y_val_new.map(label_mapping_new)
        
        suggested_n_states = hmm_params['n_states']
        actual_n_states = max(num_classes_new, suggested_n_states)
        
        temp_xgb_hmm_model = XGBHMMModel(
            n_states=actual_n_states,
            max_iter=hmm_params['max_iter'],
            tol=hmm_params['tol'],
            random_state=42
        )
        
        xgb_params['verbosity'] = 0
        temp_xgb_hmm_model.xgb_params = xgb_params
        
        temp_xgb_hmm_model.fit(X_train_new, y_train_mapped_new)
        
        y_val_pred_new = temp_xgb_hmm_model.predict(X_val_new)
        val_accuracy_new = accuracy_score(y_val_mapped_new, y_val_pred_new)
        
        return val_accuracy_new
        
    except Exception as e:
        logging.warning(f"Trial failed with error: {str(e)}")
        return 0.0

logging.info("Combined HMM, XGBoost, and LSTM Parameters Tuning with Optuna")
logging.info("-" * 50)

if TENSORFLOW_AVAILABLE:
    logging.info("TensorFlow available - using XGB-HMM-LSTM hybrid optimization")
    objective_function = combined_xgb_hmm_lstm_objective
    study_name = "combined_xgb_hmm_lstm_barrier_parameters"
else:
    logging.info("TensorFlow not available - using XGB-HMM optimization only")
    objective_function = combined_xgb_hmm_objective
    study_name = "combined_xgb_hmm_barrier_parameters"

logging.info(f"Starting {study_name} optimization with Optuna...")
start_time = time.time()

optuna.logging.set_verbosity(optuna.logging.WARNING)

combined_study = create_study_with_storage(
    study_name=study_name,
    direction='maximize'
)

combined_callback = EnhancedProgressCallback(
    study_name="Combined Parameters Optimization", 
    patience=15, 
    min_improvement=0.005,
    max_trial_time=600
)

combined_study.optimize(
    objective_function, 
    n_trials=10,
    callbacks=[combined_callback],
    show_progress_bar=True
)

combined_tuning_time = time.time() - start_time

best_combined_trial = combined_study.best_trial
best_combined_score = best_combined_trial.value

best_hmm_params_combined = {
    'n_states': best_combined_trial.params['n_states'],
    'max_iter': best_combined_trial.params['max_iter'],
    'tol': best_combined_trial.params['tol']
}

best_xgb_params_combined = {
    'n_estimators': best_combined_trial.params['n_estimators'],
    'max_depth': best_combined_trial.params['max_depth'],
    'learning_rate': best_combined_trial.params['learning_rate'],
    'subsample': best_combined_trial.params['subsample'],
    'colsample_bytree': best_combined_trial.params['colsample_bytree'],
    'reg_alpha': best_combined_trial.params['reg_alpha'],
    'reg_lambda': best_combined_trial.params['reg_lambda'],
    'min_child_weight': best_combined_trial.params['min_child_weight'],
    'gamma': best_combined_trial.params['gamma']
}

if TENSORFLOW_AVAILABLE and 'sequence_length' in best_combined_trial.params:
    best_lstm_params_combined = {
        'sequence_length': best_combined_trial.params['sequence_length'],
        'lstm_units': best_combined_trial.params['lstm_units'],
        'dropout_rate': best_combined_trial.params['dropout_rate'],
        'learning_rate': best_combined_trial.params['lstm_learning_rate']
    }
    
    best_ensemble_weights_combined = [
        best_combined_trial.params['xgb_weight'],
        best_combined_trial.params['hmm_weight'],
        1.0 - best_combined_trial.params['xgb_weight'] - best_combined_trial.params['hmm_weight']
    ]
else:
    best_lstm_params_combined = None
    best_ensemble_weights_combined = [0.6, 0.4, 0.0]

best_barrier_params_combined = {
    'volatility_window': best_combined_trial.params['volatility_window'],
    'upper_barrier_multiplier': best_combined_trial.params['barrier_multiplier'],
    'lower_barrier_multiplier': best_combined_trial.params['barrier_multiplier'],
    'time_barrier_days': best_combined_trial.params['time_barrier_days']
}

logging.info(f"\nCombined hyperparameter tuning completed in {combined_tuning_time:.2f} seconds")
logging.info(f"Best combined validation accuracy: {best_combined_score:.4f}")
logging.info(f"Best trial number: {best_combined_trial.number}")

logging.info(f"\nBest HMM Parameters:")
for param, value in best_hmm_params_combined.items():
    logging.info(f"  {param}: {value}")

logging.info(f"\nBest XGBoost Parameters:")
for param, value in best_xgb_params_combined.items():
    logging.info(f"  {param}: {value}")

if best_lstm_params_combined:
    logging.info(f"\nBest LSTM Parameters:")
    for param, value in best_lstm_params_combined.items():
        logging.info(f"  {param}: {value}")
    logging.info(f"\nBest Ensemble Weights: XGB={best_ensemble_weights_combined[0]:.2f}, HMM={best_ensemble_weights_combined[1]:.2f}, LSTM={best_ensemble_weights_combined[2]:.2f}")

logging.info(f"\nBest Barrier Parameters:")
for param, value in best_barrier_params_combined.items():
    logging.info(f"  {param}: {value}")

optimization_summary = {
    'study_name': study_name,
    'tensorflow_available': TENSORFLOW_AVAILABLE,
    'best_score': best_combined_score,
    'best_trial_number': best_combined_trial.number,
    'total_trials': len(combined_study.trials),
    'tuning_time_seconds': combined_tuning_time,
    'best_hmm_params': best_hmm_params_combined,
    'best_xgb_params': best_xgb_params_combined,
    'best_lstm_params': best_lstm_params_combined,
    'best_ensemble_weights': best_ensemble_weights_combined,
    'best_barrier_params': best_barrier_params_combined
}

optimization_summary_path = 'logs/xgb_results/combined_optimization_summary.pkl'
joblib.dump(optimization_summary, optimization_summary_path)
logging.info(f"Optimization summary saved to: {optimization_summary_path}")

logging.info(f"\nUsing best parameters from combined tuning for final model training...")
final_hmm_params = best_hmm_params_combined
final_xgb_params = best_xgb_params_combined
final_lstm_params = best_lstm_params_combined
final_ensemble_weights = best_ensemble_weights_combined
final_barrier_params = best_barrier_params_combined

logging.info(f"Applying triple barrier labeling with optimized parameters...")
final_triple_barrier_df = apply_triple_barrier_labeling(
    data=filtered_df,
    **final_barrier_params,
    verbose=True
)

final_triple_barrier_df['decision_date'] = pd.to_datetime(final_triple_barrier_df['decision_date'])
final_merged_df = final_triple_barrier_df.merge(
    filtered_df, 
    left_on='decision_date', 
    right_on='date', 
    how='left'
)

if 'date' in final_merged_df.columns:
    final_merged_df = final_merged_df.drop('date', axis=1)

merged_label_df = final_merged_df.copy()
logging.info(f"Updated merged data with optimized barrier parameters: {merged_label_df.shape}")

logging.info("\n9. Training XGB-HMM Hybrid Model")
logging.info("-" * 50)

logging.info("Using existing triple barrier data for XGB-HMM training...")
logging.info(f"Triple barrier data already available: {len(triple_barrier_df)} samples")

logging.info("Proceeding directly to XGB-HMM training with existing merged data...")

df_with_labels_final = merged_label_df.copy()

df_with_labels_final = df_with_labels_final.dropna(subset=['label'])

logging.info(f"Final dataset shape: {df_with_labels_final.shape}")
logging.info(f"Label distribution:")
logging.info(df_with_labels_final['label'].value_counts().sort_index())

df_with_labels_final['decision_date'] = pd.to_datetime(df_with_labels_final['decision_date'])

q3_2024_start_final = pd.Timestamp('2024-07-01')

test_mask_final = df_with_labels_final['decision_date'] >= q3_2024_start_final
train_val_mask_final = ~test_mask_final

test_data_final = df_with_labels_final[test_mask_final].copy()
logging.info(f"Final test data period: {test_data_final['decision_date'].min()} to {test_data_final['decision_date'].max()}")
logging.info(f"Final test data shape: {test_data_final.shape}")

train_val_data_final = df_with_labels_final[train_val_mask_final].copy()
logging.info(f"Final train+val data period: {train_val_data_final['decision_date'].min()} to {train_val_data_final['decision_date'].max()}")
logging.info(f"Final train+val data shape: {train_val_data_final.shape}")

final_columns_to_drop_clean = [
    'decision_date', 'entry_date', 'end_date', 'end_price',
    'return', 'barrier_touched', 'value_at_barrier_touched',
    'time_converted', 'datetime', 'time', 'time_barrier'
]

existing_cols_test_final = [col for col in final_columns_to_drop_clean if col in test_data_final.columns]
if existing_cols_test_final:
    test_data_clean_final = test_data_final.drop(columns=existing_cols_test_final)
else:
    test_data_clean_final = test_data_final.copy()

existing_cols_train_val_final = [col for col in final_columns_to_drop_clean if col in train_val_data_final.columns]
if existing_cols_train_val_final:
    train_val_data_clean_final = train_val_data_final.drop(columns=existing_cols_train_val_final)
else:
    train_val_data_clean_final = train_val_data_final.copy()

X_test_final = test_data_clean_final.drop('label', axis=1)
y_test_final = test_data_clean_final['label']

X_train_val_final = train_val_data_clean_final.drop('label', axis=1)
y_train_val_final = train_val_data_clean_final['label']

X_train_final, X_val_final, y_train_final, y_val_final = train_test_split(
    X_train_val_final, y_train_val_final, test_size=0.20, random_state=42
)

logging.info(f"\nFinal data splits:")
logging.info(f"Training set: {X_train_final.shape[0]} samples")
logging.info(f"Validation set: {X_val_final.shape[0]} samples")
logging.info(f"Test set: {X_test_final.shape[0]} samples")

unique_labels_final = sorted(y_train_final.unique())
num_classes_final = len(unique_labels_final)

if num_classes_final == 2:
    label_mapping_final = {unique_labels_final[0]: 0, unique_labels_final[1]: 1}
else:
    label_mapping_final = {label: idx for idx, label in enumerate(unique_labels_final)}

y_train_mapped_final = y_train_final.map(label_mapping_final)
y_val_mapped_final = y_val_final.map(label_mapping_final)
y_test_mapped_final = y_test_final.map(label_mapping_final)

logging.info(f"\nFinal label mapping: {label_mapping_final}")
logging.info(f"Number of classes: {num_classes_final}")

logging.info("\n=== Applying Feature Selection ===")
logging.info("Initializing feature selection process...")

feature_selector = FeatureSelector(random_state=42)

original_feature_names = X_train_final.columns.tolist()
logging.info(f"Original number of features: {len(original_feature_names)}")

logging.info("Applying ensemble feature selection...")
X_train_selected_fs, selected_features, feature_votes = feature_selector.ensemble_selection(
    X_train_final, y_train_mapped_final,
    methods=['univariate', 'mutual_info', 'rfe', 'feature_importance'],
    k_univariate=min(100, len(original_feature_names) // 2),
    k_mutual_info=min(100, len(original_feature_names) // 2),
    n_rfe=min(100, len(original_feature_names) // 2),
    n_importance=min(100, len(original_feature_names) // 2),
    min_votes=2
)

selected_feature_names = [original_feature_names[i] for i in selected_features]

logging.info(f"Selected {len(selected_features)} features out of {len(original_feature_names)}")
logging.info(f"Feature reduction: {((len(original_feature_names) - len(selected_features)) / len(original_feature_names) * 100):.2f}%")

feature_summary = feature_selector.get_feature_selection_summary(original_feature_names)
logging.info("\nFeature Selection Summary:")
for method, info in feature_summary.items():
    if isinstance(info, dict) and 'selected_count' in info:
        logging.info(f"  {method}: {info['selected_count']} features selected")
        if 'scores' in info and len(info['scores']) > 0:
            logging.info(f"    Top 5 scores: {sorted(info['scores'], reverse=True)[:5]}")

logging.info("Applying feature selection to training, validation, and test sets...")
X_train_selected = X_train_final.iloc[:, selected_features]
X_val_selected = X_val_final.iloc[:, selected_features]
X_test_selected = X_test_final.iloc[:, selected_features]

logging.info(f"Selected feature names (top 10): {selected_feature_names[:10]}")
if len(selected_feature_names) > 10:
    logging.info(f"... and {len(selected_feature_names) - 10} more features")

feature_selection_summary = {
    'original_features': original_feature_names,
    'selected_features': selected_feature_names,
    'selected_indices': selected_features,
    'feature_summary': feature_summary
}

feature_selection_path = 'logs/xgb_results/feature_selection_summary.pkl'
joblib.dump(feature_selection_summary, feature_selection_path)
logging.info(f"Feature selection summary saved to: {feature_selection_path}")

logging.info("\n=== Training XGB-HMM-LSTM Triple Hybrid Model with Selected Features ===")
logging.info("Initializing XGB-HMM-LSTM model with optimal parameters and selected features...")

if TENSORFLOW_AVAILABLE and final_lstm_params is not None:
    xgb_hmm_lstm_model = XGBHMMLSTMModel(
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
    
    xgb_hmm_lstm_model.xgb_params = final_xgb_params
    
    logging.info(f"XGB-HMM-LSTM Model initialized with optimized parameters")
    logging.info(f"  n_states: {final_hmm_params['n_states']}")
    logging.info(f"  max_iter: {final_hmm_params['max_iter']}")
    logging.info(f"  tol: {final_hmm_params['tol']}")
    logging.info(f"  sequence_length: {final_lstm_params['sequence_length']}")
    logging.info(f"  lstm_units: {final_lstm_params['lstm_units']}")
    logging.info(f"  ensemble_weights: {final_ensemble_weights}")
    logging.info(f"  TensorFlow available: {TENSORFLOW_AVAILABLE}")
    
    logging.info("Starting XGB-HMM-LSTM training with EM algorithm...")
    start_time = time.time()
    
    xgb_hmm_lstm_model.fit(X_train_selected, y_train_mapped_final)
    
    training_time = time.time() - start_time
    logging.info(f"XGB-HMM-LSTM training completed in {training_time:.2f} seconds")
    
    model_to_evaluate = xgb_hmm_lstm_model
    model_name = "XGB-HMM-LSTM"
    
    if hasattr(model_to_evaluate, 'lstm_model') and model_to_evaluate.lstm_model is not None:
        logging.info(f"LSTM model successfully trained and integrated")
        logging.info(f"Ensemble weights used: XGB={final_ensemble_weights[0]:.2f}, HMM={final_ensemble_weights[1]:.2f}, LSTM={final_ensemble_weights[2]:.2f}")
    else:
        logging.warning(f"LSTM model not available, using XGB-HMM only")
    
else:
    logging.warning("Using XGB-HMM model instead of LSTM hybrid.")
    logging.info(f"Reason: TensorFlow available = {TENSORFLOW_AVAILABLE}, LSTM params = {final_lstm_params is not None}")
    
    xgb_hmm_model = XGBHMMModel(
        n_states=final_hmm_params['n_states'],
        max_iter=final_hmm_params['max_iter'],
        tol=final_hmm_params['tol'],
        random_state=42
    )
    
    xgb_hmm_model.xgb_params = final_xgb_params
    
    logging.info(f"XGB-HMM Model initialized with optimized parameters:")
    logging.info(f"  n_states: {final_hmm_params['n_states']}")
    logging.info(f"  max_iter: {final_hmm_params['max_iter']}")
    logging.info(f"  tol: {final_hmm_params['tol']}")
    logging.info(f"XGBoost parameters: {final_xgb_params}")
    
    logging.info("Starting XGB-HMM training with EM algorithm...")
    start_time = time.time()
    
    xgb_hmm_model.fit(X_train_selected, y_train_mapped_final)
    
    training_time = time.time() - start_time
    logging.info(f"XGB-HMM training completed in {training_time:.2f} seconds")
    
    model_to_evaluate = xgb_hmm_model
    model_name = "XGB-HMM"

logging.info(f"\n=== {model_name} Model Detailed Evaluation ===")

train_evaluation = model_to_evaluate.evaluate_model(X_train_selected, y_train_mapped_final)
logging.info(f"\nTraining Set {model_name} Evaluation:")
for metric, value in train_evaluation.items():
    logging.info(f"  {metric}: {value:.4f}")

val_evaluation = model_to_evaluate.evaluate_model(X_val_selected, y_val_mapped_final)
logging.info(f"\nValidation Set {model_name} Evaluation:")
for metric, value in val_evaluation.items():
    logging.info(f"  {metric}: {value:.4f}")

test_evaluation = model_to_evaluate.evaluate_model(X_test_selected, y_test_mapped_final)
logging.info(f"\nTest Set {model_name} Evaluation:")
for metric, value in test_evaluation.items():
    logging.info(f"  {metric}: {value:.4f}")

logging.info("\n=== State Transition Analysis ===")
transition_analysis = model_to_evaluate.get_state_transition_analysis(X_test_selected)
logging.info(f"\nLearned Transition Matrix:")
logging.info(transition_analysis['learned_transition_matrix'])
logging.info(f"\nEmpirical Transition Matrix:")
logging.info(transition_analysis['empirical_transition_matrix'])

logging.info(f"\nEvaluating {model_name} model on all datasets...")

y_train_pred_final = model_to_evaluate.predict(X_train_selected)
y_train_proba_final = model_to_evaluate.predict_proba(X_train_selected)

train_accuracy_final = accuracy_score(y_train_mapped_final, y_train_pred_final)
train_precision_final = precision_score(y_train_mapped_final, y_train_pred_final, average='weighted')
train_recall_final = recall_score(y_train_mapped_final, y_train_pred_final, average='weighted')
train_f1_final = f1_score(y_train_mapped_final, y_train_pred_final, average='weighted')

if num_classes_final == 2:
    train_roc_auc_final = roc_auc_score(y_train_mapped_final, y_train_proba_final[:, 1])
else:
    train_roc_auc_final = roc_auc_score(y_train_mapped_final, y_train_proba_final, multi_class='ovr')

y_val_pred_final = model_to_evaluate.predict(X_val_selected)
y_val_proba_final = model_to_evaluate.predict_proba(X_val_selected)

val_accuracy_final = accuracy_score(y_val_mapped_final, y_val_pred_final)
val_precision_final = precision_score(y_val_mapped_final, y_val_pred_final, average='weighted')
val_recall_final = recall_score(y_val_mapped_final, y_val_pred_final, average='weighted')
val_f1_final = f1_score(y_val_mapped_final, y_val_pred_final, average='weighted')

if num_classes_final == 2:
    val_roc_auc_final = roc_auc_score(y_val_mapped_final, y_val_proba_final[:, 1])
else:
    val_roc_auc_final = roc_auc_score(y_val_mapped_final, y_val_proba_final, multi_class='ovr')

y_test_pred_final = model_to_evaluate.predict(X_test_selected)
y_test_proba_final = model_to_evaluate.predict_proba(X_test_selected)

test_accuracy_final = accuracy_score(y_test_mapped_final, y_test_pred_final)
test_precision_final = precision_score(y_test_mapped_final, y_test_pred_final, average='weighted')
test_recall_final = recall_score(y_test_mapped_final, y_test_pred_final, average='weighted')
test_f1_final = f1_score(y_test_mapped_final, y_test_pred_final, average='weighted')

unique_test_classes = len(np.unique(y_test_mapped_final))
if unique_test_classes == 2 and y_test_proba_final.shape[1] == 2:
    test_roc_auc_final = roc_auc_score(y_test_mapped_final, y_test_proba_final[:, 1])
elif unique_test_classes > 2 and y_test_proba_final.shape[1] == unique_test_classes:
    test_roc_auc_final = roc_auc_score(y_test_mapped_final, y_test_proba_final, multi_class='ovr')
else:
    test_roc_auc_final = test_accuracy_final
    logging.warning(f"ROC AUC calculation skipped due to class mismatch. Test classes: {unique_test_classes}, Proba shape: {y_test_proba_final.shape}")

logging.info("\n" + "="*60)
logging.info(f"{model_name} HYBRID MODEL PERFORMANCE")
logging.info("="*60)

logging.info(f"\nTraining Set Performance:")
logging.info(f"Accuracy: {train_accuracy_final:.4f}")
logging.info(f"Precision: {train_precision_final:.4f}")
logging.info(f"Recall: {train_recall_final:.4f}")
logging.info(f"F1-Score: {train_f1_final:.4f}")
logging.info(f"ROC AUC: {train_roc_auc_final:.4f}")

logging.info(f"\nValidation Set Performance:")
logging.info(f"Accuracy: {val_accuracy_final:.4f}")
logging.info(f"Precision: {val_precision_final:.4f}")
logging.info(f"Recall: {val_recall_final:.4f}")
logging.info(f"F1-Score: {val_f1_final:.4f}")
logging.info(f"ROC AUC: {val_roc_auc_final:.4f}")

logging.info(f"\nTest Set Performance:")
logging.info(f"Accuracy: {test_accuracy_final:.4f}")
logging.info(f"Precision: {test_precision_final:.4f}")
logging.info(f"Recall: {test_recall_final:.4f}")
logging.info(f"F1-Score: {test_f1_final:.4f}")
logging.info(f"ROC AUC: {test_roc_auc_final:.4f}")

logging.info(f"\nTest Set Confusion Matrix:")
cm_test_final = confusion_matrix(y_test_mapped_final, y_test_pred_final)
logging.info(cm_test_final)

logging.info(f"\nTest Set Classification Report:")
reverse_label_mapping_final = {v: k for k, v in label_mapping_final.items()}
unique_test_classes_actual = sorted(np.unique(y_test_mapped_final))
unique_test_classes_predicted = sorted(np.unique(y_test_pred_final))
all_classes = sorted(set(unique_test_classes_actual) | set(unique_test_classes_predicted))
target_names_final = [str(reverse_label_mapping_final.get(i, f'Class_{i}')) for i in all_classes]

logging.info(f"Classes in test data: {unique_test_classes_actual}")
logging.info(f"Classes predicted by model: {unique_test_classes_predicted}")
logging.info(f"All classes for classification report: {all_classes}")
logging.info(classification_report(y_test_mapped_final, y_test_pred_final, target_names=target_names_final, labels=all_classes))

logging.info(f"\n" + "="*60)
logging.info("PERFORMANCE COMPARISON")
logging.info("="*60)

performance_comparison_final = pd.DataFrame({
    'Dataset': ['Training', 'Validation', 'Test'],
    'Accuracy': [train_accuracy_final, val_accuracy_final, test_accuracy_final],
    'Precision': [train_precision_final, val_precision_final, test_precision_final],
    'Recall': [train_recall_final, val_recall_final, test_recall_final],
    'F1-Score': [train_f1_final, val_f1_final, test_f1_final],
    'ROC AUC': [train_roc_auc_final, val_roc_auc_final, test_roc_auc_final]
})

logging.info(performance_comparison_final.round(4))

final_results_path = 'logs/xgb_results/final_performance_comparison.csv'
performance_comparison_final.to_csv(final_results_path, index=False)
logging.info(f"\nFinal performance comparison saved to: {final_results_path}")

final_model_path = f'logs/xgb_models/{model_name.lower().replace("-", "_")}_model.pkl'
joblib.dump(model_to_evaluate, final_model_path)
logging.info(f"{model_name} model saved to: {final_model_path}")

final_test_predictions_dict = {
    'actual_label': y_test_final.values,
    'predicted_label': [reverse_label_mapping_final[pred] for pred in y_test_pred_final],
    'actual_mapped': y_test_mapped_final.values,
    'predicted_mapped': y_test_pred_final
}

if num_classes_final == 2:
    final_test_predictions_dict['probability_class_0'] = y_test_proba_final[:, 0]
    final_test_predictions_dict['probability_class_1'] = y_test_proba_final[:, 1]
else:
    for i in range(num_classes_final):
        final_test_predictions_dict[f'probability_class_{i}'] = y_test_proba_final[:, i]

final_test_predictions_df = pd.DataFrame(final_test_predictions_dict)

final_test_predictions_path = 'logs/xgb_results/final_test_predictions.csv'
final_test_predictions_df.to_csv(final_test_predictions_path, index=False)
logging.info(f"Final test predictions saved to: {final_test_predictions_path}")

logging.info(f"\n" + "="*60)
logging.info(f"{model_name} HYPERPARAMETER SUMMARY")
logging.info("="*60)

logging.info(f"\n{model_name} Model Parameters:")
logging.info(f"  n_states: {model_to_evaluate.n_states}")
logging.info(f"  max_iter: {model_to_evaluate.max_iter}")
logging.info(f"  tolerance: {model_to_evaluate.tol}")
logging.info(f"  random_state: {model_to_evaluate.random_state}")

logging.info(f"\n{model_name} Training Summary:")
logging.info(f"  Training time: {training_time:.2f} seconds")
logging.info(f"  Convergence achieved: {hasattr(model_to_evaluate, 'converged_') and model_to_evaluate.converged_}")
logging.info(f"  Final log-likelihood: {model_to_evaluate.log_likelihood_history_[-1] if hasattr(model_to_evaluate, 'log_likelihood_history_') else 'N/A'}")

logging.info(f"\nHybrid XGB-HMM vs Original XGBoost Performance:")
logging.info(f"Original XGBoost Test Accuracy: ~0.45 (from previous runs)")
logging.info(f"Hybrid XGB-HMM Test Accuracy: {test_accuracy_final:.4f}")
logging.info(f"Performance Improvement: {((test_accuracy_final - 0.45) / 0.45 * 100):.2f}%")

logging.info(f"\nHybrid XGB-HMM Model Generalization:")
logging.info(f"Training Accuracy: {train_accuracy_final:.4f}")
logging.info(f"Validation Accuracy: {val_accuracy_final:.4f}")
logging.info(f"Test Accuracy: {test_accuracy_final:.4f}")
logging.info(f"Train-Val Gap: {train_accuracy_final - val_accuracy_final:.4f}")
logging.info(f"Train-Test Gap: {train_accuracy_final - test_accuracy_final:.4f}")

logging.info(f"\n=== Feature Selection Impact Analysis ===")
logging.info(f"Original features: {len(original_feature_names)}")
logging.info(f"Selected features: {len(selected_features)}")
logging.info(f"Feature reduction: {((len(original_feature_names) - len(selected_features)) / len(original_feature_names) * 100):.2f}%")
logging.info(f"Model performance maintained with {(len(selected_features) / len(original_feature_names) * 100):.2f}% of original features")

if 'feature_importance' in feature_summary and 'scores' in feature_summary['feature_importance']:
    importance_scores = feature_summary['feature_importance']['scores']
    if len(importance_scores) > 0:
        top_indices = sorted(range(len(importance_scores)), key=lambda i: importance_scores[i], reverse=True)[:10]
        top_features = [selected_feature_names[i] for i in top_indices if i < len(selected_feature_names)]
        logging.info(f"\nTop 10 Selected Features by Importance:")
        for i, feature in enumerate(top_features, 1):
            if i-1 < len(importance_scores):
                logging.info(f"  {i}. {feature}: {importance_scores[top_indices[i-1]]:.4f}")

logging.info("\n" + "="*80)
logging.info(f"{model_name} HYBRID MODEL WITH FEATURE SELECTION COMPLETED SUCCESSFULLY!")
logging.info("="*80)