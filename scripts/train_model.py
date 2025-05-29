#!/usr/bin/env python3
"""
train_model.py

Loads processed data with features and labels (-1, 0, 1); trains a model using
the configuration from config.params.MODEL_CONFIG; evaluates the model;
and saves the trained model and metadata using DataManager.

Supports RandomForest, XGBoost, and LSTM models for TERNARY classification.

Uses the updated configuration structure from config.params.MODEL_CONFIG,
config.params.GENERAL_CONFIG, and config.paths.PATHS.
Configures logging using utils/logger_config.py.
Includes optional hyperparameter tuning using RandomizedSearchCV and TimeSeriesSplit.
Handles train, validation, and test data splitting.
Removes rows with NaN values in features or labels before splitting.

MODIFIED: Added functionality to specify a subset of features to use for training
          via a command-line argument.
MODIFIED: Updated DataManager calls for loading data and saving model artifacts.
MODIFIED: Adjusted labeled data loading to no longer require a strategy-specific suffix.
"""

import sys
import logging
import argparse
from pathlib import Path
from typing import Tuple, List, Dict, Any, Optional
from collections import Counter # Import Counter for imblearn setup in tuning
import time # Import time for measuring training duration
import copy # Import copy for deepcopying config

import pandas as pd
import numpy as np

# Import scikit-learn and imblearn components for tuning
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler

# Import parameter distributions for RandomizedSearchCV
from scipy.stats import uniform, randint
from xgboost import XGBClassifier


# Add project root to Python path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Import configuration and utilities
try:
    from config.paths import PATHS
    from config.params import MODEL_CONFIG, GENERAL_CONFIG # Import GENERAL_CONFIG
    from utils.data_manager import DataManager # Import DataManager
    from utils.model_trainer import ModelTrainer # Import ModelTrainer
    # from utils.label_generator import LabelGenerator # No longer needed for choices here
    # Import setup_rotating_logging
    from utils.logger_config import setup_rotating_logging
except ImportError as e:
    # Use print for initial errors before logging is fully configured
    print(f"ERROR: Failed to import necessary modules. Ensure config/, utils/ are correctly structured and required files exist. Error: {e}", file=sys.stderr)
    sys.exit(1) # Exit if essential imports fail
except FileNotFoundError as e:
    print(f"ERROR: Configuration file not found: {e}. Ensure config/params.py and config/paths.py exist.", file=sys.stderr)
    sys.exit(1)
except AttributeError as e:
     print(f"ERROR: Configuration object missing expected attribute or key: {e}. Check config/params.py and config/paths.py.", file=sys.stderr)
     sys.exit(1)
except Exception as e:
    print(f"ERROR: An unexpected error occurred during initial imports or configuration loading: {e}", file=sys.stderr)
    sys.exit(1)


# --- Conditional Import for TensorFlow/Keras ---
# This allows the script to run even if TensorFlow is not installed,
# but LSTM functionality will be disabled.
try:
    import tensorflow as tf
    tf_version = getattr(tf, '__version__', 'unknown')
    LSTM_AVAILABLE = True
except ImportError:
    tf = None
    LSTM_AVAILABLE = False
except Exception as e:
    # Catch other potential errors during TF import (e.g., DLL issues)
    tf = None
    LSTM_AVAILABLE = False


# --- Configure Rotating Logging ---
# Set up rotating logging for this script
# Call setup_rotating_logging with filename base and level
try:
    setup_rotating_logging('train_model', logging.INFO)
    # Get logger for this script (after setup_rotating_logging has been called)
    # This ensures the logger uses the configured handlers
    logger = logging.getLogger(__name__)
    logger.info("Logging configured successfully in train_model.py using setup_rotating_logging.")

    # --- Log TensorFlow availability here, after logger is configured ---
    if LSTM_AVAILABLE:
        logger.info(f"TensorFlow (version {tf_version}) imported successfully.")
        if tf.config.list_physical_devices('GPU'):
            logger.info("GPU is available and enabled for TensorFlow.")
        else:
            logger.info("GPU is not available or not enabled for TensorFlow.")
    else:
        logger.warning("TensorFlow not found. LSTM model type will not be available.")


except ImportError:
    # Fallback basic logging if setup_rotating_logging is not available
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler(sys.stdout)]
    )
    logger = logging.getLogger(__name__)
    logger.warning("utils.logger_config not found or setup_rotating_logging failed. Using basic logging configuration.")
except Exception as e:
    # Fallback basic logging if setup_rotating_logging fails
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler(sys.stdout)]
    )
    logger = logging.getLogger(__name__)
    logger.error(f"Failed to configure logging using utils.logger_config: {e}. Using basic logging.", exc_info=True)


def run_tuning(model_key: str, X_train: pd.DataFrame, y_train: pd.Series, model_config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Performs hyperparameter tuning for the specified model using RandomizedSearchCV
    with TimeSeriesSplit.

    Args:
        model_key (str): The key for the model type ('random_forest', 'xgboost').
        X_train (pd.DataFrame): Training features (already cleaned of NaNs).
        y_train (pd.Series): Training labels (-1, 0, 1) (already cleaned of NaNs).
        model_config (Dict[str, Any]): The specific model configuration dictionary
                                       from MODEL_CONFIG.

    Returns:
        Dict[str, Any]: The best parameters found by tuning, merged with base parameters.

    Raises:
        ValueError: If model_key is 'lstm' (tuning not implemented here) or unsupported,
                    or if tuning parameters are missing.
        RuntimeError: If tuning fails.
    """
    logger.info(f"Starting hyperparameter tuning for {model_key}...")

    if model_key == 'lstm':
        logger.warning("Hyperparameter tuning for LSTM models is not implemented in this script.")
        logger.warning("Using default parameters from config for LSTM.")
        # Return the original model parameters for LSTM
        return model_config.get('params', {})


    # Get base model parameters and tuning distributions from the model_config
    base_model_params = model_config.get('params', {}).copy()
    param_dist = model_config.get('tuning_param_dist', {})

    # Ensure param_dist is not empty before proceeding with tuning
    if not param_dist:
         logger.warning(f"No tuning parameter distributions found for {model_key} in config. Skipping tuning and using default parameters.")
         # Return default parameters if no tuning distributions are defined
         return base_model_params


    logger.info(f"Tuning parameter distributions: {param_dist}")

    # Create the model instance directly for the tuning pipeline
    try:
        if model_key == 'random_forest':
             # Remove class_balancing and undersample_ratio from params before passing to RF
             rf_params_for_tuning = {k: v for k, v in base_model_params.items() if k not in ['class_balancing', 'undersample_ratio']}
             # Add class_weight parameter if specified in config
             class_weight_param = base_model_params.get('class_weight')
             if class_weight_param is not None:
                  rf_params_for_tuning['class_weight'] = class_weight_param

             model = RandomForestClassifier(random_state=GENERAL_CONFIG.get('random_seed', 42), n_jobs=GENERAL_CONFIG.get('parallel_jobs', -1), **rf_params_for_tuning)
        elif model_key == 'xgboost':
             # Remove class_balancing and undersample_ratio from params before passing to XGBoost
             xgb_params_for_tuning = {k: v for k, v in base_model_params.items() if k not in ['class_balancing', 'undersample_ratio']}
             # XGBoost specific setup for ternary classification
             xgb_params_for_tuning.update({
                 'objective': 'multi:softmax',
                 'num_class': 3,
                 'eval_metric': 'mlogloss',
                 # Removed 'use_label_encoder': False as it's deprecated/unused in recent XGBoost
                 'random_state': GENERAL_CONFIG.get('random_seed', 42),
                 'n_jobs': GENERAL_CONFIG.get('parallel_jobs', -1),
             })
             model = XGBClassifier(**xgb_params_for_tuning)
        else:
             raise ValueError(f"Model type '{model_key}' is not supported for tuning.")

        logger.debug(f"Created base model instance for tuning: {type(model).__name__}")
    except Exception as e:
        logger.error(f"Failed to create base model instance for tuning: {e}", exc_info=True)
        raise RuntimeError("Failed to create base model for tuning.")


    # Create the preprocessing step using a dummy ModelTrainer instance
    # Pass the specific model configuration to the dummy trainer
    # Pass the feature_subset to the dummy trainer's preprocessor creation
    feature_subset_for_tuning = model_config.get('features_to_use') # Get the feature subset from config
    dummy_trainer_for_preprocessor = ModelTrainer(config=model_config) # Pass the full config
    preprocessor = dummy_trainer_for_preprocessor._create_preprocessor(X_train, feature_subset=feature_subset_for_tuning)


    # Create the imblearn pipeline steps for tuning
    steps = [('preprocessor', preprocessor)]

    # Add sampler step if defined in model_config for tuning
    balanced_strategy_tuning = model_config.get('params', {}).get('class_balancing') # Use 'class_balancing' key from params
    if balanced_strategy_tuning == 'undersampling':
         # Use RandomUnderSampler with default strategy (balances to minority class size)
         sampler = RandomUnderSampler(random_state=GENERAL_CONFIG.get('random_seed', 42))
         steps.append(('sampler', sampler))
         logger.info("Added RandomUnderSampler to tuning pipeline.")

    elif balanced_strategy_tuning == 'oversampling':
         # Use SMOTE with default strategy (oversamples minority class(es) to equal majority class size)
         sampler = SMOTE(random_state=GENERAL_CONFIG.get('random_seed', 42))
         steps.append(('sampler', sampler))
         logger.info("Added SMOTE to tuning pipeline.")

    elif balanced_strategy_tuning is not None:
         logger.warning(f"Unsupported 'class_balancing' strategy '{balanced_strategy_tuning}' for tuning. Skipping sampler in tuning pipeline.")


    # Add the model step to the pipeline
    steps.append(('model', model))

    # Create the full scikit-learn/imblearn pipeline for tuning
    pipeline = Pipeline(steps)
    logger.debug(f"Tuning pipeline created with steps: {[name for name, _ in pipeline.steps]}")


    # Define the cross-validation strategy (TimeSeriesSplit)
    # Get n_splits from model-specific config or general config, default to 5
    n_splits = model_config.get('cv_n_splits', GENERAL_CONFIG.get('tscv_splits', 5))
    if not isinstance(n_splits, int) or n_splits <= 0:
         logger.warning(f"Invalid cv_n_splits ({n_splits}). Defaulting to 5.")
         n_splits = 5
    tscv = TimeSeriesSplit(n_splits=n_splits)
    logger.info(f"Using TimeSeriesSplit with {n_splits} splits for tuning.")


    # Define the scoring metric for tuning
    # Get scoring_metric from model-specific config or general config, default to 'f1_macro'
    scoring_metric = model_config.get('tuning_scoring_metric', GENERAL_CONFIG.get('tuning_scoring_metric', 'f1_macro'))
    logger.info(f"Using '{scoring_metric}' as the scoring metric for tuning.")


    # Perform RandomizedSearchCV
    # Get n_iter from model-specific config or general config, default to 10
    n_iter = model_config.get('random_search_n_iter', GENERAL_CONFIG.get('random_search_n_iter', 10))
    if not isinstance(n_iter, int) or n_iter <= 0:
         logger.warning(f"Invalid random_search_n_iter ({n_iter}). Defaulting to 10.")
         n_iter = 10

    # Get n_jobs for CV from general config, default to -1
    cv_jobs = GENERAL_CONFIG.get('cv_jobs', -1)


    logger.info(f"Running RandomizedSearchCV with {n_iter} iterations and {n_splits}-fold TimeSeriesSplit...")
    try:
        random_search = RandomizedSearchCV(
            pipeline,
            param_distributions=param_dist,
            n_iter=n_iter,
            cv=tscv,
            scoring=scoring_metric,
            random_state=GENERAL_CONFIG.get('random_seed', 42),
            n_jobs=cv_jobs,
            verbose=1 # Set verbose level to show progress
        )

        # Map y_train to integers (0, 1, 2) for models/samplers that require it
        # y_train is already cleaned of NaNs and is int type.
        # Corrected mapping: -1 -> 0, 0 -> 1, 1 -> 2
        y_train_mapped = y_train.map({-1: 0, 0: 1, 1: 2})


        # Fit RandomizedSearchCV on the training data
        random_search.fit(X_train, y_train_mapped)

        logger.info("RandomizedSearchCV complete.")
        logger.info(f"Best parameters found: {random_search.best_params_}")
        logger.info(f"Best cross-validation score ({scoring_metric}): {random_search.best_score_:.4f}")

        # Extract the best model parameters from the pipeline's best_params_
        # Keys are in the format 'stepname__parametername' (e.g., 'model__n_estimators')
        best_model_params = {k.replace('model__', ''): v for k, v in random_search.best_params_.items() if k.startswith('model__')}

        # Merge best model parameters with base parameters (overwriting defaults with tuned values)
        tuned_params = {**base_model_params, **best_model_params}
        logger.info(f"Extracted best model parameters: {best_model_params}")
        logger.info(f"Merged tuned parameters: {tuned_params}")

        return tuned_params

    except Exception as e:
        logger.error(f"An error occurred during hyperparameter tuning: {e}", exc_info=True)
        raise RuntimeError(f"Hyperparameter tuning failed: {e}")


# --- Data Loading and Splitting ---
def load_and_split_data(
    symbol: str,
    interval: str,
    # REMOVED: label_strategy: str, as it's no longer needed for loading the labeled file
    train_ratio: float,
    val_ratio: float = 0.1,
    features_to_use: Optional[List[str]] = None
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series, pd.DataFrame, pd.Series]:
    """
    Loads processed and labeled data (-1, 0, 1) using DataManager, aligns them,
    performs time-series split into training, validation, and test sets, validates
    feature data types, REMOVES ROWS WITH NA LABELS AND NA FEATURES.
    Optionally selects a subset of features if features_to_use is provided.

    Args:
        symbol (str): Trading pair symbol.
        interval (str): Time interval.
        train_ratio (float): The fraction of data to use for training (0.0 to 1.0).
        val_ratio (float): The fraction of data to use for validation (0.0 to 1.0).
        features_to_use (Optional[List[str]]): A list of feature column names to use.
                                               If None, all available features are used.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series, pd.DataFrame, pd.Series]:
        X_train, X_val, X_test, y_train, y_val, y_test DataFrames/Series (after removing NA labels and features),
        X_full_cleaned, y_full_cleaned (full data after removing NA labels and features for tuning).

    Raises:
        FileNotFoundError: If data files are not found (raised by DataManager).
        ValueError: If data is empty, splitting is not possible, ratios are invalid, or features
                    contain unexpected non-numeric values, or specified features are missing.
        TypeError: If data types or indices are incorrect after loading.
        RuntimeError: For other data processing issues.
    """
    logger.info(f"Loading data for {symbol.upper()} @ {interval}...")

    dm = DataManager()

    try:
        # --- Load processed data (features) using DataManager ---
        logger.info(f"Attempting to load processed data (features) for {symbol.upper()} {interval}")
        X = dm.load_data(
            symbol=symbol.upper(),
            interval=interval,
            data_type='processed'
        )
        logger.info(f"Successfully loaded features. Shape: {X.shape}")

        # --- Explicitly drop 'open_time' column if it exists ---
        if 'open_time' in X.columns:
            X = X.drop(columns=['open_time'])
            logger.info("Dropped 'open_time' column from features DataFrame.")


        # --- Load labeled data using DataManager ---
        # User requested to load labeled data WITHOUT strategy-specific suffix
        logger.info(f"Attempting to load labeled data for {symbol.upper()} {interval}")
        ydf = dm.load_data(
            symbol=symbol.upper(),
            interval=interval,
            data_type='labeled',
            # REMOVED: name_suffix=f'_{label_strategy}'
        )
        logger.info(f"Successfully loaded labels. Shape: {ydf.shape}")


    except FileNotFoundError:
        raise
    except Exception as e:
        logger.error(f"An unexpected error occurred during data loading using DataManager: {e}", exc_info=True)
        raise RuntimeError(f"Data loading failed: {e}")


    # --- Validate Feature Data Types ---
    logger.info("Validating feature data types...")
    non_numeric_issues = {}
    known_non_feature_cols = ['open', 'high', 'low', 'close', 'volume', 'vol_adj']
    feature_cols_to_validate = [col for col in X.columns if col not in known_non_feature_cols]

    for col in feature_cols_to_validate:
        numeric_col = pd.to_numeric(X[col], errors='coerce')
        problematic_mask = numeric_col.isna() & X[col].notna()
        if problematic_mask.any():
            problem_values = X.loc[problematic_mask, col].unique().tolist()[:10]
            non_numeric_issues[col] = problem_values
            logger.warning(f"Column '{col}' contains non-numeric values that could not be converted: {problem_values}")
        if np.isinf(numeric_col).any():
             inf_count = np.isinf(numeric_col).sum()
             logger.warning(f"Column '{col}' contains {inf_count} infinite values.")

    if non_numeric_issues:
        error_msg = f"Feature DataFrame contains unexpected non-numeric values in columns: {list(non_numeric_issues.keys())}. Examples: {non_numeric_issues}"
        logger.error(error_msg)
        raise ValueError(error_msg)

    logger.info("Feature data type validation complete.")


    # Align indices of features and labels DataFrames
    logger.info("Aligning features and labels indices...")
    if not isinstance(X.index, pd.DatetimeIndex):
         logger.error("Features DataFrame index is not a DatetimeIndex.")
         raise TypeError("Features DataFrame must have a DatetimeIndex.")
    if not isinstance(ydf.index, pd.DatetimeIndex):
         logger.error("Labels DataFrame index is not a DatetimeIndex.")
         raise TypeError("Labels DataFrame must have a DatetimeIndex.")


    common_index = X.index.intersection(ydf.index)

    if common_index.empty:
        logger.error("No common index found between features and labels data. Cannot proceed.")
        raise ValueError("No common index between features and labels.")

    X = X.loc[common_index].copy()
    ydf = ydf.loc[common_index].copy()

    logger.info(f"Data aligned to common index. Shape: {X.shape}")


    # Drop original price/volume columns from features if they are still present
    raw_ohlcv_volume_and_labeling_cols = ['open', 'high', 'low', 'close', 'volume', 'vol_adj']
    cols_to_drop_if_present = [col for col in raw_ohlcv_volume_and_labeling_cols if col in X.columns]

    if cols_to_drop_if_present:
        X = X.drop(columns=cols_to_drop_if_present, errors='ignore')
        logger.info(f"Dropped potential non-feature columns from features: {cols_to_drop_if_present}")


    # --- Select Feature Subset if provided ---
    if features_to_use is not None:
        logger.info(f"Using a specific feature subset for training: {features_to_use}")
        missing_features = [feat for feat in features_to_use if feat not in X.columns]
        if missing_features:
            error_msg = f"Specified features not found in processed data: {missing_features}"
            logger.error(error_msg)
            raise ValueError(error_msg)
        X = X[features_to_use].copy()
        logger.info(f"Feature DataFrame reduced to specified subset. Shape: {X.shape}")
    else:
        logger.info("No specific feature subset provided. Using all available features.")


    # Extract labels
    if 'label' not in ydf.columns:
         logger.error("Labeled data is missing the 'label' column.")
         raise ValueError("Labeled data is missing the 'label' column.")

    y = ydf['label']

    # --- IMPORTANT: Remove rows with NA labels AND NA features BEFORE splitting ---
    initial_rows = len(X)
    valid_labels_mask = pd.notna(y)
    feature_columns_after_drop_and_subset = X.columns.tolist()
    numeric_feature_columns = X[feature_columns_after_drop_and_subset].select_dtypes(include=np.number).columns.tolist()

    valid_features_mask = X[numeric_feature_columns].notna().all(axis=1)

    combined_mask = valid_labels_mask & valid_features_mask

    X_full_cleaned = X[combined_mask].copy()
    y_full_cleaned = y[combined_mask].copy()

    rows_removed = initial_rows - len(X_full_cleaned)
    if rows_removed > 0:
        logger.info(f"Removed {rows_removed} rows with NA labels or NA features before splitting.")

    if X_full_cleaned.empty or y_full_cleaned.empty:
        logger.error("DataFrame is empty after removing NA labels and features. Cannot proceed with training.")
        raise ValueError("DataFrame is empty after removing NA labels and features.")


    y_full_cleaned = y_full_cleaned.astype(int)


    # --- Perform time-series split into Train, Validation, and Test sets ---
    n_samples = len(X_full_cleaned)
    if n_samples == 0:
         logger.error("No data points available for splitting after cleaning.")
         raise ValueError("No data points available for training after cleaning.")

    train_end_idx = int(n_samples * train_ratio)
    val_size = int(n_samples * val_ratio)
    test_size = n_samples - train_end_idx - val_size

    if train_end_idx <= 0:
        logger.error(f"Train set size ({train_end_idx}) is not positive. Adjust train_ratio.")
        raise ValueError("Train set size is zero or negative.")

    if val_ratio > 0 and val_size <= 0 and train_end_idx < n_samples:
         val_size = 1
         logger.warning("Adjusted validation size to 1 due to rounding or small dataset.")
         test_size = n_samples - train_end_idx - val_size

    if test_size < 0:
         test_size = 0
         logger.warning("Adjusted test size to 0 as calculated size was negative.")

    val_end_idx = train_end_idx + val_size

    if train_end_idx + val_size + test_size != n_samples:
         logger.error(f"Split size mismatch: train={train_end_idx}, val={val_size}, test={test_size}, total={train_end_idx + val_size + test_size}, expected={n_samples}")
         raise RuntimeError("Data split size mismatch.")


    X_train = X_full_cleaned.iloc[:train_end_idx].copy()
    y_train = y_full_cleaned.iloc[:train_end_idx].copy()

    X_val = X_full_cleaned.iloc[train_end_idx:val_end_idx].copy()
    y_val = y_full_cleaned.iloc[train_end_idx:val_end_idx].copy()

    X_test = X_full_cleaned.iloc[val_end_idx:].copy()
    y_test = y_full_cleaned.iloc[val_end_idx:].copy()

    logger.info(f"Data split into training ({len(X_train)} samples), validation ({len(X_val)} samples), and testing ({len(X_test)} samples).")

    if X_train.empty:
         logger.error("Training set is empty after splitting.")
         raise ValueError("Training set is empty after splitting.")

    unique_train_labels = y_train.unique()
    if len(unique_train_labels) < 2:
         logger.error(f"Training set does not contain at least two unique labels. Found: {unique_train_labels}")
         raise ValueError("Training set must contain at least two unique labels.")

    if not y_train.empty:
        logger.info(f"Training set class distribution:\n{y_train.value_counts(normalize=True).sort_index()}")
    if not y_val.empty:
        logger.info(f"Validation set class distribution:\n{y_val.value_counts(normalize=True).sort_index()}")
    if not y_test.empty:
        logger.info(f"Test set class distribution:\n{y_test.value_counts(normalize=True).sort_index()}")


    logger.info("Data loading and splitting complete.")
    return X_train, X_val, X_test, y_train, y_val, y_test, X_full_cleaned, y_full_cleaned


# --- Main Training Function ---
def main(
    symbol: str,
    interval: str,
    model_key: str,
    # REMOVED: label_strategy: str, as it's no longer needed for loading the labeled file
    train_ratio: float,
    skip_tuning: bool = False,
    features_to_use: Optional[List[str]] = None
):
    """
    Main function to load data, tune hyperparameters (optionally), train, evaluate, and save a model
    for ternary classification. Allows specifying a subset of features.
    Uses DataManager for loading data and saving model artifacts.

    Args:
        symbol (str): Trading pair symbol.
        interval (str): Time interval.
        model_key (str): Key for the model configuration in config.params.MODEL_CONFIG.
        train_ratio (float): The fraction of data to use for training (0.0 to 1.0).
        skip_tuning (bool): If True, skip hyperparameter tuning and use default params.
        features_to_use (Optional[List[str]]): A list of feature column names to use.
                                               If None, all available features are used.
    """
    start_time = time.time()
    logger.info(f"Starting model training pipeline for {symbol.upper()} @ {interval} with model: {model_key} (Ternary Classification)")
    logger.info(f"Training ratio: {train_ratio}")
    logger.info(f"Hyperparameter tuning enabled: {not skip_tuning}")
    # REMOVED: logger.info(f"Using labeling strategy: {label_strategy}")
    if features_to_use is not None:
        logger.info(f"Using specified feature subset: {features_to_use}")
    else:
        logger.info("Using all available features.")


    # --- Retrieve Model Configuration ---
    if model_key not in MODEL_CONFIG:
        logger.error(f"Unknown model key '{model_key}' not found in config.params.MODEL_CONFIG.")
        available_model_keys = [key for key in MODEL_CONFIG.keys() if 'model_type' in MODEL_CONFIG[key]]
        logger.error(f"Available model keys: {available_model_keys}")
        sys.exit(1)

    model_specific_config = copy.deepcopy(MODEL_CONFIG[model_key])
    model_specific_config['features_to_use'] = features_to_use
    val_ratio = model_specific_config.get('val_ratio', 0.1)
    logger.info(f"Validation ratio: {val_ratio}")

    # --- Load and Split Data ---
    try:
        # Pass the label_strategy to load_and_split_data
        X_tr, X_val, X_test, y_tr, y_val, y_test, X_full_cleaned, y_full_cleaned = load_and_split_data(
            symbol, interval, train_ratio, val_ratio, features_to_use=features_to_use
        )
    except (FileNotFoundError, ValueError, TypeError, RuntimeError) as e:
        logger.error(f"Failed to load or split data: {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"An unexpected error occurred during data loading or splitting: {e}", exc_info=True)
        sys.exit(1)


    # --- Hyperparameter Tuning (Conditional) ---
    if not skip_tuning and model_key != 'lstm':
        logger.info(f"Starting hyperparameter tuning for {model_key}...")
        try:
            tuned_params = run_tuning(
                model_key,
                X_full_cleaned,
                y_full_cleaned,
                model_specific_config
            )

            logger.info(f"Hyperparameter tuning complete for {model_key}. Best parameters found: {tuned_params}")
            model_specific_config['params'] = tuned_params
            logger.info(f"Updated model config with best tuning parameters: {model_specific_config}")

        except (ValueError, RuntimeError) as e:
            logger.error(f"Hyperparameter tuning failed: {e}")
            logger.warning(f"Proceeding with training using default parameters from config.params.MODEL_CONFIG for {model_key}.")
        except Exception as e:
            logger.error(f"An unexpected error occurred during hyperparameter tuning: {e}", exc_info=True)
            logger.warning(f"Proceeding with training using default parameters from config.params.MODEL_CONFIG for {model_key}.")

    # --- Initialize and Train Model with (potentially) Tuned Parameters ---
    logger.info(f"Initializing and training {model_key} model with updated parameters...")
    try:
        trainer = ModelTrainer(config=model_specific_config)

        if not X_val.empty and not y_val.empty:
             logger.info("Passing training and validation data to trainer.")
             trainer.train(X_tr, y_tr, X_val, y_val)
        else:
             logger.info("Passing training data to trainer (validation set is empty).")
             trainer.train(X_tr, y_tr)


    except (ValueError, TypeError, ImportError, RuntimeError) as e:
        logger.error(f"Model training failed: {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"An unexpected error occurred during training: {e}", exc_info=True)
        sys.exit(1)


    # --- Evaluate Model ---
    logger.info("Evaluating model on test set...")
    try:
        if not X_test.empty and not y_test.empty:
             results = trainer.evaluate(X_test, y_test)
             logger.info(f"Test set evaluation results: {results}")
        else:
             logger.warning("Test set is empty. Skipping evaluation.")
             results = {"note": "Test set is empty. No evaluation performed."}


    except (ValueError, TypeError, RuntimeError) as e:
        logger.error(f"Model evaluation failed: {e}")
    except Exception as e:
        logger.error(f"An unexpected error occurred during evaluation: {e}", exc_info=True)

    # --- Save Model ---
    logger.info("Saving trained model using DataManager...")
    try:
        trainer.save(
            symbol=symbol,
            interval=interval,
            model_key=model_key
        )
        logger.info("Model saved successfully via DataManager.")

    except (ValueError, ImportError, RuntimeError, OSError) as e:
         logger.error(f"Failed to save model via DataManager: {e}")
         sys.exit(1)
    except Exception as e:
        logger.error(f"An unexpected error occurred during saving via DataManager: {e}", exc_info=True)
        sys.exit(1)

    end_time = time.time()
    duration = end_time - start_time
    logger.info(f"Model training pipeline finished in {duration:.2f} seconds.")
    logger.info(f"Pipeline complete for {symbol.upper()} @ {interval} with model: {model_key}")


# --- Script Entry Point ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Train, evaluate, and save a trading model for ternary classification.'
    )
    parser.add_argument(
        '--symbol',
        type=str,
        required=True,
        help='Trading pair symbol (e.g., BTCUSDT)'
    )
    parser.add_argument(
        '--interval',
        type=str,
        required=True,
        choices=['1m', '5m', '15m', '30m', '1h', '2h', '4h', '6h', '8h', '12h', '1d', '3d', '1w', '1M'],
        help='Time interval for candles (e.g., 5m, 1h, 1d)'
    )
    parser.add_argument(
        '--model',
        type=str,
        choices=['random_forest', 'xgboost', 'lstm'],
        default='xgboost',
        help="Model type to train. Available: ['random_forest', 'xgboost', 'lstm']. Default: xgboost."
    )
    # REMOVED --label-strategy argument as it's no longer needed for loading the labeled file
    # parser.add_argument(
    #     '--label-strategy',
    #     type=str,
    #     required=True,
    #     choices=LabelGenerator.get_available_strategies(),
    #     help=f'Labeling strategy used to generate the labels (e.g., directional_ternary). Available: {", ".join(LabelGenerator.get_available_strategies())}'
    # )
    parser.add_argument(
        '--train_ratio',
        type=float,
        default=0.8,
        help='Fraction of data to use for training (0.0 to 1.0, exclusive). Default: 0.8'
    )
    parser.add_argument(
        '--skip_tuning',
        action='store_true',
        help='Skip hyperparameter tuning and use default parameters from config.params.MODEL_CONFIG.'
    )
    parser.add_argument(
        '--features',
        type=str,
        nargs='+',
        help='Optional list of feature names to use for training. If not provided, all features are used.'
    )


    args = parser.parse_args()

    if not (0.0 < args.train_ratio < 1.0):
         logger.error(f"Invalid train_ratio value: {args.train_ratio}. Must be between 0.0 and 1.0 (exclusive).")
         sys.exit(1)

    model_specific_config_for_val_check = MODEL_CONFIG.get(args.model, {})
    val_ratio_check = model_specific_config_for_val_check.get('val_ratio', 0.1)

    if not (0.0 <= val_ratio_check < 1.0 and (args.train_ratio + val_ratio_check) <= 1.0):
         logger.error(f"Invalid val_ratio value ({val_ratio_check}) or combination with train_ratio ({args.train_ratio}). Ensure 0 <= val_ratio, and train_ratio + val_ratio <= 1.")
         sys.exit(1)


    try:
        main(
            symbol=args.symbol,
            interval=args.interval,
            model_key=args.model,
            # REMOVED: label_strategy=args.label_strategy,
            train_ratio=args.train_ratio,
            skip_tuning=args.skip_tuning,
            features_to_use=args.features
        )

    except SystemExit:
         pass
    except Exception:
        logger.exception("Model training script terminated due to an unhandled error.")
        sys.exit(1)

    """
    Usage example:

    Train the default XGBoost model for ternary classification:
        python scripts/train_model.py --symbol BTCUSDT --interval 1h

    Train the RandomForest model for ternary classification:
        python scripts.train_model.py --symbol ADAUSDT --interval 5m --model random_forest

    Train the LSTM model (requires TensorFlow):
        python scripts.train_model.py --symbol ETHUSDT --interval 15m --model lstm --train_ratio 0.7

    Train with tuning (default for non-LSTM):
        python -m scripts.train_model --symbol ADAUSDT --interval 5m --model random_forest
        python -m scripts.train_model --symbol ADAUSDT --interval 5m --model xgboost

    Train skipping tuning:
        python scripts.train_model.py --symbol ADAUSDT --interval 5m --model random_forest --skip_tuning

    Train using a specific subset of features:
        python scripts.train_model.py --symbol BTCUSDT --interval 1h --features ema_10 rsi_14 macd

    Ensure you have processed and labeled data files (including label 0) in your data/
    and config/params.py (with MODEL_CONFIG, GENERAL_CONFIG) and config/paths.py are correctly configured
    (including 'trained_models_dir' path and 'trained_model_pattern').
    The feature generation script must produce an ATR column named 'atr_{lookback}'
    (e.g., 'atr_14') matching the 'vol_adj_lookback' parameter in LABELING_CONFIG
    if using the 'triple_barrier' strategy, and save it to the processed data file.
    """
