#!/usr/bin/env python3
"""
train_model.py

Loads processed data with features and labels (-1, 0, 1); trains a model using
the configuration from config.params.app_config.model; evaluates the model;
and saves the trained model and metadata using DataManager.

Supports RandomForest, XGBoost, and LSTM models for TERNARY classification.

Uses the unified configuration structure from config.params.app_config.
Configures logging using utils/logger_config.py.
Includes optional hyperparameter tuning using RandomizedSearchCV and TimeSeriesSplit.
Handles train and test data splitting.
Removes rows with NaN values in features or labels before splitting.

MODIFIED: Configuration is now accessed via the central 'app_config' object.
MODIFIED: Data splitting now strictly follows 'train_test_split_ratio' from ModelConfig,
          with LSTM handling its own validation split internally via 'validation_split'.
MODIFIED: Removed redundant command-line arguments for train_ratio and val_ratio,
          as these are now managed by configuration.
MODIFIED: Updated DataManager calls for loading data and saving model artifacts.
MODIFIED: Adjusted labeled data loading to no longer require a strategy-specific suffix.
MODIFIED: Added command-line arguments and logic for PCA dimensionality reduction.
MODIFIED: Added explicit imports for RandomForestParams and XGBoostParams to resolve UndefinedVariable errors.
MODIFIED: ModelTrainer and related training utilities are now located in utils/training/.
MODIFIED: Ensured DataManager is correctly imported and accessible.
MODIFIED: Corrected access to TF_AVAILABLE and tf by making them attributes of ModelConfig.
MODIFIED: Ensured the model_type from CLI is correctly passed to the ModelConfig used for training.
MODIFIED: Adjusted XGBoost training to handle early stopping when no explicit validation set is provided.
"""

import sys
import logging
import argparse
from pathlib import Path
from typing import Tuple, List, Dict, Any, Optional, Union
from collections import Counter
import time
import copy

import pandas as pd
import numpy as np

# --- Import dotenv to load environment variables FIRST ---
from dotenv import load_dotenv, find_dotenv
# Load environment variables from .env file
load_dotenv(find_dotenv())
# --- End dotenv import ---

# Import scikit-learn and imblearn components for tuning
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit, train_test_split # Import train_test_split
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.compose import ColumnTransformer

# Import parameter distributions for RandomizedSearchCV
try:
    from scipy.stats import uniform, randint
    SCIPY_AVAILABLE = True
except ImportError:
    uniform = None
    randint = None
    SCIPY_AVAILABLE = False
    logging.warning("Scipy not found. Hyperparameter tuning distributions (uniform, randint) will not be available.")

from xgboost import XGBClassifier


# Add project root to Python path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Import configuration and utilities
try:
    from config.paths import PATHS
    from config.params import app_config
    from config.model_config_schema import ModelConfig, RandomForestParams, XGBoostParams, LSTMParams
    from utils.data_manager import DataManager
    from utils.training.model_trainer import ModelTrainer
    from utils.logger_config import setup_rotating_logging
except ImportError as e:
    print(f"ERROR: Failed to import necessary modules. Ensure config/, utils/ are correctly structured and required files exist. Error: {e}", file=sys.stderr)
    sys.exit(1)
except FileNotFoundError as e:
    print(f"ERROR: Configuration file not found: {e}. Ensure config/params.py and config/paths.py exist.", file=sys.stderr)
    sys.exit(1)
except AttributeError as e:
     print(f"ERROR: Configuration object missing expected attribute or key: {e}. Check config/params.py and config/paths.py.", file=sys.stderr)
     sys.exit(1)
except Exception as e:
    print(f"ERROR: An unexpected error occurred during initial imports or configuration loading: {e}", file=sys.stderr)
    sys.exit(1)


# --- Conditional Import for TensorFlow/Keras Status ---
# Now access TF_AVAILABLE and tf directly from app_config.model
TF_AVAILABLE = app_config.model.TF_AVAILABLE
tf = app_config.model.tf

# --- Configure Rotating Logging ---
try:
    setup_rotating_logging('train_model', logging.INFO)
    logger = logging.getLogger(__name__)
    logger.info("Logging configured successfully in train_model.py using setup_rotating_logging.")

    if TF_AVAILABLE:
        logger.info(f"TensorFlow (version {getattr(tf, '__version__', 'unknown')}) imported successfully.")
        if tf.config.list_physical_devices('GPU'):
            logger.info("GPU is available and enabled for TensorFlow.")
        else:
            logger.info("GPU is not available or not enabled for TensorFlow.")
    else:
        logger.warning("TensorFlow not found. LSTM model type will not be available.")


except ImportError:
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler(sys.stdout)]
    )
    logger = logging.getLogger(__name__)
    logger.warning("utils.logger_config not found or setup_rotating_logging failed. Using basic logging configuration.")
except Exception as e:
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler(sys.stdout)]
    )
    logger = logging.getLogger(__name__)
    logger.error(f"Failed to configure logging using utils.logger_config: {e}. Using basic logging.", exc_info=True)


def run_tuning(model_key: str, X_full_cleaned: pd.DataFrame, y_full_cleaned: pd.Series, current_model_config: ModelConfig) -> Dict[str, Any]:
    """
    Performs hyperparameter tuning for the specified model using RandomizedSearchCV
    with TimeSeriesSplit.

    Args:
        model_key (str): The key for the model type ('random_forest', 'xgboost').
        X_full_cleaned (pd.DataFrame): Full cleaned features data (for tuning CV).
        y_full_cleaned (pd.Series): Full cleaned labels data (for tuning CV).
        current_model_config (ModelConfig): The ModelConfig instance for the current model type,
                                            potentially updated with CLI args (e.g., PCA).

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
        return current_model_config.lstm_params.__dict__

    if model_key == 'random_forest':
        base_model_params = current_model_config.random_forest_params.__dict__.copy()
        param_dist_dataclass = current_model_config.random_forest_tuning_params
    elif model_key == 'xgboost':
        base_model_params = current_model_config.xgboost_params.__dict__.copy()
        param_dist_dataclass = current_model_config.xgboost_tuning_params
    else:
        raise ValueError(f"Model type '{model_key}' is not supported for tuning.")

    param_dist = {k: v for k, v in param_dist_dataclass.__dict__.items() if v is not None}

    if not param_dist:
         logger.warning(f"No tuning parameter distributions found for {model_key} in config. Skipping tuning and using default parameters.")
         return base_model_params


    logger.info(f"Tuning parameter distributions: {param_dist}")

    try:
        if model_key == 'random_forest':
             rf_params_for_tuning = {k: v for k, v in base_model_params.items() if k not in ['class_balancing']}
             model = RandomForestClassifier(
                 random_state=app_config.general.random_seed,
                 n_jobs=app_config.general.n_processors,
                 **rf_params_for_tuning
             )
        elif model_key == 'xgboost':
             xgb_params_for_tuning = {k: v for k, v in base_model_params.items() if k not in ['class_balancing', 'early_stopping_rounds']} # Exclude early_stopping_rounds from base for tuning
             final_xgb_params_for_tuning = {
                 'objective': 'multi:softmax',
                 'num_class': 3,
                 'eval_metric': 'mlogloss',
                 'random_state': app_config.general.random_seed,
                 'n_jobs': app_config.general.n_processors,
                 **xgb_params_for_tuning
             }
             model = XGBClassifier(**final_xgb_params_for_tuning)
        else:
             raise ValueError(f"Model type '{model_key}' is not supported for tuning.")

        logger.debug(f"Created base model instance for tuning: {type(model).__name__}")
    except Exception as e:
        logger.error(f"Failed to create base model instance for tuning: {e}", exc_info=True)
        raise RuntimeError("Failed to create base model for tuning.")


    scaler_type = current_model_config.scaler_type
    scaler_step = ('scaler', StandardScaler())

    if scaler_type == 'minmax':
        scaler_step = ('scaler', MinMaxScaler())
    elif scaler_type is None:
        scaler_step = None
    else:
        pass

    numeric_transformer_steps = [scaler_step] if scaler_step is not None else []

    pca_enabled_tuning = current_model_config.pca_enabled
    pca_n_components_tuning = current_model_config.pca_n_components

    if pca_enabled_tuning:
        pca_params = {'n_components': pca_n_components_tuning}
        logger.info(f"Adding PCA to tuning preprocessor with params: {pca_params}")
        numeric_transformer_steps.append(('pca', PCA(**pca_params)))

    all_numeric_cols = X_full_cleaned.select_dtypes(include=np.number).columns.tolist()
    numeric_features_for_preprocessor = all_numeric_cols
    if current_model_config.features_to_use:
        numeric_features_for_preprocessor = [f for f in all_numeric_cols if f in current_model_config.features_to_use]


    transformers = []
    if numeric_features_for_preprocessor:
        numeric_transformer_pipeline = Pipeline(steps=numeric_transformer_steps)
        transformers.append(('num', numeric_transformer_pipeline, numeric_features_for_preprocessor))
    else:
        logger.warning("No numeric features found to apply preprocessor for tuning. Preprocessor will be mostly passthrough.")


    preprocessor = ColumnTransformer(
        transformers=transformers,
        remainder='passthrough'
    )


    steps = [('preprocessor', preprocessor)]

    balanced_strategy_tuning = base_model_params.get('class_balancing')
    if balanced_strategy_tuning == 'undersampling':
         sampler = RandomUnderSampler(random_state=app_config.general.random_seed)
         steps.append(('sampler', sampler))
         logger.info("Added RandomUnderSampler to tuning pipeline.")
    elif balanced_strategy_tuning == 'oversampling':
         sampler = SMOTE(random_state=app_config.general.random_seed)
         steps.append(('sampler', sampler))
         logger.info("Added SMOTE to tuning pipeline.")
    elif isinstance(balanced_strategy_tuning, dict):
        sampler = SMOTE(random_state=app_config.general.random_seed, **balanced_strategy_tuning)
        steps.append(('sampler', sampler))
        logger.info(f"Added SMOTE to tuning pipeline with custom params: {balanced_strategy_tuning}")
    elif balanced_strategy_tuning is not None:
         logger.warning(f"Unsupported 'class_balancing' strategy '{balanced_strategy_tuning}'. Skipping sampler in tuning pipeline.")


    steps.append(('model', model))

    pipeline = Pipeline(steps)
    logger.debug(f"Tuning pipeline created with steps: {[name for name, _ in pipeline.steps]}")


    n_splits = app_config.general.hyperparameter_tuning_cv_folds
    if not isinstance(n_splits, int) or n_splits <= 0:
         logger.warning(f"Invalid hyperparameter_tuning_cv_folds ({n_splits}) in general config. Defaulting to 5.")
         n_splits = 5
    tscv = TimeSeriesSplit(n_splits=n_splits)
    logger.info(f"Using TimeSeriesSplit with {n_splits} splits for tuning.")


    scoring_metric = current_model_config.tuning_scoring_metric
    if not scoring_metric:
        scoring_metric = 'f1_macro'
        logger.warning(f"No tuning scoring metric found in model config. Defaulting to '{scoring_metric}'.")

    logger.info(f"Using '{scoring_metric}' as the scoring metric for tuning.")


    n_iter = app_config.general.hyperparameter_tuning_n_iter
    if not isinstance(n_iter, int) or n_iter <= 0:
         logger.warning(f"Invalid hyperparameter_tuning_n_iter ({n_iter}) in general config. Defaulting to 10.")
         n_iter = 10

    cv_jobs = app_config.general.n_processors


    logger.info(f"Running RandomizedSearchCV with {n_iter} iterations and {n_splits}-fold TimeSeriesSplit...")
    try:
        random_search = RandomizedSearchCV(
            pipeline,
            param_distributions=param_dist,
            n_iter=n_iter,
            cv=tscv,
            scoring=scoring_metric,
            random_state=app_config.general.random_seed,
            n_jobs=cv_jobs,
            verbose=1
        )

        y_full_cleaned_mapped = y_full_cleaned.map({-1: 0, 0: 1, 1: 2})


        random_search.fit(X_full_cleaned, y_full_cleaned_mapped)

        logger.info("RandomizedSearchCV complete.")
        logger.info(f"Best parameters found: {random_search.best_params_}")
        logger.info(f"Best cross-validation score ({scoring_metric}): {random_search.best_score_:.4f}")

        best_model_params = {k.replace('model__', ''): v for k, v in random_search.best_params_.items() if k.startswith('model__')}

        tuned_params = {**base_model_params, **best_model_params}
        logger.info(f"Extracted best model parameters: {best_model_params}")
        logger.info(f"Merged tuned parameters: {tuned_params}")

        return tuned_params

    except Exception as e:
        logger.error(f"An error occurred during hyperparameter tuning: {e}", exc_info=True)
        raise RuntimeError(f"Hyperparameter tuning failed: {e}")


def load_and_split_data(
    symbol: str,
    interval: str,
    train_split_ratio: float,
    features_to_use: Optional[List[str]] = None
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.DataFrame, pd.Series]:
    logger.info(f"Loading data for {symbol.upper()} @ {interval}...")

    dm = DataManager()

    try:
        logger.info(f"Attempting to load processed data (features) for {symbol.upper()} {interval}")
        X = dm.load_data(
            symbol=symbol.upper(),
            interval=interval,
            data_type='processed'
        )
        logger.info(f"Successfully loaded features. Shape: {X.shape}")

        if 'open_time' in X.columns:
            X = X.drop(columns=['open_time'])
            logger.info("Dropped 'open_time' column from features DataFrame.")


        logger.info(f"Attempting to load labeled data for {symbol.upper()} {interval}")
        ydf = dm.load_data(
            symbol=symbol.upper(),
            interval=interval,
            data_type='labeled',
        )
        logger.info(f"Successfully loaded labels. Shape: {ydf.shape}")


    except FileNotFoundError:
        raise
    except Exception as e:
        logger.error(f"An unexpected error occurred during data loading using DataManager: {e}", exc_info=True)
        raise RuntimeError(f"Data loading failed: {e}")


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


    raw_ohlcv_volume_and_labeling_cols = ['open', 'high', 'low', 'close', 'volume', 'vol_adj']
    cols_to_drop_if_present = [col for col in raw_ohlcv_volume_and_labeling_cols if col in X.columns]

    if cols_to_drop_if_present:
        X = X.drop(columns=cols_to_drop_if_present, errors='ignore')
        logger.info(f"Dropped potential non-feature columns from features: {cols_to_drop_if_present}")


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


    if 'label' not in ydf.columns:
         logger.error("Labeled data is missing the 'label' column.")
         raise ValueError("Labeled data is missing the 'label' column.")

    y = ydf['label']

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


    n_samples = len(X_full_cleaned)
    if n_samples == 0:
         logger.error("No data points available for splitting after cleaning.")
         raise ValueError("No data points available for training after cleaning.")

    train_end_idx = int(n_samples * train_split_ratio)
    test_size = n_samples - train_end_idx

    if train_end_idx <= 0:
        logger.error(f"Train set size ({train_end_idx}) is not positive. Adjust train_split_ratio.")
        raise ValueError("Train set size is zero or negative.")

    if test_size < 0:
         test_size = 0
         logger.warning("Adjusted test size to 0 as calculated size was negative.")

    X_train = X_full_cleaned.iloc[:train_end_idx].copy()
    y_train = y_full_cleaned.iloc[:train_end_idx].copy()

    X_test = X_full_cleaned.iloc[train_end_idx:].copy()
    y_test = y_full_cleaned.iloc[train_end_idx:].copy()

    logger.info(f"Data split into training ({len(X_train)} samples) and testing ({len(X_test)} samples).")

    if X_train.empty:
         logger.error("Training set is empty after splitting.")
         raise ValueError("Training set is empty after splitting.")

    unique_train_labels = y_train.unique()
    if len(unique_train_labels) < 2:
         logger.error(f"Training set does not contain at least two unique labels. Found: {unique_train_labels}")
         raise ValueError("Training set must contain at least two unique labels.")

    if not y_train.empty:
        logger.info(f"Training set class distribution:\n{y_train.value_counts(normalize=True).sort_index()}")
    if not y_test.empty:
        logger.info(f"Test set class distribution:\n{y_test.value_counts(normalize=True).sort_index()}")


    logger.info("Data loading and splitting complete.")
    return X_train, X_test, y_train, y_test, X_full_cleaned, y_full_cleaned


def main(
    symbol: str,
    interval: str,
    model_key: str,
    skip_tuning: bool = False,
    features_to_use: Optional[List[str]] = None,
    enable_pca: bool = False,
    pca_n_components: Optional[Union[int, float]] = None
):
    start_time = time.time()
    logger.info(f"Starting model training pipeline for {symbol.upper()} @ {interval} with model: {model_key} (Ternary Classification)")
    logger.info(f"Hyperparameter tuning enabled: {not skip_tuning}")
    if features_to_use is not None:
        logger.info(f"Using specified feature subset: {features_to_use}")
    else:
        logger.info("Using all available features.")

    if enable_pca:
        logger.info(f"PCA dimensionality reduction enabled with n_components: {pca_n_components}")
    else:
        logger.info("PCA dimensionality reduction disabled.")


    model_config_for_trainer = copy.deepcopy(app_config.model)
    # --- FIX 1: Set the model_type based on the command-line argument ---
    model_config_for_trainer.model_type = model_key
    model_config_for_trainer.features_to_use = features_to_use

    if enable_pca:
        model_config_for_trainer.pca_enabled = True
        if pca_n_components is not None:
            model_config_for_trainer.pca_n_components = pca_n_components
        else:
            logger.info("No specific n_components provided for PCA. Using default from ModelConfig.")
    else:
        model_config_for_trainer.pca_enabled = False


    try:
        X_tr, X_test, y_tr, y_test, X_full_cleaned, y_full_cleaned = load_and_split_data(
            symbol,
            interval,
            train_split_ratio=model_config_for_trainer.train_test_split_ratio,
            features_to_use=features_to_use
        )
    except (FileNotFoundError, ValueError, TypeError, RuntimeError) as e:
        logger.error(f"Failed to load or split data: {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"An unexpected error occurred during data loading or splitting: {e}", exc_info=True)
        sys.exit(1)


    if not skip_tuning and model_key != 'lstm':
        logger.info(f"Starting hyperparameter tuning for {model_key}...")
        try:
            tuned_params = run_tuning(
                model_key,
                X_full_cleaned,
                y_full_cleaned,
                model_config_for_trainer
            )

            logger.info(f"Hyperparameter tuning complete for {model_key}. Best parameters found: {tuned_params}")
            if model_key == 'random_forest':
                model_config_for_trainer.random_forest_params = RandomForestParams(**tuned_params)
            elif model_key == 'xgboost':
                model_config_for_trainer.xgboost_params = XGBoostParams(**tuned_params)

            logger.info(f"Updated model config with best tuning parameters: {model_config_for_trainer}")

        except (ValueError, RuntimeError) as e:
            logger.error(f"Hyperparameter tuning failed: {e}")
            logger.warning(f"Proceeding with training using parameters from app_config.model for {model_key}.")
        except Exception as e:
            logger.error(f"An unexpected error occurred during hyperparameter tuning: {e}", exc_info=True)
            logger.warning(f"Proceeding with training using parameters from app_config.model for {model_key}.")

    logger.info(f"Initializing and training {model_key} model with updated parameters...")
    try:
        # --- FIX 2: Prepare validation data for XGBoost early stopping if not tuning and enabled ---
        X_val_for_trainer = pd.DataFrame()
        y_val_for_trainer = pd.Series(dtype=int)

        if model_key == 'xgboost' and not skip_tuning and model_config_for_trainer.xgboost_params.early_stopping_rounds is not None:
            # If early stopping is enabled for XGBoost but no explicit tuning/validation split,
            # create a small validation set from training data.
            # Using simple train_test_split for this internal validation set for simplicity
            # since TimeSeriesSplit is more complex to subset within this context.
            if len(X_tr) > 1000: # Ensure enough data to split
                X_tr, X_val_for_trainer, y_tr, y_val_for_trainer = train_test_split(
                    X_tr, y_tr,
                    test_size=0.1, # Use 10% of training data for validation
                    shuffle=False, # Maintain time series order
                    stratify=None # Stratification can be tricky with time series, keep it simple
                )
                logger.info(f"Created internal validation set for XGBoost early stopping: {len(X_val_for_trainer)} samples.")
            else:
                logger.warning("Not enough training data to create internal validation set for XGBoost early stopping. Disabling early stopping.")
                model_config_for_trainer.xgboost_params.early_stopping_rounds = None


        trainer = ModelTrainer(model_config=model_config_for_trainer)
        trainer.train(X_tr, y_tr, X_val=X_val_for_trainer, y_val=y_val_for_trainer)


    except (ValueError, TypeError, ImportError, RuntimeError) as e:
        logger.error(f"Model training failed: {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"An unexpected error occurred during training: {e}", exc_info=True)
        sys.exit(1)


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
    parser.add_argument(
        '--skip_tuning',
        action='store_true',
        help='Skip hyperparameter tuning and use default parameters from app_config.model.'
    )
    parser.add_argument(
        '--features',
        type=str,
        nargs='+',
        help='Optional list of feature names to use for training. If not provided, all features are used.'
    )
    parser.add_argument(
        '--enable_pca',
        action='store_true',
        help='Enable PCA dimensionality reduction for the model.'
    )
    parser.add_argument(
        '--pca_components',
        type=lambda x: int(x) if x.isdigit() else float(x),
        default=None,
        help='Number of PCA components (int) or variance to explain (float between 0 and 1). '
             'Default from ModelConfig if not specified.'
    )

    args = parser.parse_args()

    try:
        main(
            symbol=args.symbol,
            interval=args.interval,
            model_key=args.model,
            skip_tuning=args.skip_tuning,
            features_to_use=args.features,
            enable_pca=args.enable_pca,
            pca_n_components=args.pca_components
        )

    except SystemExit:
         pass
    except Exception:
        logger.exception("Model training script terminated due-to an unhandled error.")
        sys.exit(1)

    """
    Usage example:

    Train the default XGBoost model for ternary classification:
        python scripts/train_model.py --symbol BTCUSDT --interval 1h

    Train the RandomForest model for ternary classification:
        python scripts.train_model.py --symbol ADAUSDT --interval 5m --model random_forest

    Train the LSTM model (requires TensorFlow):
        python scripts.train_model.py --symbol ETHUSDT --interval 15m --model lstm

    Train with tuning (default for non-LSTM):
        python -m scripts.train_model --symbol ADAUSDT --interval 5m --model random_forest
        python -m scripts.train_model --symbol ADAUSDT --interval 5m --model xgboost

    Train skipping tuning:
        python scripts/train_model.py --symbol ADAUSDT --interval 5m --model random_forest --skip_tuning

    Train using a specific subset of features:
        python scripts/train_model.py --symbol BTCUSDT --interval 1h --features ema_10 rsi_14 macd

    Train with PCA, explaining 90% variance:
        python scripts/train_model.py --symbol BTCUSDT --interval 1h --enable_pca --pca_components 0.9

    Train with PCA, keeping 10 components:
        python scripts/train_model.py --symbol ADAUSDT --interval 5m --model random_forest --enable_pca --pca_components 10

    Ensure you have processed and labeled data files (including label 0) in your data/
    and config/params.py (with AppConfig) and config/paths.py are correctly configured
    (including 'trained_models_dir' path and 'trained_model_pattern').
    The feature generation script must produce an ATR column named 'atr_{lookback}'
    (e.g., 'atr_14') matching the 'vol_adj_lookback' parameter in LABELING_CONFIG
    if using the 'triple_barrier' strategy, and save it to the processed data file.
    """
