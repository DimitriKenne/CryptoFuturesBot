#!/usr/bin/env python3
"""
train_model.py

Loads processed data with features and labels (-1, 0, 1); trains a model using
the configuration from config.params.app_config.model; evaluates the model;
and saves the trained model and metadata using DataManager.
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

from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit, train_test_split
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.compose import ColumnTransformer

try:
    from scipy.stats import uniform, randint
except ImportError:
    uniform = None
    randint = None

from xgboost import XGBClassifier

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Using new PATH_CONFIG for consistency
from config.paths import PATH_CONFIG, PATHS
from config.params import app_config
from config.model import ModelConfig, RandomForestParams, XGBoostParams, LSTMParams
from config.validator import validate_config
from utils.data_management.data_manager import DataManager
from utils.training.model_trainer import ModelTrainer
from utils.logger_config import setup_rotating_logging

# --- Logging and Validation ---
setup_rotating_logging('train_model', logging.INFO)
logger = logging.getLogger(__name__)
validate_config(app_config)
logger.info("Configuration validated successfully.")

def run_tuning(model_type: str, X_full_cleaned: pd.DataFrame, y_full_cleaned: pd.Series, current_model_config: ModelConfig) -> Dict[str, Any]:
    """
    Performs hyperparameter tuning for the specified model using RandomizedSearchCV
    with TimeSeriesSplit.
    """
    logger.info(f"Starting hyperparameter tuning for {model_type}...")

    if model_type == 'lstm':
        logger.warning("Hyperparameter tuning for LSTM models is not implemented in this script.")
        return current_model_config.lstm_params.__dict__

    if model_type == 'random_forest':
        base_model_params = current_model_config.random_forest_params.__dict__.copy()
        param_dist_dataclass = current_model_config.random_forest_tuning_params
    elif model_type == 'xgboost':
        base_model_params = current_model_config.xgboost_params.__dict__.copy()
        param_dist_dataclass = current_model_config.xgboost_tuning_params
    else:
        raise ValueError(f"Model type '{model_type}' is not supported for tuning.")

    param_dist = {k: v for k, v in param_dist_dataclass.__dict__.items() if v is not None}
    if not param_dist:
        logger.warning(f"No tuning parameter distributions found for {model_type}. Using defaults.")
        return base_model_params

    if model_type == 'random_forest':
        model = RandomForestClassifier(random_state=app_config.general.random_seed, n_jobs=app_config.general.n_processors)
    elif model_type == 'xgboost':
        model = XGBClassifier(objective='multi:softmax', num_class=3, eval_metric='mlogloss', random_state=app_config.general.random_seed, n_jobs=app_config.general.n_processors)

    numeric_features = X_full_cleaned.select_dtypes(include=np.number).columns.tolist()
    numeric_transformer_steps = []
    if current_model_config.scaler_type == 'standard':
        numeric_transformer_steps.append(('scaler', StandardScaler()))
    elif current_model_config.scaler_type == 'minmax':
        numeric_transformer_steps.append(('scaler', MinMaxScaler()))

    if current_model_config.pca_enabled:
        numeric_transformer_steps.append(('pca', PCA(n_components=current_model_config.pca_n_components)))

    preprocessor = ColumnTransformer(transformers=[('num', Pipeline(steps=numeric_transformer_steps), numeric_features)], remainder='passthrough')
    
    steps = [('preprocessor', preprocessor)]
    if base_model_params.get('class_balancing') == 'undersampling':
        steps.append(('sampler', RandomUnderSampler(random_state=app_config.general.random_seed)))
    elif base_model_params.get('class_balancing') == 'oversampling':
        steps.append(('sampler', SMOTE(random_state=app_config.general.random_seed)))

    steps.append(('model', model))
    pipeline = Pipeline(steps)

    tscv = TimeSeriesSplit(n_splits=app_config.general.hyperparameter_tuning_cv_folds)
    random_search = RandomizedSearchCV(
        pipeline, param_distributions=param_dist, n_iter=app_config.general.hyperparameter_tuning_n_iter,
        cv=tscv, scoring=current_model_config.tuning_scoring_metric, random_state=app_config.general.random_seed,
        n_jobs=app_config.general.n_processors, verbose=1
    )
    
    y_full_cleaned_mapped = y_full_cleaned.map({-1: 0, 0: 1, 1: 2})
    random_search.fit(X_full_cleaned, y_full_cleaned_mapped)

    logger.info(f"Best parameters found: {random_search.best_params_}")
    best_model_params = {k.replace('model__', ''): v for k, v in random_search.best_params_.items()}
    return {**base_model_params, **best_model_params}

def load_and_split_data(symbol: str, interval: str, train_split_ratio: float, features_to_use: Optional[List[str]] = None) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.DataFrame, pd.Series]:
    logger.info(f"Loading data for {symbol.upper()} @ {interval}...")
    dm = DataManager()

    X = dm.load_data(symbol=symbol.upper(), interval=interval, data_type='processed')
    if X is None or X.empty:
        raise FileNotFoundError(f"Processed feature data not found or is empty for {symbol.upper()} {interval}.")
    
    ydf = dm.load_data(symbol=symbol.upper(), interval=interval, data_type='labeled')
    if ydf is None or ydf.empty:
        raise FileNotFoundError(f"Labeled data not found or is empty for {symbol.upper()} {interval}.")

    common_index = X.index.intersection(ydf.index)
    X = X.loc[common_index].copy()
    y = ydf.loc[common_index]['label'].copy()

    if features_to_use:
        X = X[features_to_use]

    combined = pd.concat([X, y], axis=1).dropna()
    X_full_cleaned = combined.drop(columns=['label'])
    y_full_cleaned = combined['label'].astype(int)

    train_end_idx = int(len(X_full_cleaned) * train_split_ratio)
    X_train = X_full_cleaned.iloc[:train_end_idx]
    y_train = y_full_cleaned.iloc[:train_end_idx]
    X_test = X_full_cleaned.iloc[train_end_idx:]
    y_test = y_full_cleaned.iloc[train_end_idx:]

    return X_train, X_test, y_train, y_test, X_full_cleaned, y_full_cleaned

def main(
    symbol: str,
    interval: str,
    model_type: str,
    user_login: str, # ADDED
    skip_tuning: bool = False,
    features: Optional[List[str]] = None,
    enable_pca: bool = False,
    pca_components: Optional[Union[int, float]] = None
):
    start_time = time.time()
    logger.info(f"Starting model training pipeline for {symbol.upper()} @ {interval} with model: {model_type}")

    model_config_for_trainer = copy.deepcopy(app_config.model)
    model_config_for_trainer.model_type = model_type
    model_config_for_trainer.features_to_use = features
    if enable_pca:
        model_config_for_trainer.pca_enabled = True
        if pca_components is not None:
            model_config_for_trainer.pca_n_components = pca_components
    else:
        model_config_for_trainer.pca_enabled = False

    try:
        X_tr, X_test, y_tr, y_test, X_full_cleaned, y_full_cleaned = load_and_split_data(
            symbol, interval, model_config_for_trainer.train_test_split_ratio, features
        )
    except FileNotFoundError as e:
        logger.error(f"Data loading failed: {e}")
        logger.error("Please ensure you have run 'generate_features.py' and 'generate_labels.py' for the symbol and interval.")
        sys.exit(1)
    except Exception as e:
        logger.error(f"An unexpected error occurred during data loading: {e}", exc_info=True)
        sys.exit(1)

    if not skip_tuning and model_type != 'lstm':
        try:
            tuned_params = run_tuning(model_type, X_full_cleaned, y_full_cleaned, model_config_for_trainer)
            if model_type == 'random_forest':
                model_config_for_trainer.random_forest_params = RandomForestParams(**tuned_params)
            elif model_type == 'xgboost':
                model_config_for_trainer.xgboost_params = XGBoostParams(**tuned_params)
        except Exception as e:
            logger.error(f"Hyperparameter tuning failed: {e}", exc_info=True)
            logger.warning("Proceeding with default parameters.")

    logger.info(f"Initializing and training {model_type} model...")
    try:
        X_val_for_trainer, y_val_for_trainer = pd.DataFrame(), pd.Series(dtype=int)
        if model_type == 'xgboost' and model_config_for_trainer.xgboost_params.early_stopping_rounds:
            if len(X_tr) > 1000:
                X_tr, X_val_for_trainer, y_tr, y_val_for_trainer = train_test_split(
                    X_tr, y_tr, test_size=0.1, shuffle=False
                )
        trainer = ModelTrainer(model_config=model_config_for_trainer)
        trainer.train(X_tr, y_tr, X_val=X_val_for_trainer, y_val=y_val_for_trainer, symbol=symbol, interval=interval)
    except Exception as e:
        logger.error(f"Model training failed: {e}", exc_info=True)
        sys.exit(1)

    logger.info("Evaluating model on test set...")
    if not X_test.empty:
        try:
            results = trainer.evaluate(X_test, y_test)
            logger.info(f"Test set evaluation results: {results}")
        except Exception as e:
            logger.error(f"Model evaluation failed: {e}", exc_info=True)

    logger.info("Saving trained model using DataManager...")
    try:
        # UPDATED: Pass the user_login to the save method
        trainer.save(symbol=symbol, interval=interval, saved_by=user_login)
        logger.info("Model saved successfully.")
    except Exception as e:
         logger.error(f"Failed to save model via DataManager: {e}", exc_info=True)
         sys.exit(1)

    duration = time.time() - start_time
    logger.info(f"Model training pipeline finished in {duration:.2f} seconds.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train, evaluate, and save a trading model.')
    parser.add_argument('--symbol', type=str, required=True, help='Trading pair symbol (e.g., BTCUSDT)')
    parser.add_argument(
        '--interval', type=str, required=True,
        choices=['1m', '5m', '15m', '30m', '1h', '2h', '4h', '6h', '8h', '12h', '1d', '3d', '1w', '1M'],
        help='Time interval for candles.'
    )
    parser.add_argument(
        '--model_type', type=str, choices=['random_forest', 'xgboost', 'lstm'], default='xgboost',
        help="Model type to train. Default: xgboost."
    )
    # ADDED: Command-line argument for user login
    parser.add_argument(
        '--user_login', type=str, default='Dimitri',
        help='The user login to record in the model metadata. Defaults to "Dimitri".'
    )
    parser.add_argument('--skip_tuning', action='store_true', help='Skip hyperparameter tuning.')
    parser.add_argument('--features', type=str, nargs='+', help='Optional list of feature names to use.')
    parser.add_argument('--enable_pca', action='store_true', help='Enable PCA dimensionality reduction.')
    parser.add_argument('--pca_components', type=lambda x: int(x) if x.isdigit() else float(x), default=None, help='Number of PCA components.')

    args = parser.parse_args()
    main_args = vars(args)
    main(**main_args)