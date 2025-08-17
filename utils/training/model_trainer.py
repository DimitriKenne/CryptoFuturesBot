# utils/training/model_trainer.py

import logging
from collections import Counter
from typing import Dict, Any, Optional, List, Tuple, TYPE_CHECKING, Union
from dataclasses import fields

import joblib
import numpy as np
import pandas as pd
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (balanced_accuracy_score, classification_report,
                             confusion_matrix, accuracy_score)

from pandas import Int8Dtype

from pathlib import Path
from datetime import datetime
import copy

# --- Add project root to Python path for imports ---
import sys
# Adjust PROJECT_ROOT to point to the main project directory,
# assuming utils/training/model_trainer.py is 3 levels deep from project root
# (your_project/utils/training/model_trainer.py)
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Import necessary items from params.py (the central AppConfig) and config schemas
try:
    from config.params import app_config, FLOAT_EPSILON
    # ModelConfig now explicitly has TF_AVAILABLE and tf attributes
    from config.model import ModelConfig, LSTMParams, RandomForestParams, XGBoostParams
    from utils.data_management.data_manager import DataManager

    # Import the new modular training utilities (relative imports since they are in the same folder now)
    from .preprocessor_builder import PreprocessorBuilder
    from .model_builder import ModelBuilder
    from .data_sequencer import DataSequencer

    # Access TF_AVAILABLE and tf from app_config.model's attributes
    TF_AVAILABLE = app_config.model.TF_AVAILABLE
    tf = app_config.model.tf # The tensorflow module
except ImportError as e:
    logging.error(f"Failed to import necessary modules for ModelTrainer: {e}")
    raise # Re-raise the exception to stop execution if essential imports fail


# Conditional import for type hinting if TYPE_CHECKING is True
if TYPE_CHECKING:
    from sklearn.base import BaseEstimator
    if TF_AVAILABLE:
        from tensorflow.keras.models import Model as KerasModel # type: ignore


# Get logger for this module
logger = logging.getLogger(__name__)


class ModelTrainer:
    """
    Handles training, evaluation, and saving of different trading models for ternary
    classification (-1, 0, 1). Supports scikit-learn compatible models (like
    RandomForest, XGBoost) and Keras LSTM models. Integrates modular
    preprocessing, model building, and data sequencing.
    Allows specifying a subset of features to use.
    Uses DataManager for saving and loading model artifacts.
    """

    def __init__(self, model_config: Optional[Union[ModelConfig, Dict[str, Any]]] = None):
        """
        Initializes the ModelTrainer with the model configuration.
        Delegates building of preprocessor, model, and data sequencing to dedicated classes.

        Args:
            model_config (Optional[Union[ModelConfig, Dict[str, Any]]]): A dictionary or ModelConfig
                                     instance containing the model configuration.
                                     If None, defaults to a deep copy of app_config.model.
                                     If a dictionary, it will be converted to ModelConfig.
        Raises:
            TypeError: If the config is not a ModelConfig instance or dictionary.
            ImportError: If TensorFlow is required for LSTM but not installed.
        """
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self.logger.info("ModelTrainer initializing...")

        # Convert the input config to a ModelConfig dataclass instance.
        if model_config is None:
            self._model_config: ModelConfig = copy.deepcopy(app_config.model)
        elif isinstance(model_config, dict):
            self._model_config: ModelConfig = ModelConfig(**copy.deepcopy(model_config))
        elif isinstance(model_config, ModelConfig):
            self._model_config: ModelConfig = copy.deepcopy(model_config)
        else:
            raise TypeError("Config must be a ModelConfig instance or a dictionary, not " + str(type(model_config)))

        self.model_type = self._model_config.model_type
        self.features_to_use = self._model_config.features_to_use

        # Access TF_AVAILABLE directly from the _model_config instance
        if self.model_type == 'lstm' and not self._model_config.TF_AVAILABLE:
            raise ImportError("TensorFlow is required for LSTM model but is not installed or available.")

        self.model: Optional[Union['BaseEstimator', 'KerasModel']] = None # type: ignore
        self.pipeline: Optional[Pipeline] = None
        self.preprocessor: Optional[ColumnTransformer] = None
        self.feature_columns_processed: Optional[List[str]] = None
        self.feature_columns_original: Optional[List[str]] = None

        self.label_map: Dict[int, int] = {-1: 0, 0: 1, 1: 2}
        self.inverse_label_map: Dict[int, int] = {0: -1, 1: 0, 2: 1}
        self.classes: np.ndarray = np.array([-1, 0, 1])

        # Initialize the modular utility builders
        self.preprocessor_builder = PreprocessorBuilder(
            scaler_type=self._model_config.scaler_type,
            pca_enabled=self._model_config.pca_enabled,
            pca_n_components=self._model_config.pca_n_components,
            features_to_use=self.features_to_use
        )
        # Pass tf_available and tf from the _model_config instance to ModelBuilder
        self.model_builder = ModelBuilder(tf_available=self._model_config.TF_AVAILABLE, tf_module=self._model_config.tf)

        if self.model_type == 'lstm':
            self.sequence_length = self._model_config.lstm_params.sequence_length_bars
            # Pass tf from the _model_config instance to DataSequencer
            self.data_sequencer = DataSequencer(sequence_length=self.sequence_length, tf_module=self._model_config.tf)
            self.logger.info(f"LSTM Sequence Length: {self.sequence_length}")
        else:
            self.sequence_length = 1
            self.data_sequencer = None

        self.dm = DataManager()
        # Do NOT log model type here; log it after loading correct config in load()


    def train(self, X_train: pd.DataFrame, y_train: pd.Series, X_val: Optional[pd.DataFrame] = None, y_val: Optional[pd.Series] = None):
        """
        Trains the model using the provided training data.
        Includes preprocessing and handling of class imbalance if configured.

        Args:
            X_train (pd.DataFrame): Training features.
            y_train (pd.Series): Training labels (-1, 0, 1).
            X_val (Optional[pd.DataFrame]): Validation features (for LSTM, should be cleaned of NaNs).
            y_val (Optional[pd.Series]): Validation labels (for LSTM, should be cleaned of NaNs).

        Raises:
            ValueError: If training data is empty or contains issues.
            RuntimeError: If training fails.
            ImportError: If TensorFlow is required for LSTM but not installed.
        """
        if X_train.empty or y_train.empty:
            raise ValueError("Training data is empty.")
        if len(X_train) != len(y_train):
            raise ValueError("Training features and labels have different lengths.")

        self.logger.info(f"Starting training for {self.model_type} model...")

        self.feature_columns_original = X_train.columns.tolist()
        self.logger.info(f"Original feature columns from training data: {self.feature_columns_original}")

        # Build and fit the preprocessor using PreprocessorBuilder
        self.preprocessor = self.preprocessor_builder.build_and_fit(X_train)
        self.feature_columns_processed = self.preprocessor_builder.processed_feature_names

        if not self.feature_columns_processed or len(self.feature_columns_processed) == 0:
            self.logger.critical("No features were selected or created by the preprocessor. Cannot train.")
            raise RuntimeError("No features selected by preprocessor.")


        # Access TF_AVAILABLE directly from the _model_config instance
        if self.model_type == 'lstm':
            if not self._model_config.TF_AVAILABLE:
                raise ImportError("TensorFlow is not installed. Cannot train LSTM model.")
            if self.data_sequencer is None:
                raise RuntimeError("DataSequencer not initialized for LSTM model.")

            lstm_params: LSTMParams = self._model_config.lstm_params
            n_features_after_prep = len(self.feature_columns_processed)

            # Build the model using ModelBuilder
            self.model = self.model_builder.build_model(
                model_type=self.model_type,
                n_features=n_features_after_prep,
                model_params=lstm_params,
                general_config_random_seed=app_config.general.random_seed,
                general_config_n_processors=app_config.general.n_processors
            )

            # Transform data using PreprocessorBuilder
            X_train_scaled = self.preprocessor_builder.transform(X_train)

            if np.isnan(X_train_scaled).any() or np.isinf(X_train_scaled).any():
                nan_count = np.isnan(X_train_scaled).sum()
                inf_count = np.isinf(X_train_scaled).sum()
                error_msg = f"Scaled training data contains NaN ({nan_count}) or Inf ({inf_count}) values. Cannot train LSTM."
                self.logger.critical(error_msg)
                raise ValueError(error_msg)
            self.logger.info("Scaled training data checked: No NaN or Inf values found.")

            y_train_mapped = y_train.map(self.label_map).values
            X_train_seq, y_train_seq_one_hot = self.data_sequencer.create_sequences(X_train_scaled, y_train_mapped)

            if X_train_seq.shape[0] == 0:
                self.logger.error("No training sequences generated for LSTM. Cannot train.")
                raise ValueError("No training sequences generated.")

            val_data = None
            if X_val is not None and y_val is not None and not X_val.empty and not y_val.empty:
                if len(X_val) != len(y_val):
                    self.logger.error("Validation features and labels have different lengths.")
                    raise ValueError("Validation features and labels have different lengths.")

                X_val_scaled = self.preprocessor_builder.transform(X_val)

                if np.isnan(X_val_scaled).any() or np.isinf(X_val_scaled).any():
                    nan_count = np.isnan(X_val_scaled).sum()
                    inf_count = np.isinf(X_val_scaled).sum()
                    error_msg = f"Scaled validation data contains NaN ({nan_count}) or Inf ({inf_count}) values. Cannot train LSTM with validation data."
                    self.logger.critical(error_msg)
                    raise ValueError(error_msg)
                self.logger.info("Scaled validation data checked: No NaN or Inf values found.")

                y_val_mapped = y_val.map(self.label_map).values
                X_val_seq, y_val_seq_one_hot = self.data_sequencer.create_sequences(X_val_scaled, y_val_mapped)

                if X_val_seq.shape[0] > 0:
                    val_data = (X_val_seq, y_val_seq_one_hot)
                    self.logger.info(f"Prepared external validation data for LSTM. Input shape: {X_val_seq.shape}, Labels shape: {y_val_seq_one_hot.shape}")
                else:
                    self.logger.warning("External validation sequences generated empty. LSTM training will proceed without external validation.")
            elif lstm_params.validation_split > 0 and lstm_params.validation_split < 1:
                self.logger.info(f"Using internal validation split of {lstm_params.validation_split} from training data for LSTM.")
                pass


            class_weight = None
            class_balancing_strategy = lstm_params.class_balancing
            if class_balancing_strategy == 'balanced':
                class_counts = Counter(np.argmax(y_train_seq_one_hot, axis=1))
                total_samples = len(y_train_seq_one_hot)
                if total_samples > 0:
                    n_classes = 3
                    class_weight = {cls_int: total_samples / (n_classes * count)
                                    for cls_int, count in class_counts.items() if count > 0}
                    self.logger.info(f"Calculated balanced class weights for LSTM: {class_weight}")
                else:
                    self.logger.warning("Cannot calculate class weights: No training samples after sequence creation.")

            elif isinstance(class_balancing_strategy, dict):
                class_weight = {self.label_map.get(orig_lbl, orig_lbl): weight
                                for orig_lbl, weight in class_balancing_strategy.items()}
                self.logger.info(f"Using provided class weights for LSTM: {class_weight}")
            elif class_balancing_strategy is not None:
                self.logger.warning(f"Unsupported 'class_balancing' strategy '{class_balancing_strategy}'. Skipping class weighting.")


            callbacks = []
            es_patience = lstm_params.early_stopping_patience
            if es_patience is not None and es_patience > 0:
                monitor_metric = 'val_loss' if val_data or (lstm_params.validation_split > 0 and lstm_params.validation_split < 1) else 'loss'
                callbacks.append(self._model_config.tf.keras.callbacks.EarlyStopping(monitor=monitor_metric, patience=es_patience, restore_best_weights=True)) # type: ignore
                self.logger.info(f"Added EarlyStopping with patience {es_patience} monitoring '{monitor_metric}'.")

            rlrop_factor = lstm_params.reduce_lr_on_plateau_factor
            rlrop_patience = lstm_params.reduce_lr_on_plateau_patience
            if rlrop_factor is not None and rlrop_patience is not None and rlrop_patience > 0:
                monitor_metric = 'val_loss' if val_data or (lstm_params.validation_split > 0 and lstm_params.validation_split < 1) else 'loss'
                callbacks.append(self._model_config.tf.keras.callbacks.ReduceLROnPlateau(monitor=monitor_metric, factor=rlrop_factor, patience=rlrop_patience)) # type: ignore
                self.logger.info(f"Added ReduceLROnPlateau with factor {rlrop_factor} and patience {rlrop_patience} monitoring '{monitor_metric}'.")

            epochs = lstm_params.epochs
            batch_size = lstm_params.batch_size
            validation_split_param = lstm_params.validation_split if val_data is None else 0.0


            self.logger.info(f"Training LSTM model for {epochs} epochs with batch size {batch_size}...")
            history = self.model.fit( # type: ignore
                X_train_seq,
                y_train_seq_one_hot,
                epochs=epochs,
                batch_size=batch_size,
                validation_data=val_data, # Use external val_data if present
                validation_split=validation_split_param, # Use internal split only if no external val_data
                class_weight=class_weight,
                callbacks=callbacks,
                verbose=1
            )
            self.logger.info("LSTM model training complete.")
            self.training_history = history.history


        else: # Scikit-learn compatible models (RandomForest, XGBoost)
            steps = [('preprocessor', self.preprocessor_builder.preprocessor)]

            if self.model_type == 'random_forest':
                model_params = self._model_config.random_forest_params
            elif self.model_type == 'xgboost':
                model_params = self._model_config.xgboost_params
            else:
                raise ValueError(f"Unsupported model type for training: {self.model_type}")

            # --- Handle early stopping for XGBoost ---
            fit_kwargs = {}
            if self.model_type == 'xgboost' and model_params.early_stopping_rounds is not None:
                if X_val is not None and not X_val.empty and y_val is not None and not y_val.empty:
                    # Transform validation data using the preprocessor
                    X_val_processed = self.preprocessor_builder.transform(X_val)
                    y_val_mapped = y_val.map(self.label_map)
                    fit_kwargs['eval_set'] = [(X_val_processed, y_val_mapped)]
                    fit_kwargs['early_stopping_rounds'] = model_params.early_stopping_rounds
                    self.logger.info(f"XGBoost early stopping enabled with eval_set. Patience: {model_params.early_stopping_rounds}")
                else:
                    self.logger.warning("XGBoost early stopping rounds specified, but no validation set (X_val, y_val) provided. Disabling early stopping for this run.")
                    # Temporarily disable early stopping for this specific fit if no validation data is provided
                    model_params.early_stopping_rounds = None # This will affect the current model_params object, but it's okay for this run.

            self.model = self.model_builder.build_model(
                model_type=self.model_type,
                model_params=model_params,
                general_config_random_seed=app_config.general.random_seed,
                general_config_n_processors=app_config.general.n_processors
            )

            class_balancing_strategy = model_params.class_balancing
            if class_balancing_strategy == 'undersampling':
                sampler = RandomUnderSampler(random_state=app_config.general.random_seed)
                steps.append(('sampler', sampler))
                self.logger.info("Added RandomUnderSampler to training pipeline.")
            elif class_balancing_strategy == 'oversampling':
                sampler = SMOTE(random_state=app_config.general.random_seed)
                steps.append(('sampler', sampler))
                self.logger.info("Added SMOTE to training pipeline.")
            elif isinstance(class_balancing_strategy, dict):
                sampler = SMOTE(random_state=app_config.general.random_seed, **class_balancing_strategy)
                steps.append(('sampler', sampler))
                self.logger.info(f"Added SMOTE to training pipeline with custom params: {class_balancing_strategy}")
            elif class_balancing_strategy is not None:
                self.logger.warning(f"Unsupported 'class_balancing' strategy '{class_balancing_strategy}'. Skipping sampler.")

            steps.append(('model', self.model))

            self.pipeline = Pipeline(steps)
            self.logger.info(f"Training pipeline created with steps: {[name for name, _ in self.pipeline.steps]}")

            self.logger.info(f"Training {self.model_type} pipeline...")
            y_train_mapped = y_train.map(self.label_map)
            # Pass fit_kwargs to the pipeline's fit method if it supports it (XGBoost does via __call__)
            # Or directly to the model inside the pipeline if accessing it.
            # For Pipeline, fit_params are passed as <step_name>__<param_name>
            pipeline_fit_params = {}
            if self.model_type == 'xgboost' and fit_kwargs:
                # Need to map eval_set to 'model__eval_set' for pipeline
                if 'eval_set' in fit_kwargs:
                    pipeline_fit_params['model__eval_set'] = fit_kwargs['eval_set']
                if 'early_stopping_rounds' in fit_kwargs: # This is handled by the model itself, not pipeline fit_params
                    # It's already set on the model directly via the model_builder.
                    pass


            self.pipeline.fit(X_train, y_train_mapped, **pipeline_fit_params) # Pass pipeline_fit_params
            self.logger.info(f"{self.model_type} pipeline training complete.")


    def evaluate(self, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, Any]:
        """
        Evaluates the trained model on the test data.

        Args:
            X_test (pd.DataFrame): Test features.
            y_test (pd.Series): Test labels (-1, 0, 1).

        Returns:
            Dict[str, Any]: Evaluation metrics.

        Raises:
            ValueError: If test data is empty or contains issues.
            RuntimeError: If evaluation fails or model/preprocessor is not available.
        """
        if X_test.empty or y_test.empty:
            raise ValueError("Test data is empty.")
        if len(X_test) != len(y_test):
            raise ValueError("Test features and labels have different lengths.")

        self.logger.info(f"Evaluating {self.model_type} model on test set...")

        y_test_mapped = y_test.map(self.label_map).values

        if self.model_type == 'lstm':
            if self.model is None or self.preprocessor_builder.preprocessor is None or self.data_sequencer is None:
                raise RuntimeError("LSTM model, preprocessor, or data sequencer not available for evaluation.")
            if self.sequence_length <= 0:
                raise ValueError(f"Invalid sequence_length for LSTM evaluation: {self.sequence_length}")

            self.logger.info("Preparing test data for LSTM evaluation...")
            X_test_scaled = self.preprocessor_builder.transform(X_test)

            if np.isnan(X_test_scaled).any() or np.isinf(X_test_scaled).any():
                nan_count = np.isnan(X_test_scaled).sum()
                inf_count = np.isinf(X_test_scaled).sum()
                error_msg = f"Scaled test data contains NaN ({nan_count}) or Inf ({inf_count}) values. Cannot evaluate LSTM."
                self.logger.critical(error_msg)
                raise ValueError(error_msg)

            X_test_seq, y_test_seq_one_hot_aligned = self.data_sequencer.create_sequences(X_test_scaled, y_test_mapped)

            if X_test_seq.shape[0] == 0:
                self.logger.warning("No test sequences generated for LSTM evaluation. Skipping evaluation.")
                return {"note": "No test sequences generated for evaluation."}

            self.logger.info("Evaluating Keras LSTM model on test sequences...")
            loss, accuracy = self.model.evaluate(X_test_seq, y_test_seq_one_hot_aligned, verbose=0) # type: ignore

            y_pred_proba = self.model.predict(X_test_seq) # type: ignore
            y_pred_mapped = np.argmax(y_pred_proba, axis=1)

            y_pred_original = pd.Series(y_pred_mapped).map(self.inverse_label_map).fillna(0).values.astype(int)

            y_test_aligned_original = y_test.iloc[self.sequence_length - 1:].values


            if len(y_pred_original) != len(y_test_aligned_original):
                self.logger.error(f"Mismatch in length between LSTM predictions ({len(y_pred_original)}) and aligned test labels ({len(y_test_aligned_original)}). Evaluation may be incorrect.")
                raise RuntimeError("LSTM prediction and label length mismatch during evaluation.")

            report = classification_report(y_test_aligned_original, y_pred_original, labels=self.classes, zero_division=0, output_dict=True)
            cm = confusion_matrix(y_test_aligned_original, y_pred_original, labels=self.classes)
            bal_acc = balanced_accuracy_score(y_test_aligned_original, y_pred_original)

            results = {
                "loss": loss,
                "accuracy": accuracy,
                "balanced_accuracy": bal_acc,
                "classification_report": report,
                "confusion_matrix": cm.tolist()
            }
            self.logger.info("LSTM model evaluation complete.")


        else: # Scikit-learn compatible models
            if self.pipeline is None:
                raise RuntimeError("Model pipeline not available for evaluation.")

            y_pred_mapped = self.pipeline.predict(X_test)

            y_pred_original = pd.Series(y_pred_mapped).map(self.inverse_label_map).fillna(0).values.astype(int)

            report = classification_report(y_test.values, y_pred_original, labels=self.classes, zero_division=0, output_dict=True)
            cm = confusion_matrix(y_test.values, y_pred_original, labels=self.classes)
            bal_acc = balanced_accuracy_score(y_test.values, y_pred_original)
            acc = accuracy_score(y_test.values, y_pred_original)

            results = {
                "overall_accuracy": acc,
                "balanced_accuracy": bal_acc,
                "classification_report": report,
                "confusion_matrix": cm.tolist()
            }
            self.logger.info(f"{self.model_type} model evaluation complete.")


        return results

    def predict(self, X: pd.DataFrame) -> pd.Series:
        """
        Makes predictions on new data using the trained model pipeline.

        Args:
            X (pd.DataFrame): New features data.

        Returns:
            pd.Series: Predicted labels (-1, 0, 1), aligned to the original index of X.
                       For LSTM, predictions are aligned to the end of each sequence.

        Raises:
            RuntimeError: If the model or preprocessor is not trained/loaded.
            ValueError: If input data is empty or has incorrect features.
        """
        if X.empty:
            self.logger.warning("Input data for prediction is empty. Returning empty Series.")
            return pd.Series(dtype=Int8Dtype())

        self.logger.info(f"Making predictions with {self.model_type} model...")

        if self.model_type == 'lstm':
            if self.model is None or self.preprocessor_builder.preprocessor is None or self.data_sequencer is None:
                raise RuntimeError("LSTM model, preprocessor, or data sequencer not available for prediction.")
            if self.sequence_length <= 0:
                raise ValueError(f"Invalid sequence_length for LSTM prediction: {self.sequence_length}")

            try:
                X_scaled = self.preprocessor_builder.transform(X)
            except ValueError as e:
                self.logger.error(f"Error transforming input data for LSTM prediction: {e}. Likely due to missing features.", exc_info=True)
                return pd.Series(np.nan, index=X.index, dtype=float).astype(Int8Dtype())
            except Exception as e:
                self.logger.error(f"An unexpected error occurred during data transformation for LSTM prediction: {e}", exc_info=True)
                return pd.Series(np.nan, index=X.index, dtype=float).astype(Int8Dtype())

            if np.isnan(X_scaled).any() or np.isinf(X_scaled).any():
                nan_count = np.isnan(X_scaled).sum()
                inf_count = np.isinf(X_scaled).sum()
                error_msg = f"Scaled LSTM prediction data contains NaN ({nan_count}) or Inf ({inf_count}) values after preprocessing. Cannot make prediction."
                self.logger.critical(error_msg)
                return pd.Series(np.nan, index=X.index, dtype=float).astype(Int8Dtype())


            dummy_y_mapped = np.zeros(X_scaled.shape[0], dtype=int)
            X_sequences, _ = self.data_sequencer.create_sequences(X_scaled, dummy_y_mapped)

            if X_sequences.shape[0] == 0:
                self.logger.warning("No sequences generated from input data for LSTM prediction. Returning empty predictions.")
                prediction_index = X.index[self.sequence_length - 1:] if self.sequence_length <= len(X.index) else X.index
                return pd.Series(np.nan, index=prediction_index, dtype=float).astype(Int8Dtype())


            self.logger.info("Making predictions with Keras LSTM model on sequences...")
            y_pred_proba_array = self.model.predict(X_sequences) # type: ignore
            y_pred_mapped = np.argmax(y_pred_proba_array, axis=1)

            y_pred_original_values = pd.Series(y_pred_mapped).map(self.inverse_label_map).fillna(0).values

            prediction_index = X.index[self.sequence_length - 1:]

            aligned_predictions = pd.Series(np.nan, index=X.index, dtype=float)

            if len(y_pred_original_values) == len(prediction_index):
                aligned_predictions.loc[prediction_index] = y_pred_original_values
                self.logger.info("LSTM predictions made and aligned.")
            else:
                self.logger.error(f"Mismatch in length between LSTM predictions ({len(y_pred_original_values)}) and aligned input index ({len(prediction_index)}). Prediction alignment failed.")
                self.logger.warning("Returning Series with NaNs due to alignment failure.")

            try:
                return aligned_predictions.astype(Int8Dtype())
            except Exception as e:
                self.logger.warning(f"Could not convert aligned predictions to Int8Dtype: {e}. Returning as float Series.")
                return aligned_predictions


        else: # Scikit-learn compatible models
            if self.pipeline is None:
                raise RuntimeError("Model pipeline is not loaded or trained. Cannot make predictions.")

            y_pred_mapped = self.pipeline.predict(X)

            y_pred_original = pd.Series(y_pred_mapped).map(self.inverse_label_map).fillna(0).values

            if len(y_pred_original) != len(X.index):
                self.logger.error(f"Prediction output length ({len(y_pred_original)}) does not match input length ({len(X.index)}). Prediction failed.")
                return pd.Series(np.nan, index=X.index, dtype=float).astype(Int8Dtype())

            y_pred_series = pd.Series(y_pred_original, index=X.index, dtype=Int8Dtype())

            self.logger.info(f"{self.model_type} predictions made.")
            return y_pred_series


    def predict_proba(self, X: pd.DataFrame) -> Optional[pd.DataFrame]:
        """
        Makes probability predictions on new data using the trained model.
        Returns None if the model does not support predict_proba or if prediction fails.

        Args:
            X (pd.DataFrame): New features data.

        Returns:
            Optional[pd.DataFrame]: Predicted probabilities (columns: -1, 0, 1),
                                    aligned to the original index of X.
                                    For LSTM, probabilities are aligned to the end
                                    of each sequence. Returns None if prediction fails
                                    or not supported.

        Raises:
            RuntimeError: If the model or preprocessor is not trained/loaded.
            ValueError: If input data is empty or has incorrect features.
        """
        if X.empty:
            self.logger.warning("Input data for probability prediction is empty. Returning None.")
            return None

        self.logger.info(f"Getting probability predictions with {self.model_type} model...")

        if self.model_type == 'lstm':
            if self.model is None or self.preprocessor_builder.preprocessor is None or self.data_sequencer is None:
                raise RuntimeError("LSTM model, preprocessor, or data sequencer not available for probability prediction.")
            if self.sequence_length <= 0:
                raise ValueError(f"Invalid sequence_length for LSTM probability prediction: {self.sequence_length}")

            try:
                X_scaled = self.preprocessor_builder.transform(X)
            except ValueError as e:
                self.logger.error(f"Error transforming input data for LSTM probability prediction: {e}. Likely due to missing features.", exc_info=True)
                return None
            except Exception as e:
                self.logger.error(f"An unexpected error occurred during data transformation for LSTM probability prediction: {e}", exc_info=True)
                return None

            if np.isnan(X_scaled).any() or np.isinf(X_scaled).any():
                nan_count = np.isnan(X_scaled).sum()
                inf_count = np.isinf(X_scaled).sum()
                error_msg = f"Scaled LSTM probability prediction data contains NaN ({nan_count}) or Inf ({inf_count}) values after preprocessing. Cannot make probability prediction."
                self.logger.critical(error_msg)
                return None


            dummy_y_mapped = np.zeros(X_scaled.shape[0], dtype=int)
            X_sequences, _ = self.data_sequencer.create_sequences(X_scaled, dummy_y_mapped)

            if X_sequences.shape[0] == 0:
                self.logger.warning("No sequences generated from input data for LSTM probability prediction. Returning None.")
                return None

            self.logger.info("Making probability predictions with Keras LSTM model on sequences...")
            y_pred_proba_array = self.model.predict(X_sequences) # type: ignore

            self.logger.info("LSTM probability predictions made.")
            prediction_index = X.index[self.sequence_length - 1:]

            aligned_probabilities_df = pd.DataFrame(np.nan, index=X.index, columns=['proba_-1', 'proba_0', 'proba_1'], dtype=float)
            if len(y_pred_proba_array) == len(prediction_index):
                aligned_probabilities_df.loc[prediction_index, :] = y_pred_proba_array
                self.logger.debug("LSTM probability predictions aligned.")
            else:
                self.logger.error(f"Mismatch in length between LSTM probability predictions ({len(y_pred_proba_array)}) and aligned input index ({len(prediction_index)}). Probability alignment failed.")
                self.logger.warning("Returning None due to alignment failure.")
                return None

            return aligned_probabilities_df


        else: # Scikit-learn compatible models
            if self.pipeline is None:
                raise RuntimeError("Model pipeline is not loaded or trained. Cannot make probability predictions.")

            if hasattr(self.pipeline, 'predict_proba'):
                y_pred_proba_array = self.pipeline.predict_proba(X)
                self.logger.info(f"{self.model_type} probability predictions made.")
                if len(y_pred_proba_array) != len(X.index):
                    self.logger.error(f"Probability prediction output length ({len(y_pred_proba_array)}) does not match input length ({len(X.index)}). Prediction failed.")
                    return None

                return pd.DataFrame(y_pred_proba_array, index=X.index, columns=['proba_-1', 'proba_0', 'proba_1'])
            else:
                self.logger.info(f"Model type '{self.model_type}' or its pipeline does not support predict_proba.")
                return None


    def save(self, symbol: str, interval: str, model_key: str):
        """
        Saves the trained model (pipeline or Keras model) and its metadata using DataManager.
        """
        if self.model is None and self.pipeline is None:
            raise RuntimeError("No model or pipeline trained/loaded to save.")
        if self.preprocessor_builder.preprocessor is None:
            raise RuntimeError("Preprocessor is not trained/loaded. Cannot save model.")

        self.logger.info(f"Saving trained {self.model_type} model and metadata for {symbol.upper()} {interval} using DataManager...")

        # Filter out non-init fields from ModelConfig before saving
        model_config_fields = set(f.name for f in fields(ModelConfig) if f.init)
        filtered_model_config_dict = {k: v for k, v in self._model_config.__dict__.items() if k in model_config_fields}

        metadata = {
            'model_type': self.model_type,
            'feature_columns_processed': self.feature_columns_processed,
            'feature_columns_original': self.feature_columns_original,
            'label_map': self.label_map,
            'inverse_label_map': self.inverse_label_map,
            'classes': self.classes.tolist(),
            'model_config': filtered_model_config_dict,  # Only save valid fields
            'save_timestamp': datetime.now().isoformat(),
        }

        try:
            self.dm.save_model_artifact(
                artifact=metadata,
                symbol=symbol,
                interval=interval,
                model_key=model_key,
                artifact_type='metadata'
            )
            self.logger.info("Model metadata saved successfully via DataManager.")
        except Exception as e:
            self.logger.error(f"Failed to save model metadata via DataManager: {e}", exc_info=True)
            self.logger.warning("Failed to save model metadata. Model might not be loadable correctly.")

        try:
            if self.model_type == 'lstm':
                self.dm.save_model_artifact(
                    artifact=self.model,
                    symbol=symbol,
                    interval=interval,
                    model_key=model_key,
                    artifact_type='model'
                )
                self.logger.info(f"Keras LSTM model saved successfully via DataManager.")

                self.dm.save_model_artifact(
                    artifact=self.preprocessor_builder.preprocessor, # Save preprocessor from builder
                    symbol=symbol,
                    interval=interval,
                    model_key=model_key,
                    artifact_type='preprocessor'
                )
                self.logger.info(f"Preprocessor saved successfully via DataManager.")


            else: # Scikit-learn compatible models (pipeline)
                self.dm.save_model_artifact(
                    artifact=self.pipeline,
                    symbol=symbol,
                    interval=interval,
                    model_key=model_key,
                    artifact_type='pipeline'
                )
                self.logger.info(f"{self.model_type} pipeline saved successfully via DataManager.")
                self.logger.debug("Preprocessor is part of the scikit-learn pipeline, not saved separately when saving the pipeline.")


        except Exception as e:
            self.logger.error(f"Failed to save model artifact(s) via DataManager: {e}", exc_info=True)
            raise OSError(f"Failed to save model artifact(s): {e}")


        self.logger.info(f"Model saving process completed for {self.model_type}.")


    def load(self, symbol: str, interval: str, model_key: str) -> 'ModelTrainer':
        """
        Loads a trained model (pipeline or Keras model) and its metadata using DataManager.
        Updates the current ModelTrainer instance with the loaded components.

        Args:
            symbol (str): Trading pair symbol.
            interval (str): Time interval.
            model_key (str): Key for the model configuration (e.g., 'xgboost', 'random_forest', 'lstm').

        Returns:
            ModelTrainer: The current instance, updated with the loaded model and metadata.

        Raises:
            FileNotFoundError: If the model or metadata file is not found (raised by DataManager).
            RuntimeError: If loading fails for other reasons.
            ValueError: If metadata is missing crucial info.
            ImportError: If TensorFlow is required but not available.
        """
        self.logger.info(f"Loading trained model for {symbol.upper()} {interval} using DataManager...")

        self.logger.info(f"Loading metadata for model... from {symbol.upper()} {interval} {model_key}")
        try:
            metadata_dict = self.dm.load_model_artifact(
                symbol=symbol,
                interval=interval,
                model_key=model_key,
                artifact_type='metadata'
            )
            self.logger.info("Metadata loaded successfully via DataManager.")

            # Reconstruct ModelConfig from loaded metadata
            if 'model_config' in metadata_dict and isinstance(metadata_dict['model_config'], dict):
                loaded_model_config_dict = metadata_dict['model_config']

                # Filter out non-init fields before reconstructing ModelConfig
                model_config_fields = set(f.name for f in fields(ModelConfig) if f.init)
                loaded_model_config_dict = {k: v for k, v in loaded_model_config_dict.items() if k in model_config_fields}

                # Manually reconstruct nested dataclasses
                from config.model import (
                    LSTMParams, RandomForestParams,
                    XGBoostParams, XGBoostTuningParams, RandomForestTuningParams
                )

                if 'lstm_params' in loaded_model_config_dict and isinstance(loaded_model_config_dict['lstm_params'], dict):
                    loaded_model_config_dict['lstm_params'] = LSTMParams(**loaded_model_config_dict['lstm_params'])
                if 'random_forest_params' in loaded_model_config_dict and isinstance(loaded_model_config_dict['random_forest_params'], dict):
                    loaded_model_config_dict['random_forest_params'] = RandomForestParams(**loaded_model_config_dict['random_forest_params'])
                if 'xgboost_params' in loaded_model_config_dict and isinstance(loaded_model_config_dict['xgboost_params'], dict):
                    loaded_model_config_dict['xgboost_params'] = XGBoostParams(**loaded_model_config_dict['xgboost_params'])

                if 'xgboost_tuning_params' in loaded_model_config_dict and isinstance(loaded_model_config_dict['xgboost_tuning_params'], dict):
                    loaded_model_config_dict['xgboost_tuning_params'] = XGBoostTuningParams(**loaded_model_config_dict['xgboost_tuning_params'])
                if 'random_forest_tuning_params' in loaded_model_config_dict and isinstance(loaded_model_config_dict['random_forest_tuning_params'], dict):
                    loaded_model_config_dict['random_forest_tuning_params'] = RandomForestTuningParams(**loaded_model_config_dict['random_forest_tuning_params'])

                self._model_config = ModelConfig(**loaded_model_config_dict)
                self.logger.info("ModelConfig reconstructed from loaded metadata.")

                # Update ModelTrainer attributes from the loaded _model_config
                self.model_type = self._model_config.model_type
                self.features_to_use = self._model_config.features_to_use
                # Re-initialize preprocessor and data sequencer builders based on loaded config
                self.preprocessor_builder = PreprocessorBuilder(
                    scaler_type=self._model_config.scaler_type,
                    pca_enabled=self._model_config.pca_enabled,
                    pca_n_components=self._model_config.pca_n_components,
                    features_to_use=self.features_to_use
                )
                # Use the tf_available and tf from the _model_config instance
                self.model_builder = ModelBuilder(tf_available=self._model_config.TF_AVAILABLE, tf_module=self._model_config.tf)

                if self.model_type == 'lstm':
                    self.sequence_length = self._model_config.lstm_params.sequence_length_bars
                    # Use the tf from the _model_config instance
                    self.data_sequencer = DataSequencer(sequence_length=self.sequence_length, tf_module=self._model_config.tf)
                else:
                    self.sequence_length = 1
                    self.data_sequencer = None
            else:
                self.logger.warning("ModelConfig not found in metadata. Attempting to use default or initialized config for structure.")
                # Fallback to direct attribute setting if ModelConfig not explicitly saved
                self.model_type = metadata_dict.get('model_type', self.model_type)
                self.features_to_use = metadata_dict.get('features_to_use', self.features_to_use)
                self.sequence_length = metadata_dict.get('sequence_length_bars', metadata_dict.get('sequence_length', self.sequence_length))
                # Note: PreprocessorBuilder and DataSequencer will retain their initial defaults if ModelConfig isn't loaded properly here.

            # Update other instance attributes from metadata
            self.feature_columns_processed = metadata_dict.get('feature_columns_processed')
            self.feature_columns_original = metadata_dict.get('feature_columns_original')
            self.label_map = metadata_dict.get('label_map', self.label_map)
            self.inverse_label_map = metadata_dict.get('inverse_label_map', self.inverse_label_map)
            self.classes = np.array(metadata_dict.get('classes', [-1, 0, 1]))

            if self.feature_columns_processed is None:
                self.logger.warning("Processed feature columns not found in metadata.")
            if self.feature_columns_original is None:
                self.logger.warning("Original feature columns not found in metadata.")
            if self.label_map is None or self.inverse_label_map is None or self.classes is None or len(self.classes) == 0:
                self.logger.warning(f"Label mapping or classes not found or empty in metadata. Using defaults: labels={self.classes}, map={self.label_map}, inverse={self.inverse_label_map}.")

        except FileNotFoundError:
            raise
        except Exception as e:
            self.logger.error(f"Error loading model metadata via DataManager: {e}", exc_info=True)
            raise RuntimeError(f"Failed to load model metadata: {e}")


        # Log the correct model type after loading config from metadata
        self.logger.info(f"ModelTrainer loaded for model type: {self.model_type}")

        # --- Load Model / Pipeline using DataManager ---
        self.logger.info(f"Loading {self.model_type} model artifact...")
        try:
            if self.model_type == 'lstm':
                # Use the TF_AVAILABLE from the _model_config instance
                if not self._model_config.TF_AVAILABLE:
                    raise ImportError("TensorFlow is required to load LSTM model but is not installed.")

                self.model = self.dm.load_model_artifact(
                    symbol=symbol,
                    interval=interval,
                    model_key=model_key,
                    artifact_type='model'
                )
                self.logger.info("Keras LSTM model loaded successfully via DataManager.")

                self.logger.info(f"Attempting to load preprocessor for {self.model_type} model...")
                try:
                    loaded_preprocessor = self.dm.load_model_artifact(
                        symbol=symbol,
                        interval=interval,
                        model_key=model_key,
                        artifact_type='preprocessor'
                    )
                    self.preprocessor_builder.preprocessor = loaded_preprocessor
                    self.logger.info("Preprocessor loaded successfully via DataManager.")
                    if self.feature_columns_processed is None:
                        if hasattr(self.preprocessor_builder.preprocessor, 'get_feature_names_out'):
                            self.feature_columns_processed = self.preprocessor_builder.preprocessor.get_feature_names_out().tolist()
                            self.logger.info(f"Inferred processed feature columns from loaded preprocessor: {self.feature_columns_processed}")
                        elif hasattr(self.preprocessor_builder.preprocessor, 'feature_names_in_'):
                            self.feature_columns_processed = self.preprocessor_builder.preprocessor.feature_names_in_.tolist()
                            self.logger.info(f"Inferred processed feature columns from preprocessor.feature_names_in_: {self.feature_columns_processed}")
                        else:
                            self.logger.warning("Could not infer processed feature columns from preprocessor.")

                except FileNotFoundError:
                    self.logger.warning("Preprocessor artifact not found. LSTM predictions/evaluation might fail.")
                    self.preprocessor_builder.preprocessor = None
                except Exception as e:
                    self.logger.warning(f"Error loading preprocessor via DataManager: {e}", exc_info=True)
                    self.preprocessor_builder.preprocessor = None


            else: # Load Scikit-learn pipeline using DataManager
                self.pipeline = self.dm.load_model_artifact(
                    symbol=symbol,
                    interval=interval,
                    model_key=model_key,
                    artifact_type='pipeline'
                )
                self.logger.info(f"{self.model_type} pipeline loaded successfully via DataManager.")
                if self.pipeline is not None and len(self.pipeline.steps) > 0:
                    self.preprocessor_builder.preprocessor = self.pipeline.steps[0][1]
                    self.model = self.pipeline.steps[-1][1]

                    if self.feature_columns_processed is None:
                        if hasattr(self.preprocessor_builder.preprocessor, 'get_feature_names_out'):
                            self.feature_columns_processed = self.preprocessor_builder.preprocessor.get_feature_names_out().tolist()
                            self.logger.info(f"Inferred processed feature columns from loaded preprocessor: {self.feature_columns_processed}")
                        elif hasattr(self.preprocessor_builder.preprocessor, 'feature_names_in_'):
                            self.feature_columns_processed = self.preprocessor_builder.preprocessor.feature_names_in_.tolist()
                            self.logger.info(f"Inferred processed feature columns from preprocessor.feature_names_in_: {self.feature_columns_processed}")
                        else:
                            self.logger.warning("Could not infer processed feature columns from preprocessor.")
                else:
                    self.logger.warning("Loaded pipeline has no steps. Preprocessor and model not extracted.")
                    self.preprocessor_builder.preprocessor = None
                    self.model = None

        except FileNotFoundError:
            raise
        except Exception as e:
            self.logger.error(f"An unexpected error occurred during model artifact loading via DataManager: {e}", exc_info=True)
            raise RuntimeError(f"Failed to load trained model artifact: {e}")

        # Final check for essential components after loading
        if self.preprocessor_builder.preprocessor is None:
            self.logger.error("Preprocessor could not be loaded. Model might not work correctly for prediction/analysis.")
        if self.model is None and self.pipeline is None:
            self.logger.error("Model or pipeline could not be loaded. Model is not usable.")
        if self.feature_columns_original is None:
            self.logger.error("Original feature columns could not be loaded. Cannot prepare data for analysis.")

        self.logger.info(f"ModelTrainer instance loaded successfully for {self.model_type}.")
        return self
