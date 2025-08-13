# utils/model_trainer.py

import logging
from collections import Counter
from typing import Dict, Any, Optional, List, Tuple, TYPE_CHECKING, Union

import joblib
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (balanced_accuracy_score, classification_report,
                             confusion_matrix, accuracy_score)
from sklearn.model_selection import TimeSeriesSplit # Keep TimeSeriesSplit for potential CV within trainer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA # Import PCA

from pandas import Int8Dtype

from xgboost import XGBClassifier

from pathlib import Path
from datetime import datetime
import copy # For deep copying config

# --- Add project root to Python path for imports ---
import sys
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Import necessary items from params.py and config schemas
try:
    from config.params import (
        DEFAULT_MODEL_CONFIG,
        FLOAT_EPSILON,
        LSTM_AVAILABLE, # Import LSTM_AVAILABLE from params.py
        tf, # Import tensorflow if available (or None)
    )
    from config.model_config_schema import ModelConfig, LSTMParams, RandomForestParams, XGBoostParams
    from utils.data_manager import DataManager
except ImportError as e:
    logging.error(f"Failed to import necessary modules for ModelTrainer: {e}")
    raise # Re-raise the exception to stop execution if essential imports fail


# Conditional import for type hinting if TYPE_CHECKING is True
if TYPE_CHECKING:
    from sklearn.base import BaseEstimator
    if tf is not None:
        from tensorflow.keras.models import Model as KerasModel # type: ignore


# Get logger for this module
logger = logging.getLogger(__name__)


class ModelTrainer:
    """
    Handles training, evaluation, and saving of different trading models for ternary
    classification (-1, 0, 1). Supports scikit-learn compatible models (like
    RandomForest, XGBoost) and Keras LSTM models. Includes preprocessing,
    handling of class imbalance, and metadata management.
    Allows specifying a subset of features to use.
    Uses DataManager for saving and loading model artifacts.
    """

    def __init__(self, config: Optional[Union[ModelConfig, Dict[str, Any]]] = None):
        """
        Initializes the ModelTrainer with the model configuration.
        Does NOT build the model here; model is built during train() or loaded during load().

        Args:
            config (Optional[Union[ModelConfig, Dict[str, Any]]]): A dictionary or ModelConfig
                                     instance containing the model configuration.
                                     If None, defaults to a deep copy of DEFAULT_MODEL_CONFIG.
                                     If a dictionary, it will be converted to ModelConfig.
        Raises:
            ValueError: If the config is invalid or the model_type is unsupported.
            ImportError: If TensorFlow is required for LSTM but not installed.
        """
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self.logger.info("ModelTrainer initializing...")

        # Convert the input config to a ModelConfig dataclass instance.
        if config is None:
            self._model_config: ModelConfig = copy.deepcopy(DEFAULT_MODEL_CONFIG)
        elif isinstance(config, dict):
            # ModelConfig's __post_init__ handles nested dicts for dimensionality_reduction and model params
            self._model_config: ModelConfig = ModelConfig(**copy.deepcopy(config))
        elif isinstance(config, ModelConfig):
            self._model_config: ModelConfig = copy.deepcopy(config)
        else:
            raise TypeError("Config must be a ModelConfig instance or a dictionary, not " + str(type(config)))

        # Direct access to configuration parameters via the ModelConfig object
        self.model_type = self._model_config.model_type
        self.features_to_use: Optional[List[str]] = self._model_config.features_to_use

        # PCA Configuration from ModelConfig
        self.pca_enabled = self._model_config.dimensionality_reduction.enabled
        self.pca_method = self._model_config.dimensionality_reduction.method
        self.pca_params = self._model_config.dimensionality_reduction.params

        if self.pca_enabled and self.pca_method != 'pca':
            self.logger.warning(f"Unsupported PCA method: {self.pca_method}. Only 'pca' is supported. Disabling PCA.")
            self.pca_enabled = False # Disable if method is not supported

        if self.model_type == 'LSTM' and not LSTM_AVAILABLE:
            raise ImportError("TensorFlow is required for LSTM model but is not installed or available.")

        # Initialize model and pipeline as None
        self.model: Optional[Union['BaseEstimator', 'KerasModel']] = None # type: ignore
        self.pipeline: Optional[Pipeline] = None
        self.preprocessor: Optional[ColumnTransformer] = None
        self.feature_columns_processed: Optional[List[str]] = None
        self.feature_columns_original: Optional[List[str]] = None

        # Define label mapping for ternary classification (-1, 0, 1) to integers (0, 1, 2)
        self.label_map: Dict[int, int] = {-1: 0, 0: 1, 1: 2}
        self.inverse_label_map: Dict[int, int] = {0: -1, 1: 0, 2: 1}
        self.classes: np.ndarray = np.array([-1, 0, 1])

        # Get sequence length for LSTM (default to 1 for non-LSTM models)
        # This is now specific to LSTM in its params
        self.sequence_length: int = self._model_config.lstm_params.sequence_length_bars if self.model_type == 'LSTM' else 1

        self.logger.info(f"ModelTrainer initialized for model type: {self.model_type}")
        if self.model_type == 'LSTM':
            self.logger.info(f"LSTM Sequence Length: {self.sequence_length}")
            self.logger.debug(f"LSTM parameters: {self._model_config.lstm_params}")
        elif self.model_type == 'RandomForest':
            self.logger.debug(f"RandomForest parameters: {self._model_config.random_forest_params}")
        elif self.model_type == 'XGBoost':
            self.logger.debug(f"XGBoost parameters: {self._model_config.xgboost_params}")

        if self.features_to_use is not None:
            self.logger.info(f"Using specified feature subset: {self.features_to_use}")
        else:
            self.logger.info("Using all available features from input data.")
        
        if self.pca_enabled:
            self.logger.info(f"PCA enabled with method: {self.pca_method}, params: {self.pca_params}")

        # Instantiate DataManager here for use in save/load methods
        self.dm = DataManager()
        self.logger.debug("DataManager instance created.")


    def _build_lstm_model(self, n_features: int) -> 'KerasModel': # type: ignore
        """
        Builds the Keras LSTM model based on model_params and the number of features.

        Args:
            n_features (int): The number of features per timestep *after* preprocessing.

        Returns:
            tf.keras.Model: The built Keras Sequential model.
        """
        if not LSTM_AVAILABLE or tf is None:
            raise ImportError("TensorFlow is not available to build LSTM model.")

        # Get LSTM specific parameters from the ModelConfig's lstm_params
        lstm_params: LSTMParams = self._model_config.lstm_params
        sequence_length = lstm_params.sequence_length_bars
        n_layers = lstm_params.n_layers
        units_per_layer = lstm_params.units_per_layer
        dropout_rate = lstm_params.dropout_rate
        learning_rate = lstm_params.learning_rate
        clipnorm = lstm_params.clipnorm
        clipvalue = lstm_params.clipvalue


        if sequence_length <= 0:
            raise ValueError("LSTM model parameter 'sequence_length_bars' must be positive.")
        if n_features <= 0:
            raise ValueError("Number of features (n_features) must be positive to build LSTM model.")
        if n_layers <= 0:
            raise ValueError("Number of LSTM layers (n_layers) must be positive.")
        if units_per_layer <= 0:
            raise ValueError("Units per LSTM layer (units_per_layer) must be positive.")
        if not (0.0 <= dropout_rate <= 1.0):
            raise ValueError("Dropout rate must be between 0.0 and 1.0.")
        if learning_rate <= 0:
            raise ValueError("Learning rate must be positive.")


        model = tf.keras.models.Sequential()
        # Input layer expects shape (sequence_length, n_features)
        model.add(tf.keras.layers.Input(shape=(sequence_length, n_features)))

        # Add LSTM layers
        for i in range(n_layers):
            # Return sequences for all but the last LSTM layer
            return_sequences = i < n_layers - 1
            model.add(tf.keras.layers.LSTM(units_per_layer, return_sequences=return_sequences))
            if dropout_rate > 0:
                model.add(tf.keras.layers.Dropout(dropout_rate))

        # Output layer for ternary classification (3 classes)
        model.add(tf.keras.layers.Dense(3, activation='softmax'))

        # Configure optimizer with potential gradient clipping
        optimizer_params = {'learning_rate': learning_rate}
        if clipnorm is not None:
            optimizer_params['clipnorm'] = clipnorm
            self.logger.info(f"Using gradient clipping (clipnorm={clipnorm}) in Adam optimizer.")
        if clipvalue is not None:
            optimizer_params['clipvalue'] = clipvalue
            self.logger.info(f"Using gradient clipping (clipvalue={clipvalue}) in Adam optimizer.")

        optimizer = tf.keras.optimizers.Adam(**optimizer_params) # type: ignore

        # Compile the model
        model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

        self.logger.info("LSTM model architecture built and compiled.")
        # Log model summary
        model.summary(print_fn=lambda x: self.logger.info(x)) # type: ignore
        return model

    def _create_preprocessor(self, X: pd.DataFrame, feature_subset: Optional[List[str]] = None) -> ColumnTransformer:
        """
        Creates and fits a ColumnTransformer for preprocessing.
        Applies StandardScaler to all numeric features in the specified subset or all numeric features.
        Optionally includes a PCA step after StandardScaler if PCA is enabled in config.

        Args:
            X (pd.DataFrame): The input DataFrame containing features.
            feature_subset (Optional[List[str]]): A list of feature column names to use.
                                                  If None, all numeric columns in X are used.

        Returns:
            ColumnTransformer: The fitted preprocessor.

        Raises:
            ValueError: If specified features are not found in X.
        """
        # If a feature subset is provided, select only those columns
        if feature_subset is not None:
            # Check if all requested features exist in the DataFrame
            missing_features = [feat for feat in feature_subset if feat not in X.columns]
            if missing_features:
                error_msg = f"Specified features not found in input data for preprocessor: {missing_features}"
                self.logger.error(error_msg)
                raise ValueError(error_msg)
            X_subset = X[feature_subset].copy()
            self.logger.info(f"Preprocessor will be fitted on the specified feature subset: {feature_subset}")
        else:
            # If no subset is provided, use all numeric columns in the input X
            X_subset = X.select_dtypes(include=np.number).copy()
            self.logger.info("Preprocessor will be fitted on all numeric features in the input data.")


        # Select numeric features from the (potentially subsetted) DataFrame
        numeric_features = X_subset.select_dtypes(include=np.number).columns.tolist()

        if not numeric_features:
            self.logger.warning("No numeric features found in the input DataFrame (or subset) for preprocessing.")
            # Return an identity transformer if no numeric features to scale
            preprocessor = ColumnTransformer(transformers=[], remainder='passthrough')
            self.feature_columns_processed = []
        else:
            self.logger.info(f"Applying StandardScaler to numeric features: {numeric_features}")

            # Create a pipeline for numeric features: StandardScaler -> (Optional) PCA
            numeric_transformer_steps = [('scaler', StandardScaler())]

            if self.pca_enabled and self.pca_method == 'pca':
                self.logger.info(f"Adding PCA step with params: {self.pca_params}")
                # Initialize PCA with parameters from config
                pca = PCA(**self.pca_params)
                numeric_transformer_steps.append(('pca', pca))

            numeric_transformer = Pipeline(steps=numeric_transformer_steps)

            preprocessor = ColumnTransformer(
                transformers=[
                    ('num', numeric_transformer, numeric_features)
                ],
                remainder='passthrough'
            )

            self.logger.info("Fitting preprocessor...")
            preprocessor.fit(X_subset[numeric_features])
            self.logger.info("Preprocessor fitted.")

            try:
                self.feature_columns_processed = preprocessor.get_feature_names_out().tolist()
                self.logger.info(f"Feature columns after preprocessing: {self.feature_columns_processed}")

                if self.pca_enabled and self.pca_method == 'pca':
                    fitted_pca = preprocessor.named_transformers_['num'].named_steps['pca']
                    actual_n_components = fitted_pca.n_components_
                    self.logger.info(f"PCA reduced features to {actual_n_components} components.")
                    self.logger.info(f"Explained variance ratio: {fitted_pca.explained_variance_ratio_.sum():.4f}")

            except AttributeError:
                self.logger.warning("get_feature_names_out not available. Assuming feature columns are the original numeric features.")
                self.feature_columns_processed = numeric_features


        return preprocessor


    def _prepare_lstm_sequences(self, X_scaled: np.ndarray, y_mapped: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepares data into sequences for LSTM training/prediction.

        Args:
            X_scaled (np.ndarray): Scaled feature data (numpy array) *containing only the features
                                   that the model expects*.
            y_mapped (np.ndarray): Mapped label data (numpy array, 0, 1, or 2).

        Returns:
            Tuple[np.ndarray, np.ndarray]: Tuple containing:
                - X_sequences (np.ndarray): Feature sequences for LSTM input.
                - y_sequences_one_hot (np.ndarray): One-hot encoded labels for the end of each sequence.
        """
        sequence_length = self.sequence_length
        if sequence_length <= 0:
            raise ValueError("Sequence length must be a positive integer.")

        n_samples = X_scaled.shape[0]
        n_features = X_scaled.shape[1]

        if n_samples < sequence_length:
            self.logger.warning(f"Not enough data points ({n_samples}) to create sequences of length {sequence_length}. Returning empty arrays.")
            return np.empty((0, sequence_length, n_features)), np.empty((0, 3)) # 3 classes for one-hot

        X_sequences = []
        y_sequences = []

        for i in range(sequence_length - 1, n_samples):
            X_sequences.append(X_scaled[i - sequence_length + 1 : i + 1])
            y_sequences.append(y_mapped[i])

        X_sequences = np.array(X_sequences)
        y_sequences = np.array(y_sequences)

        y_sequences_one_hot = tf.keras.utils.to_categorical(y_sequences, num_classes=3) # type: ignore

        self.logger.info(f"Prepared {len(X_sequences)} LSTM sequences with shape {X_sequences.shape}")
        self.logger.info(f"Prepared {len(y_sequences_one_hot)} LSTM labels with shape {y_sequences_one_hot.shape}")

        return X_sequences, y_sequences_one_hot


    def train(self, X_train: pd.DataFrame, y_train: pd.Series, X_val: Optional[pd.DataFrame] = None, y_val: Optional[pd.Series] = None):
        """
        Trains the model using the provided training data.
        Includes preprocessing and handling of class imbalance if configured.
        Builds the model architecture here if it's an LSTM model.

        Args:
            X_train (pd.DataFrame): Training features (should be cleaned of NaNs in calling script).
                                    This DataFrame should contain *all* original features
                                    from the processed data before subsetting.
            y_train (pd.Series): Training labels (-1, 0, 1) (should be cleaned of NaNs in calling script).
            X_val (Optional[pd.DataFrame]): Validation features (for LSTM, should be cleaned of NaNs).
                                            This DataFrame should contain *all* original features
                                            from the processed data before subsetting.
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

        self.preprocessor = self._create_preprocessor(X_train, feature_subset=self.features_to_use)

        if not self.feature_columns_processed or len(self.feature_columns_processed) == 0:
            self.logger.critical("No features were selected or created by the preprocessor. Cannot train.")
            raise RuntimeError("No features selected by preprocessor.")


        if self.model_type == 'LSTM':
            if not LSTM_AVAILABLE:
                raise ImportError("TensorFlow is not installed. Cannot train LSTM model.")

            lstm_params: LSTMParams = self._model_config.lstm_params
            n_features_after_prep = len(self.feature_columns_processed)

            self.model = self._build_lstm_model(n_features=n_features_after_prep)

            X_train_scaled = self.preprocessor.transform(X_train)

            if np.isnan(X_train_scaled).any() or np.isinf(X_train_scaled).any():
                nan_count = np.isnan(X_train_scaled).sum()
                inf_count = np.isinf(X_train_scaled).sum()
                error_msg = f"Scaled training data contains NaN ({nan_count}) or Inf ({inf_count}) values. Cannot train LSTM."
                self.logger.critical(error_msg)
                raise ValueError(error_msg)
            self.logger.info("Scaled training data checked: No NaN or Inf values found.")

            y_train_mapped = y_train.map(self.label_map).values
            X_train_seq, y_train_seq_one_hot = self._prepare_lstm_sequences(X_train_scaled, y_train_mapped)

            if X_train_seq.shape[0] == 0:
                self.logger.error("No training sequences generated for LSTM. Cannot train.")
                raise ValueError("No training sequences generated.")

            val_data = None
            if X_val is not None and y_val is not None and not X_val.empty and not y_val.empty:
                if len(X_val) != len(y_val):
                    self.logger.error("Validation features and labels have different lengths.")
                    raise ValueError("Validation features and labels have different lengths.")

                X_val_scaled = self.preprocessor.transform(X_val)

                if np.isnan(X_val_scaled).any() or np.isinf(X_val_scaled).any():
                    nan_count = np.isnan(X_val_scaled).sum()
                    inf_count = np.isinf(X_val_scaled).sum()
                    error_msg = f"Scaled validation data contains NaN ({nan_count}) or Inf ({inf_count}) values. Cannot train LSTM with validation data."
                    self.logger.critical(error_msg)
                    raise ValueError(error_msg)
                self.logger.info("Scaled validation data checked: No NaN or Inf values found.")

                y_val_mapped = y_val.map(self.label_map).values
                X_val_seq, y_val_seq_one_hot = self._prepare_lstm_sequences(X_val_scaled, y_val_mapped)

                if X_val_seq.shape[0] > 0:
                    val_data = (X_val_seq, y_val_seq_one_hot)
                    self.logger.info(f"Prepared validation data for LSTM. Input shape: {X_val_seq.shape}, Labels shape: {y_val_seq_one_hot.shape}")
                else:
                    self.logger.warning("No validation sequences generated. LSTM training will proceed without validation.")

            class_weight = None
            class_weight_param = lstm_params.class_weight
            if class_weight_param == 'balanced':
                class_counts = Counter(np.argmax(y_train_seq_one_hot, axis=1))
                total_samples = len(y_train_seq_one_hot)
                if total_samples > 0:
                    n_classes = 3
                    class_weight = {cls_int: total_samples / (n_classes * count)
                                    for cls_int, count in class_counts.items() if count > 0}
                    self.logger.info(f"Calculated balanced class weights for LSTM: {class_weight}")
                else:
                    self.logger.warning("Cannot calculate class weights: No training samples after sequence creation.")

            elif isinstance(class_weight_param, dict):
                class_weight = {self.label_map.get(orig_lbl, orig_lbl): weight
                                for orig_lbl, weight in class_weight_param.items()}
                self.logger.info(f"Using provided class weights for LSTM: {class_weight}")
            elif class_weight_param is not None:
                self.logger.warning(f"Unsupported 'class_weight' strategy '{class_weight_param}' for LSTM. Skipping class weighting.")

            callbacks = []
            es_patience = lstm_params.early_stopping_patience
            if es_patience is not None and es_patience > 0:
                monitor_metric = 'val_loss' if val_data else 'loss'
                callbacks.append(tf.keras.callbacks.EarlyStopping(monitor=monitor_metric, patience=es_patience, restore_best_weights=True)) # type: ignore
                self.logger.info(f"Added EarlyStopping with patience {es_patience} monitoring '{monitor_metric}'.")

            rlrop_factor = lstm_params.reduce_lr_on_plateau_factor
            rlrop_patience = lstm_params.reduce_lr_on_plateau_patience
            if rlrop_factor is not None and rlrop_patience is not None and rlrop_patience > 0:
                monitor_metric = 'val_loss' if val_data else 'loss'
                callbacks.append(tf.keras.callbacks.ReduceLROnPlateau(monitor=monitor_metric, factor=rlrop_factor, patience=rlrop_patience)) # type: ignore
                self.logger.info(f"Added ReduceLROnPlateau with factor {rlrop_factor} and patience {rlrop_patience} monitoring '{monitor_metric}'.")

            epochs = lstm_params.epochs
            batch_size = lstm_params.batch_size

            self.logger.info(f"Training LSTM model for {epochs} epochs with batch size {batch_size}...")
            history = self.model.fit( # type: ignore
                X_train_seq,
                y_train_seq_one_hot,
                epochs=epochs,
                batch_size=batch_size,
                validation_data=val_data,
                class_weight=class_weight,
                callbacks=callbacks,
                verbose=1
            )
            self.logger.info("LSTM model training complete.")
            self.training_history = history.history


        else: # Scikit-learn compatible models (RandomForest, XGBoost)
            steps = [('preprocessor', self.preprocessor)]

            if self.model_type == 'RandomForest':
                rf_params: RandomForestParams = self._model_config.random_forest_params
                
                # Create a mutable dict from dataclass for model init
                model_init_params = rf_params.__dict__.copy()
                
                # Remove class_balancing as it's handled by imblearn pipeline
                class_balancing_strategy = model_init_params.pop('class_balancing', None)
                model_init_params.pop('undersample_ratio', None) # Remove if present

                # Use sampling strategy for imblearn Pipeline
                if class_balancing_strategy == 'undersampling':
                    sampler = RandomUnderSampler(random_state=42)
                    steps.append(('sampler', sampler))
                    self.logger.info("Added RandomUnderSampler to training pipeline.")
                elif class_balancing_strategy == 'oversampling':
                    sampler = SMOTE(random_state=42)
                    steps.append(('sampler', sampler))
                    self.logger.info("Added SMOTE to training pipeline.")
                elif class_balancing_strategy is not None:
                    self.logger.warning(f"Unsupported 'class_balancing' strategy '{class_balancing_strategy}'. Skipping sampler.")

                model = RandomForestClassifier(random_state=42, n_jobs=-1, **model_init_params)

            elif self.model_type == 'XGBoost':
                xgb_params: XGBoostParams = self._model_config.xgboost_params

                # Create a mutable dict from dataclass for model init
                model_init_params = xgb_params.__dict__.copy()

                # Remove class_balancing as it's handled by imblearn pipeline
                class_balancing_strategy = model_init_params.pop('class_balancing', None)
                model_init_params.pop('undersample_ratio', None) # Remove if present

                # Use sampling strategy for imblearn Pipeline
                if class_balancing_strategy == 'undersampling':
                    sampler = RandomUnderSampler(random_state=42)
                    steps.append(('sampler', sampler))
                    self.logger.info("Added RandomUnderSampler to training pipeline.")
                elif class_balancing_strategy == 'oversampling':
                    sampler = SMOTE(random_state=42)
                    steps.append(('sampler', sampler))
                    self.logger.info("Added SMOTE to training pipeline.")
                elif class_balancing_strategy is not None:
                    self.logger.warning(f"Unsupported 'class_balancing' strategy '{class_balancing_strategy}'. Skipping sampler.")

                # XGBoost specific setup for ternary classification
                model_init_params.update({
                    'objective': 'multi:softmax',
                    'num_class': 3,
                    'eval_metric': 'mlogloss',
                    'random_state': 42,
                    'n_jobs': -1,
                })
                model = XGBClassifier(**model_init_params)

            else:
                raise ValueError(f"Unsupported model type for training: {self.model_type}")

            steps.append(('model', model))

            self.pipeline = Pipeline(steps)
            self.logger.info(f"Training pipeline created with steps: {[name for name, _ in self.pipeline.steps]}")

            self.logger.info(f"Training {self.model_type} pipeline...")
            y_train_mapped = y_train.map(self.label_map)
            self.pipeline.fit(X_train, y_train_mapped)
            self.logger.info(f"{self.model_type} pipeline training complete.")


    def evaluate(self, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, Any]:
        """
        Evaluates the trained model on the test data.

        Args:
            X_test (pd.DataFrame): Test features (should be cleaned of NaNs in calling script).
                                   This DataFrame should contain *all* original features
                                   from the processed data before subsetting.
            y_test (pd.Series): Test labels (-1, 0, 1) (should be cleaned of NaNs in calling script).

        Returns:
            Dict[str, Any]: Evaluation metrics.

        Raises:
            ValueError: If test data is empty or contains issues.
            RuntimeError: If evaluation fails or model/preprocessor is not available.
            ImportError: If TensorFlow is required for LSTM but not installed.
        """
        if X_test.empty or y_test.empty:
            raise ValueError("Test data is empty.")
        if len(X_test) != len(y_test):
            raise ValueError("Test features and labels have different lengths.")

        self.logger.info(f"Evaluating {self.model_type} model on test set...")

        y_test_mapped = y_test.map(self.label_map).values

        if self.model_type == 'LSTM':
            if self.model is None or self.preprocessor is None:
                raise RuntimeError("LSTM model or preprocessor not available for evaluation.")
            if self.sequence_length <= 0:
                raise ValueError(f"Invalid sequence_length for LSTM evaluation: {self.sequence_length}")

            self.logger.info("Preparing test data for LSTM evaluation...")
            X_test_scaled = self.preprocessor.transform(X_test)

            if np.isnan(X_test_scaled).any() or np.isinf(X_test_scaled).any():
                nan_count = np.isnan(X_test_scaled).sum()
                inf_count = np.isinf(X_test_scaled).sum()
                error_msg = f"Scaled test data contains NaN ({nan_count}) or Inf ({inf_count}) values. Cannot evaluate LSTM."
                self.logger.critical(error_msg)
                raise ValueError(error_msg)

            X_test_seq, y_test_seq_one_hot_aligned = self._prepare_lstm_sequences(X_test_scaled, y_test_mapped)

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
            X (pd.DataFrame): New features data. This DataFrame should contain
                              *all* original features from the processed data,
                              as the preprocessor handles subsetting internally.

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

        if self.preprocessor is None:
            raise RuntimeError("Preprocessor is not loaded or trained. Cannot make predictions.")
        if self.model is None and self.pipeline is None:
            raise RuntimeError("Model or pipeline is not loaded or trained. Cannot make predictions.")


        if self.model_type == 'LSTM':
            if self.sequence_length <= 0:
                raise ValueError(f"Invalid sequence_length for LSTM prediction: {self.sequence_length}")

            try:
                X_scaled = self.preprocessor.transform(X)
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
            X_sequences, _ = self._prepare_lstm_sequences(X_scaled, dummy_y_mapped)

            if X_sequences.shape[0] == 0:
                self.logger.warning("No sequences generated from input data for LSTM prediction. Returning empty predictions.")
                prediction_index = X.index[self.sequence_length - 1:]
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
            X (pd.DataFrame): New features data. This DataFrame should contain
                              *all* original features from the processed data,
                              as the preprocessor handles subsetting internally.

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

        if self.preprocessor is None:
            raise RuntimeError("Preprocessor is not loaded or trained. Cannot make probability predictions.")
        if self.model is None and self.pipeline is None:
            raise RuntimeError("Model or pipeline is not loaded or trained. Cannot make probability predictions.")


        if self.model_type == 'LSTM':
            if self.sequence_length <= 0:
                raise ValueError(f"Invalid sequence_length for LSTM probability prediction: {self.sequence_length}")

            try:
                X_scaled = self.preprocessor.transform(X)
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
            X_sequences, _ = self._prepare_lstm_sequences(X_scaled, dummy_y_mapped)

            if X_sequences.shape[0] == 0:
                self.logger.warning("No sequences generated from input data for LSTM probability prediction. Returning None.")
                return None

            self.logger.info("Making probability predictions with Keras LSTM model on sequences...")
            y_pred_proba_array = self.model.predict(X_sequences) # type: ignore

            self.logger.info("LSTM probability predictions made.")
            prediction_index = X.index[self.sequence_length - 1:]

            aligned_probabilities_df = pd.DataFrame(np.nan, index=X.index, columns=self.classes.tolist(), dtype=float)

            if len(y_pred_proba_array) == len(prediction_index):
                aligned_probabilities_df.loc[prediction_index] = y_pred_proba_array
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

                return pd.DataFrame(y_pred_proba_array, index=X.index, columns=self.classes.tolist())
            else:
                self.logger.info(f"Model type '{self.model_type}' or its pipeline does not support predict_proba.")
                return None


    def save(self, symbol: str, interval: str, model_key: str):
        """
        Saves the trained model (pipeline or Keras model) and its metadata using DataManager.

        Args:
            symbol (str): Trading pair symbol.
            interval (str): Time interval.
            model_key (str): Key for the model configuration in config.params.MODEL_CONFIG.

        Raises:
            ValueError: If model_key is invalid or model/preprocessor is not trained.
            OSError: If there's an error saving files via DataManager.
            RuntimeError: If no model or preprocessor is available to save.
        """
        if self.model is None and self.pipeline is None:
            raise RuntimeError("No model or pipeline trained/loaded to save.")
        if self.preprocessor is None:
            raise RuntimeError("Preprocessor is not trained/loaded. Cannot save model.")

        self.logger.info(f"Saving trained {self.model_type} model and metadata for {symbol.upper()} {interval} using DataManager...")

        # Prepare metadata dictionary
        metadata = {
            'model_type': self.model_type,
            'feature_columns_processed': self.feature_columns_processed,
            'feature_columns_original': self.feature_columns_original,
            'label_map': self.label_map,
            'inverse_label_map': self.inverse_label_map,
            'classes': self.classes.tolist(),
            # Save the full ModelConfig object directly (it's serializable due to dataclasses)
            'model_config': self._model_config.__dict__, # Convert dataclass to dict for saving
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
            if self.model_type == 'LSTM':
                self.dm.save_model_artifact(
                    artifact=self.model,
                    symbol=symbol,
                    interval=interval,
                    model_key=model_key,
                    artifact_type='model'
                )
                self.logger.info(f"Keras LSTM model saved successfully via DataManager.")

                self.dm.save_model_artifact(
                    artifact=self.preprocessor,
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
                self.logger.debug("Preprocessor is part of the scikit-learn pipeline, not saved separately.")


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
            model_key (str): Key for the model configuration.

        Returns:
            ModelTrainer: The current instance, updated with the loaded model and metadata.

        Raises:
            FileNotFoundError: If the model or metadata file is not found (raised by DataManager).
            RuntimeError: If loading fails for other reasons.
            ValueError: If metadata is missing crucial info.
            ImportError: If TensorFlow is required but not available.
        """
        self.logger.info(f"Loading trained model for {symbol.upper()} {interval} ({self.model_type}) using DataManager...")

        # --- Load Metadata using DataManager ---
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
                # We need to manually reconstruct nested dataclasses if they were saved as dicts
                loaded_model_config_dict = metadata_dict['model_config']
                
                # Handle dimensionality_reduction separately if it's a dict
                if 'dimensionality_reduction' in loaded_model_config_dict and \
                   isinstance(loaded_model_config_dict['dimensionality_reduction'], dict):
                    loaded_model_config_dict['dimensionality_reduction'] = \
                        self._model_config.dimensionality_reduction.__class__(**loaded_model_config_dict['dimensionality_reduction'])

                # Handle model-specific params separately
                if 'lstm_params' in loaded_model_config_dict and isinstance(loaded_model_config_dict['lstm_params'], dict):
                    loaded_model_config_dict['lstm_params'] = LSTMParams(**loaded_model_config_dict['lstm_params'])
                if 'random_forest_params' in loaded_model_config_dict and isinstance(loaded_model_config_dict['random_forest_params'], dict):
                    loaded_model_config_dict['random_forest_params'] = RandomForestParams(**loaded_model_config_dict['random_forest_params'])
                if 'xgboost_params' in loaded_model_config_dict and isinstance(loaded_model_config_dict['xgboost_params'], dict):
                    loaded_model_config_dict['xgboost_params'] = XGBoostParams(**loaded_model_config_dict['xgboost_params'])

                self._model_config = ModelConfig(**loaded_model_config_dict)
                self.logger.info("ModelConfig reconstructed from loaded metadata.")
                # Update instance attributes based on the loaded config
                self.model_type = self._model_config.model_type
                self.features_to_use = self._model_config.features_to_use
                self.pca_enabled = self._model_config.dimensionality_reduction.enabled
                self.pca_method = self._model_config.dimensionality_reduction.method
                self.pca_params = self._model_config.dimensionality_reduction.params
                self.sequence_length = self._model_config.lstm_params.sequence_length_bars if self.model_type == 'LSTM' else 1
            else:
                self.logger.warning("ModelConfig not found in metadata. Attempting to use default or initialized config for structure.")
                # Fallback to direct attribute setting if ModelConfig not explicitly saved
                self.model_type = metadata_dict.get('model_type', self.model_type)
                self.features_to_use = metadata_dict.get('features_to_use', self.features_to_use)
                self.pca_enabled = metadata_dict.get('pca_enabled', self.pca_enabled)
                self.pca_method = metadata_dict.get('pca_method', self.pca_method)
                self.pca_params = metadata_dict.get('pca_params', self.pca_params)
                self.sequence_length = metadata_dict.get('sequence_length_bars', metadata_dict.get('sequence_length', self.sequence_length)) # Also handle old key

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


        # --- Load Model / Pipeline using DataManager ---
        self.logger.info(f"Loading {self.model_type} model artifact...")
        try:
            if self.model_type == 'LSTM':
                if not LSTM_AVAILABLE:
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
                    self.preprocessor = self.dm.load_model_artifact(
                        symbol=symbol,
                        interval=interval,
                        model_key=model_key,
                        artifact_type='preprocessor'
                    )
                    self.logger.info("Preprocessor loaded successfully via DataManager.")
                    # If feature_columns_processed was not loaded from metadata, try to infer from preprocessor
                    if self.feature_columns_processed is None:
                        if hasattr(self.preprocessor, 'get_feature_names_out'):
                            self.feature_columns_processed = self.preprocessor.get_feature_names_out().tolist()
                            self.logger.info(f"Inferred processed feature columns from loaded preprocessor: {self.feature_columns_processed}")
                        elif hasattr(self.preprocessor, 'feature_names_in_'):
                            self.feature_columns_processed = self.preprocessor.feature_names_in_.tolist()
                            self.logger.info(f"Inferred processed feature columns from preprocessor.feature_names_in_: {self.feature_columns_processed}")
                        else:
                            self.logger.warning("Could not infer processed feature columns from loaded preprocessor.")

                except FileNotFoundError:
                    self.logger.warning("Preprocessor artifact not found. LSTM predictions/evaluation might fail.")
                    self.preprocessor = None
                except Exception as e:
                    self.logger.warning(f"Error loading preprocessor via DataManager: {e}", exc_info=True)
                    self.preprocessor = None


            else: # Load Scikit-learn pipeline using DataManager
                self.pipeline = self.dm.load_model_artifact(
                    symbol=symbol,
                    interval=interval,
                    model_key=model_key,
                    artifact_type='pipeline'
                )
                self.logger.info(f"{self.model_type} pipeline loaded successfully via DataManager.")
                if self.pipeline is not None and len(self.pipeline.steps) > 0:
                    self.preprocessor = self.pipeline.steps[0][1]
                    self.model = self.pipeline.steps[-1][1]

                    if self.feature_columns_processed is None:
                        if hasattr(self.preprocessor, 'get_feature_names_out'):
                            self.feature_columns_processed = self.preprocessor.get_feature_names_out().tolist()
                            self.logger.info(f"Inferred processed feature columns from loaded preprocessor: {self.feature_columns_processed}")
                        elif hasattr(self.preprocessor, 'feature_names_in_'):
                            self.feature_columns_processed = self.preprocessor.feature_names_in_.tolist()
                            self.logger.info(f"Inferred processed feature columns from preprocessor.feature_names_in_: {self.feature_columns_processed}")
                        else:
                            self.logger.warning("Could not infer processed feature columns from preprocessor.")
                else:
                    self.logger.warning("Loaded pipeline has no steps. Preprocessor and model not extracted.")
                    self.preprocessor = None
                    self.model = None

        except FileNotFoundError:
            raise
        except Exception as e:
            self.logger.error(f"An unexpected error occurred during model artifact loading via DataManager: {e}", exc_info=True)
            raise RuntimeError(f"Failed to load trained model artifact: {e}")

        # Final check for essential components after loading
        if self.preprocessor is None:
            self.logger.error("Preprocessor could not be loaded. Model might not work correctly for prediction/analysis.")
        if self.model is None and self.pipeline is None:
            self.logger.error("Model or pipeline could not be loaded. Model is not usable.")
        if self.feature_columns_original is None:
            self.logger.error("Original feature columns could not be loaded. Cannot prepare data for analysis.")

        self.logger.info(f"ModelTrainer instance loaded successfully for {self.model_type}.")
        return self

