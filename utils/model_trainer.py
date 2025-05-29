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
from sklearn.model_selection import TimeSeriesSplit # Keep TimeSeriesSplit for potential CV within trainer (though tuning is in train_model)
from sklearn.preprocessing import StandardScaler


from pandas import Int8Dtype

from xgboost import XGBClassifier

from pathlib import Path

from datetime import datetime

# --- Conditional Import for TensorFlow and Keras for LSTM ---
# Need to install tensorflow: pip install tensorflow
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential, load_model # type: ignore
    from tensorflow.keras.layers import LSTM, Dense, Dropout, Input # type: ignore
    from tensorflow.keras.optimizers import Adam # type: ignore
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau # type: ignore
    from tensorflow.keras.utils import to_categorical # type: ignore

    # Check for GPU availability and log TensorFlow version only once per module load
    tf_version = getattr(tf, '__version__', 'unknown')
    # Use module-level logger initially for import status
    _module_logger = logging.getLogger(__name__)
    _module_logger.info(f"TensorFlow (version {tf_version}) imported successfully in ModelTrainer.")
    if tf.config.list_physical_devices('GPU'):
        _module_logger.info("GPU is available and enabled for TensorFlow.")
    else:
        _module_logger.info("GPU is not available or not enabled for TensorFlow.")

    LSTM_AVAILABLE = True
except ImportError:
    _module_logger = logging.getLogger(__name__)
    _module_logger.warning("TensorFlow not found. LSTM model type will not be available in ModelTrainer.")
    tf = None
    LSTM_AVAILABLE = False
except Exception as e:
    # Catch other potential errors during TF import (e.e.g., DLL issues)
    _module_logger = logging.getLogger(__name__)
    _module_logger.error(f"Error importing TensorFlow/Keras in ModelTrainer: {e}", exc_info=True)
    tf = None
    LSTM_AVAILABLE = False


# Conditional import for type hinting if TYPE_CHECKING is True
if TYPE_CHECKING:
    from sklearn.base import BaseEstimator
    # from tensorflow.keras.models import Model as KerasModel


# Get logger for this module (used for messages not tied to a specific trainer instance)
logger = logging.getLogger(__name__)

# Import DataManager here
try:
    from utils.data_manager import DataManager
except ImportError:
    logger.error("DataManager not found in utils. ModelTrainer cannot save/load artifacts.")
    # Define a dummy DataManager or handle this gracefully if DataManager is essential
    # For this case, DataManager is essential for save/load, so we might need to raise error or exit later
    DataManager = None # type: ignore


class ModelTrainer:
    """
    Handles training, evaluation, and saving of different trading models for ternary
    classification (-1, 0, 1). Supports scikit-learn compatible models (like
    RandomForest, XGBoost) and Keras LSTM models. Includes preprocessing,
    handling of class imbalance, and metadata management.
    Allows specifying a subset of features to use.
    Uses DataManager for saving and loading model artifacts.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initializes the ModelTrainer with the model configuration.
        Does NOT build the model here; model is built during train() or loaded during load().

        Args:
            config (Dict[str, Any]): A dictionary containing the model configuration
                                     including 'model_type', 'params', and optionally
                                     'features_to_use'.
        Raises:
            ValueError: If 'model_type' or 'params' are missing in the config,
                        or if the model_type is unsupported.
            ImportError: If TensorFlow is required for LSTM but not installed,
                         or if DataManager is not available.
        """
        # --- Use instance-specific logger here ---
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self.logger.info("ModelTrainer initialized.")

        # Check if DataManager is available
        if DataManager is None:
             raise ImportError("DataManager is not available. Cannot initialize ModelTrainer.")


        if not isinstance(config, dict):
            raise TypeError("config must be a dictionary.")

        self.config = config # Store the full config
        self.model_type = config.get('model_type')
        # Use a copy of the params dictionary
        self.model_params = config.get('params', {}).copy()
        # Get the optional list of features to use
        self.features_to_use: Optional[List[str]] = config.get('features_to_use')


        if self.model_type is None:
            raise ValueError("Model configuration must contain 'model_type'.")
        if not isinstance(self.model_params, dict):
             raise ValueError("'params' in model configuration must be a dictionary.")
        if self.features_to_use is not None and not isinstance(self.features_to_use, list):
             raise TypeError("'features_to_use' in config must be a list or None.")


        # Check if LSTM is requested and available
        if self.model_type == 'lstm' and not LSTM_AVAILABLE:
             raise ImportError("TensorFlow is required for LSTM model but is not installed.")

        # Initialize model and pipeline as None
        self.model: Optional[Union['BaseEstimator', 'tf.keras.Model']] = None # type: ignore
        self.pipeline: Optional[Pipeline] = None
        self.preprocessor: Optional[ColumnTransformer] = None
        # self.feature_columns_processed will store the names of the features *after* preprocessing
        self.feature_columns_processed: Optional[List[str]] = None # Renamed for clarity
        # self.feature_columns_original will store the names of the features *before* preprocessing
        self.feature_columns_original: Optional[List[str]] = None # Added attribute


        # Define label mapping for ternary classification (-1, 0, 1) to integers (0, 1, 2)
        self.label_map: Dict[int, int] = {-1: 0, 0: 1, 1: 2}
        self.inverse_label_map: Dict[int, int] = {0: -1, 1: 0, 2: 1}
        self.classes: np.ndarray = np.array([-1, 0, 1])

        # Get sequence length for LSTM (default to 1 for non-LSTM models)
        self.sequence_length: int = self.model_params.get('sequence_length_bars', 1) # Use updated key name

        self.logger.info(f"ModelTrainer initialized for model type: {self.model_type}")
        if self.model_type == 'lstm':
             self.logger.info(f"LSTM Sequence Length: {self.sequence_length}")
        if self.features_to_use is not None:
             self.logger.info(f"Using specified feature subset: {self.features_to_use}")
        else:
             self.logger.info("Using all available features from input data.")


        self.logger.debug(f"Model parameters: {self.model_params}")

        # Instantiate DataManager here for use in save/load methods
        self.dm = DataManager()


    def _build_lstm_model(self, n_features: int) -> 'tf.keras.Model': # type: ignore
        """
        Builds the Keras LSTM model based on model_params and the number of features.

        Args:
            n_features (int): The number of features per timestep *after* preprocessing.

        Returns:
            tf.keras.Model: The built Keras Sequential model.
        """
        if not LSTM_AVAILABLE:
             # This should be caught in __init__, but defensive check
             raise ImportError("TensorFlow is not available to build LSTM model.")

        # Get LSTM specific parameters from self.model_params
        sequence_length = self.sequence_length # Use instance variable
        n_layers = self.model_params.get('n_layers', 1)
        units_per_layer = self.model_params.get('units_per_layer', 50)
        dropout_rate = self.model_params.get('dropout_rate', 0.2)
        learning_rate = self.model_params.get('learning_rate', 0.001)
        clipnorm = self.model_params.get('clipnorm')
        clipvalue = self.model_params.get('clipvalue')


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


        model = Sequential()
        # Input layer expects shape (sequence_length, n_features)
        model.add(Input(shape=(sequence_length, n_features)))

        # Add LSTM layers
        for i in range(n_layers):
            # Return sequences for all but the last LSTM layer
            return_sequences = i < n_layers - 1
            model.add(LSTM(units_per_layer, return_sequences=return_sequences))
            if dropout_rate > 0:
                model.add(Dropout(dropout_rate))

        # Output layer for ternary classification (3 classes)
        model.add(Dense(3, activation='softmax'))

        # Configure optimizer with potential gradient clipping
        optimizer_params = {'learning_rate': learning_rate}
        if clipnorm is not None:
            optimizer_params['clipnorm'] = clipnorm
            self.logger.info(f"Using gradient clipping (clipnorm={clipnorm}) in Adam optimizer.")
        if clipvalue is not None:
            optimizer_params['clipvalue'] = clipvalue
            self.logger.info(f"Using gradient clipping (clipvalue={clipvalue}) in Adam optimizer.")

        optimizer = Adam(**optimizer_params)

        # Compile the model
        model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

        self.logger.info("LSTM model architecture built and compiled.")
        # Log model summary
        model.summary(print_fn=lambda x: self.logger.info(x))
        return model

    def _create_preprocessor(self, X: pd.DataFrame, feature_subset: Optional[List[str]] = None) -> ColumnTransformer:
        """
        Creates and fits a ColumnTransformer for preprocessing.
        Applies StandardScaler to all numeric features in the specified subset or all numeric features.

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
             # Store the actual columns used by the preprocessor (empty in this case)
             self.feature_columns_processed = [] # Use processed attribute
        else:
            self.logger.info(f"Applying StandardScaler to numeric features: {numeric_features}")
            preprocessor = ColumnTransformer(
                transformers=[
                    ('num', StandardScaler(), numeric_features)
                ],
                remainder='passthrough' # Keep other columns (e.g., potential non-numeric in original X, though they shouldn't be features)
            )

            self.logger.info("Fitting preprocessor...")
            # Fit the preprocessor on the subsetted numeric data
            preprocessor.fit(X_subset[numeric_features])
            self.logger.info("Preprocessor fitted.")

            # Store the names of the features that were scaled by the preprocessor.
            # These are the names *after* any ColumnTransformer naming conventions
            # if remainder='passthrough' adds prefixes, but for StandardScaler on numeric
            # features, they should largely remain the original names.
            # A more robust way to get feature names out:
            try:
                 # This method is available in newer scikit-learn versions
                 self.feature_columns_processed = preprocessor.get_feature_names_out().tolist()
                 self.logger.info(f"Feature columns after preprocessing: {self.feature_columns_processed}")
            except AttributeError:
                 # Fallback for older scikit-learn versions or simpler cases
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
        n_features = X_scaled.shape[1] # Number of features *after* preprocessing/selection

        # Check if there is enough data to create at least one sequence
        if n_samples < sequence_length:
             self.logger.warning(f"Not enough data points ({n_samples}) to create sequences of length {sequence_length}. Returning empty arrays.")
             # Return empty arrays with the correct shape dimensions but zero samples
             return np.empty((0, sequence_length, n_features)), np.empty((0, 3)) # 3 classes for one-hot

        X_sequences = []
        y_sequences = []

        # Iterate and create sequences
        # A sequence of length `sequence_length` ending at index `i`
        # corresponds to the label at index `i`.
        # The first sequence ends at index `sequence_length - 1`.
        for i in range(sequence_length - 1, n_samples):
            X_sequences.append(X_scaled[i - sequence_length + 1 : i + 1])
            y_sequences.append(y_mapped[i]) # Label for the last timestep in the sequence

        X_sequences = np.array(X_sequences)
        y_sequences = np.array(y_sequences)

        # One-hot encode the labels
        y_sequences_one_hot = to_categorical(y_sequences, num_classes=3)

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

        # Store the original feature column names from the training data
        self.feature_columns_original = X_train.columns.tolist()
        self.logger.info(f"Original feature columns from training data: {self.feature_columns_original}")


        # Create and fit the preprocessor using the training data.
        # Pass the optional features_to_use list to the preprocessor creation.
        self.preprocessor = self._create_preprocessor(X_train, feature_subset=self.features_to_use)
        # The self.feature_columns_processed attribute is set within _create_preprocessor

        # Check if any features were selected by the preprocessor
        if not self.feature_columns_processed or len(self.feature_columns_processed) == 0:
             self.logger.critical("No features were selected or created by the preprocessor. Cannot train.")
             raise RuntimeError("No features selected by preprocessor.")


        if self.model_type == 'lstm':
             if not LSTM_AVAILABLE:
                  raise ImportError("TensorFlow is not installed. Cannot train LSTM model.")

             # The number of features for the LSTM input layer is the number of columns
             # output by the preprocessor.
             n_features_after_prep = len(self.feature_columns_processed)

             # Build the LSTM model architecture
             self.model = self._build_lstm_model(n_features=n_features_after_prep)

             # Transform data using the fitted preprocessor
             # Pass the original X_train to the preprocessor; it will handle subsetting internally
             X_train_scaled = self.preprocessor.transform(X_train)

             # Check for NaN/Inf in scaled training data
             if np.isnan(X_train_scaled).any() or np.isinf(X_train_scaled).any():
                 nan_count = np.isnan(X_train_scaled).sum()
                 inf_count = np.isinf(X_train_scaled).sum()
                 error_msg = f"Scaled training data contains NaN ({nan_count}) or Inf ({inf_count}) values. Cannot train LSTM."
                 self.logger.critical(error_msg)
                 raise ValueError(error_msg)
             self.logger.info("Scaled training data checked: No NaN or Inf values found.")

             # Prepare sequences for training
             y_train_mapped = y_train.map(self.label_map).values
             X_train_seq, y_train_seq_one_hot = self._prepare_lstm_sequences(X_train_scaled, y_train_mapped)

             if X_train_seq.shape[0] == 0:
                  self.logger.error("No training sequences generated for LSTM. Cannot train.")
                  raise ValueError("No training sequences generated.")

             # Prepare validation data if provided
             val_data = None
             if X_val is not None and y_val is not None and not X_val.empty and not y_val.empty:
                  if len(X_val) != len(y_val):
                       self.logger.error("Validation features and labels have different lengths.")
                       raise ValueError("Validation features and labels have different lengths.")

                  # Transform validation data using the fitted preprocessor
                  # Pass the original X_val to the preprocessor
                  X_val_scaled = self.preprocessor.transform(X_val)

                  # Check for NaN/Inf in scaled validation data
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


             # Handle class weighting for LSTM
             class_weight = None
             class_weight_param = self.model_params.get('class_weight')
             if class_weight_param == 'balanced':
                  # Calculate weights based on the actual labels in the training sequences
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
                  # Map user-provided labels (-1, 0, 1) to internal integers (0, 1, 2)
                  class_weight = {self.label_map.get(orig_lbl, orig_lbl): weight
                                  for orig_lbl, weight in class_weight_param.items()}
                  self.logger.info(f"Using provided class weights for LSTM: {class_weight}")
             elif class_weight_param is not None:
                  self.logger.warning(f"Unsupported 'class_weight' strategy '{class_weight_param}' for LSTM. Skipping class weighting.")


             # Configure callbacks
             callbacks = []
             es_patience = self.model_params.get('early_stopping_patience')
             if es_patience is not None and es_patience > 0:
                  monitor_metric = 'val_loss' if val_data else 'loss'
                  callbacks.append(EarlyStopping(monitor=monitor_metric, patience=es_patience, restore_best_weights=True))
                  self.logger.info(f"Added EarlyStopping with patience {es_patience} monitoring '{monitor_metric}'.")

             rlrop_factor = self.model_params.get('reduce_lr_on_plateau_factor')
             rlrop_patience = self.model_params.get('reduce_lr_on_plateau_patience')
             if rlrop_factor is not None and rlrop_patience is not None and rlrop_patience > 0:
                  monitor_metric = 'val_loss' if val_data else 'loss'
                  callbacks.append(ReduceLROnPlateau(monitor=monitor_metric, factor=rlrop_factor, patience=rlrop_patience))
                  self.logger.info(f"Added ReduceLROnPlateau with factor {rlrop_factor} and patience {rlrop_patience} monitoring '{monitor_metric}'.")


             # Train the LSTM model
             epochs = self.model_params.get('epochs', 50)
             batch_size = self.model_params.get('batch_size', 32)

             self.logger.info(f"Training LSTM model for {epochs} epochs with batch size {batch_size}...")
             history = self.model.fit(
                 X_train_seq,
                 y_train_seq_one_hot,
                 epochs=epochs,
                 batch_size=batch_size,
                 validation_data=val_data,
                 class_weight=class_weight,
                 callbacks=callbacks,
                 verbose=1 # Show training progress
             )
             self.logger.info("LSTM model training complete.")
             self.training_history = history.history # Store training history


        else: # Scikit-learn compatible models (RandomForest, XGBoost, etc.)
            # Define the steps for the scikit-learn pipeline
            steps = [('preprocessor', self.preprocessor)]

            # Handle class imbalance using imblearn samplers if configured
            balanced_strategy_train = self.model_params.get('class_balancing') # Use 'class_balancing' key
            if balanced_strategy_train == 'undersampling':
                 # Get undersampling ratio if specified
                 undersample_ratio_train = self.model_params.get('undersample_ratio', 1.0)
                 # Use sampling_strategy='majority' to balance majority class to undersample_ratio * minority class size
                 # Or use a dictionary mapping class labels to desired number of samples
                 # For simplicity, using default which balances to the minority class size
                 sampler = RandomUnderSampler(random_state=42) # Use a fixed random state for reproducibility
                 steps.append(('sampler', sampler))
                 self.logger.info("Added RandomUnderSampler to training pipeline.")

            elif balanced_strategy_train == 'oversampling':
                 # Use sampling_strategy='minority' to oversample minority class(es) to equal the majority class size)
                 sampler = SMOTE(random_state=42) # Use a fixed random state for reproducibility
                 steps.append(('sampler', sampler))
                 self.logger.info("Added SMOTE to training pipeline.")

            elif balanced_strategy_train is not None:
                 self.logger.warning(f"Unsupported 'class_balancing' strategy '{balanced_strategy_train}' for training. Skipping sampler in training pipeline.")


            # Create the model instance based on model_type
            if self.model_type == 'random_forest':
                 # Remove class_balancing and undersample_ratio from params before passing to RF
                 rf_params = {k: v for k, v in self.model_params.items() if k not in ['class_balancing', 'undersample_ratio']}
                 # Add class_weight parameter if specified in config
                 class_weight_param = self.model_params.get('class_weight')
                 if class_weight_param is not None:
                      rf_params['class_weight'] = class_weight_param
                      self.logger.info(f"Using class_weight='{class_weight_param}' for RandomForest.")

                 model = RandomForestClassifier(random_state=42, n_jobs=-1, **rf_params) # Use fixed random_state and n_jobs
            elif self.model_type == 'xgboost':
                 # Remove class_balancing and undersample_ratio from params before passing to XGBoost
                 xgb_params = {k: v for k, v in self.model_params.items() if k not in ['class_balancing', 'undersample_ratio']}
                 # XGBoost specific setup for ternary classification
                 xgb_params.update({
                     'objective': 'multi:softmax', # Output class labels directly
                     'num_class': 3, # Should be 3
                     'eval_metric': 'mlogloss', # Logloss for multi-class
                     # Removed 'use_label_encoder': False as it's deprecated/unused in recent XGBoost
                     'random_state': 42, # Use fixed random_state
                     'n_jobs': -1, # Use all available cores
                 })
                 model = XGBClassifier(**xgb_params)
            else:
                 # This should be caught by the initial check, but defensive programming
                 raise ValueError(f"Unsupported model type for training: {self.model_type}")

            # Add the model instance to the pipeline steps
            steps.append(('model', model))

            # Create the full imblearn pipeline
            self.pipeline = Pipeline(steps)
            self.logger.info(f"Training pipeline created with steps: {[name for name, _ in self.pipeline.steps]}")

            # Check for NaN/Inf in training data before fitting the pipeline
            # The pipeline's preprocessor step will handle scaling and NaN/Inf check internally
            # if _create_preprocessor includes it, but an explicit check here is good.
            # However, load_and_split_data is supposed to remove NaNs/Infs in features/labels.
            # Let's trust the data loading step and assume X_train is clean.

            self.logger.info(f"Training {self.model_type} pipeline...")
            # Map y_train to integers (0, 1, 2) for models/samplers that require it
            y_train_mapped = y_train.map(self.label_map)
            # Fit the pipeline on the original X_train (the pipeline handles the scaling internally)
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
        """
        if X_test.empty or y_test.empty:
            raise ValueError("Test data is empty.")
        if len(X_test) != len(y_test):
            raise ValueError("Test features and labels have different lengths.")

        self.logger.info(f"Evaluating {self.model_type} model on test set...")

        # Map test labels to integers (0, 1, 2)
        y_test_mapped = y_test.map(self.label_map).values

        if self.model_type == 'lstm':
             if self.model is None or self.preprocessor is None:
                  raise RuntimeError("LSTM model or preprocessor not available for evaluation.")
             if self.sequence_length <= 0:
                  raise ValueError(f"Invalid sequence_length for LSTM evaluation: {self.sequence_length}")

             self.logger.info("Preparing test data for LSTM evaluation...")
             # Transform test data using the fitted preprocessor
             # Pass the original X_test to the preprocessor; it will handle subsetting internally
             X_test_scaled = self.preprocessor.transform(X_test)

             # Check for NaN/Inf in scaled test data
             if np.isnan(X_test_scaled).any() or np.isinf(X_test_scaled).any():
                 nan_count = np.isnan(X_test_scaled).sum()
                 inf_count = np.isinf(X_test_scaled).sum()
                 error_msg = f"Scaled test data contains NaN ({nan_count}) or Inf ({inf_count}) values. Cannot evaluate LSTM."
                 self.logger.critical(error_msg)
                 raise ValueError(error_msg) # Raise error to prevent misleading evaluation

             # Prepare sequences for evaluation
             # Note: _prepare_lstm_sequences aligns the labels to the sequence output length
             X_test_seq, y_test_seq_one_hot_aligned = self._prepare_lstm_sequences(X_test_scaled, y_test_mapped)

             if X_test_seq.shape[0] == 0:
                  self.logger.warning("No test sequences generated for LSTM evaluation. Skipping evaluation.")
                  # Return a dictionary indicating no evaluation was performed
                  return {"note": "No test sequences generated for evaluation."}

             self.logger.info("Evaluating Keras LSTM model on test sequences...")
             # Evaluate the model using the prepared sequences and aligned one-hot labels
             loss, accuracy = self.model.evaluate(X_test_seq, y_test_seq_one_hot_aligned, verbose=0)

             # Get predictions for detailed metrics
             y_pred_proba = self.model.predict(X_test_seq)
             y_pred_mapped = np.argmax(y_pred_proba, axis=1)

             # Convert mapped predictions back to original labels (-1, 0, 1)
             y_pred_original = pd.Series(y_pred_mapped).map(self.inverse_label_map).fillna(0).values.astype(int)

             # Get the original test labels that correspond to the end of the sequences
             # The sequences start at index 0 and end at sequence_length - 1, then start at 1 and end at sequence_length, etc.
             # The labels are for the last timestep of the sequence.
             # So, the valid labels for evaluation start from index sequence_length - 1 of the original test set.
             y_test_aligned_original = y_test.iloc[self.sequence_length - 1:].values

             # Ensure prediction length matches aligned test label length
             if len(y_pred_original) != len(y_test_aligned_original):
                  self.logger.error(f"Mismatch in length between LSTM predictions ({len(y_pred_original)}) and aligned test labels ({len(y_test_aligned_original)}). Evaluation may be incorrect.")
                  raise RuntimeError("LSTM prediction and label length mismatch during evaluation.")


             # Calculate classification report and confusion matrix
             report = classification_report(y_test_aligned_original, y_pred_original, labels=self.classes, zero_division=0, output_dict=True)
             cm = confusion_matrix(y_test_aligned_original, y_pred_original, labels=self.classes)
             bal_acc = balanced_accuracy_score(y_test_aligned_original, y_pred_original)


             results = {
                 "loss": loss,
                 "accuracy": accuracy,
                 "balanced_accuracy": bal_acc,
                 "classification_report": report,
                 "confusion_matrix": cm.tolist() # Convert numpy array to list for easier saving/handling
             }
             self.logger.info("LSTM model evaluation complete.")


        else: # Scikit-learn compatible models
            if self.pipeline is None:
                raise RuntimeError("Model pipeline not available for evaluation.")

            # The pipeline handles preprocessing internally, so pass original X_test
            # The preprocessor within the pipeline will handle subsetting if features_to_use was provided during training
            # *** CORRECTED: Pass original X_test DataFrame to the pipeline's predict method ***
            y_pred_mapped = self.pipeline.predict(X_test)

            # Convert mapped predictions back to original labels (-1, 0, 1)
            y_pred_original = pd.Series(y_pred_mapped).map(self.inverse_label_map).fillna(0).values.astype(int)

            # Calculate metrics using original test labels and converted predictions
            report = classification_report(y_test.values, y_pred_original, labels=self.classes, zero_division=0, output_dict=True)
            cm = confusion_matrix(y_test.values, y_pred_original, labels=self.classes)
            bal_acc = balanced_accuracy_score(y_test.values, y_pred_original)
            # Calculate overall accuracy (can be misleading with imbalanced data)
            acc = accuracy_score(y_test.values, y_pred_original)

            results = {
                "overall_accuracy": acc,
                "balanced_accuracy": bal_acc,
                "classification_report": report,
                "confusion_matrix": cm.tolist() # Convert numpy array to list
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
            return pd.Series(dtype=Int8Dtype()) # Return empty Series with Int8Dtype

        self.logger.info(f"Making predictions with {self.model_type} model...")

        if self.preprocessor is None:
             raise RuntimeError("Preprocessor is not loaded or trained. Cannot make predictions.")
        if self.model is None and self.pipeline is None:
             raise RuntimeError("Model or pipeline is not loaded or trained. Cannot make predictions.")

        # Pass the original X DataFrame to the preprocessor.
        # The preprocessor will handle selecting the correct features based on how it was fitted.
        # If features_to_use was provided during training, it will only use those.
        # If not, it will use all numeric features it was fitted on.
        # *** REMOVED MANUAL PREPROCESSING FOR SCIKIT-LEARN MODELS ***
        # try:
        #     X_processed = self.preprocessor.transform(X)
        #     # Note: X_processed is a numpy array after transform
        # except ValueError as e:
        #      self.logger.error(f"Error transforming input data for prediction: {e}. Likely due to missing features.", exc_info=True)
        #      # Create a Series of NaNs aligned to the original index
        #      return pd.Series(np.nan, index=X.index, dtype=float).astype(Int8Dtype()) # Return NaN predictions
        # except Exception as e:
        #      self.logger.error(f"An unexpected error occurred during data transformation for prediction: {e}", exc_info=True)
        #      # Create a Series of NaNs aligned to the original index
        #      return pd.Series(np.nan, index=X.index, dtype=float).astype(Int8Dtype()) # Return NaN predictions


        # Check for NaN/Inf in scaled prediction data after transformation
        # This check is now handled within the pipeline's preprocessor step.
        # if np.isnan(X_processed).any() or np.isinf(X_processed).any():
        #     nan_count = np.isnan(X_processed).sum()
        #     inf_count = np.isinf(X_processed).sum()
        #     error_msg = f"Scaled prediction data contains NaN ({nan_count}) or Inf ({inf_count}) values after preprocessing. Cannot make prediction."
        #     self.logger.critical(error_msg)
        #     # For prediction, return NaNs for the corresponding rows
        #     # Create an empty array for predictions
        #     # Need to map NaNs back to the original index correctly
        #     # This is tricky with sequences. For now, return a Series of NaNs.
        #     return pd.Series(np.nan, index=X.index, dtype=float).astype(Int8Dtype())


        if self.model_type == 'lstm':
             if self.sequence_length <= 0:
                  raise ValueError(f"Invalid sequence_length for LSTM prediction: {self.sequence_length}")

             # For LSTM, we still need to manually prepare sequences *after* preprocessing.
             # So, we still need the scaled data. Let's re-add the preprocessing step here
             # but only for the LSTM case.
             try:
                 X_scaled = self.preprocessor.transform(X)
                 # Note: X_scaled is a numpy array after transform
             except ValueError as e:
                  self.logger.error(f"Error transforming input data for LSTM prediction: {e}. Likely due to missing features.", exc_info=True)
                  return pd.Series(np.nan, index=X.index, dtype=float).astype(Int8Dtype()) # Return NaN predictions
             except Exception as e:
                  self.logger.error(f"An unexpected error occurred during data transformation for LSTM prediction: {e}", exc_info=True)
                  return pd.Series(np.nan, index=X.index, dtype=float).astype(Int8Dtype()) # Return NaN predictions


             # Check for NaN/Inf in scaled prediction data after transformation
             if np.isnan(X_scaled).any() or np.isinf(X_scaled).any():
                 nan_count = np.isnan(X_scaled).sum()
                 inf_count = np.isinf(X_scaled).sum()
                 error_msg = f"Scaled LSTM prediction data contains NaN ({nan_count}) or Inf ({inf_count}) values after preprocessing. Cannot make prediction."
                 self.logger.critical(error_msg)
                 # For prediction, return NaNs for the corresponding rows
                 # Need to map NaNs back to the original index correctly
                 # This is tricky with sequences. For now, return a Series of NaNs.
                 return pd.Series(np.nan, index=X.index, dtype=float).astype(Int8Dtype())


             # Prepare sequences for prediction
             # Use a dummy y_mapped as labels are not needed for prediction sequence creation
             dummy_y_mapped = np.zeros(X_scaled.shape[0], dtype=int)
             X_sequences, _ = self._prepare_lstm_sequences(X_scaled, dummy_y_mapped)

             if X_sequences.shape[0] == 0:
                  self.logger.warning("No sequences generated from input data for LSTM prediction. Returning empty predictions.")
                  # Return a Series with NaNs aligned to the index where sequences would have ended
                  prediction_index = X.index[self.sequence_length - 1:]
                  return pd.Series(np.nan, index=prediction_index, dtype=float).astype(Int8Dtype())


             self.logger.info("Making predictions with Keras LSTM model on sequences...")
             # Get probability predictions first
             y_pred_proba_array = self.model.predict(X_sequences)
             # Get the class with the highest probability
             y_pred_mapped = np.argmax(y_pred_proba_array, axis=1)


             # Convert mapped predictions back to original labels (-1, 0, 1)
             # Handle potential NaNs in y_pred_mapped (from invalid scaled data)
             y_pred_original_values = pd.Series(y_pred_mapped).map(self.inverse_label_map).fillna(0).values # Use fillna(0) for mapping NaNs

             # Align predictions to the original index of X
             # LSTM predictions correspond to the *end* of each sequence.
             # The first prediction is for the time step at index `sequence_length - 1` of the original input data X.
             prediction_index = X.index[self.sequence_length - 1:]

             # Create a full Series with NaNs and fill in the predictions at the correct indices
             aligned_predictions = pd.Series(np.nan, index=X.index, dtype=float) # Use float dtype for NaN initially

             # Ensure the number of predictions matches the number of indices to fill
             if len(y_pred_original_values) == len(prediction_index):
                  aligned_predictions.loc[prediction_index] = y_pred_original_values
                  self.logger.info("LSTM predictions made and aligned.")
             else:
                  self.logger.error(f"Mismatch in length between LSTM predictions ({len(y_pred_original_values)}) and aligned input index ({len(prediction_index)}). Prediction alignment failed.")
                  self.logger.warning("Returning Series with NaNs due to alignment failure.")
                  # The Series is already initialized with NaNs, just return it.


             # Convert to Int8Dtype after filling, if possible (will convert NaNs to pd.NA)
             try:
                 return aligned_predictions.astype(Int8Dtype())
             except Exception as e:
                 self.logger.warning(f"Could not convert aligned predictions to Int8Dtype: {e}. Returning as float Series.")
                 return aligned_predictions # Return as float Series if conversion fails


        else: # Scikit-learn compatible models
            if self.pipeline is None:
                raise RuntimeError("Model pipeline is not loaded or trained. Cannot make predictions.")

            # The pipeline handles preprocessing internally, so pass original X DataFrame
            # The preprocessor within the pipeline will handle subsetting if features_to_use was provided during training
            # *** CORRECTED: Pass original X DataFrame to the pipeline's predict method ***
            y_pred_mapped = self.pipeline.predict(X)

            # Convert mapped predictions back to original labels (-1, 0, 1)
            # Use fillna(0) for mapping NaNs in mapped predictions
            y_pred_original = pd.Series(y_pred_mapped).map(self.inverse_label_map).fillna(0).values

            # Create a Series with the original index
            # Need to handle potential length mismatch if prediction failed for some rows
            # For now, assume predict returns same length as input rows
            if len(y_pred_original) != len(X.index):
                 self.logger.error(f"Prediction output length ({len(y_pred_original)}) does not match input length ({len(X.index)}). Prediction failed.")
                 return pd.Series(np.nan, index=X.index, dtype=float).astype(Int8Dtype()) # Return NaNs

            y_pred_series = pd.Series(y_pred_original, index=X.index, dtype=Int8Dtype()) # Use Int8Dtype

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

        # Pass the original X DataFrame to the preprocessor.
        # The preprocessor will handle selecting the correct features based on how it was fitted.
        # If features_to_use was provided during training, it will only use those.
        # If not, it will use all numeric features it was fitted on.
        # *** REMOVED MANUAL PREPROCESSING FOR SCIKIT-LEARN MODELS ***
        # try:
        #     X_processed = self.preprocessor.transform(X)
        #     # Note: X_processed is a numpy array after transform
        # except ValueError as e:
        #      self.logger.error(f"Error transforming input data for probability prediction: {e}. Likely due to missing features.", exc_info=True)
        #      return None # Return None on transformation error
        # except Exception as e:
        #      self.logger.error(f"An unexpected error occurred during data transformation for probability prediction: {e}", exc_info=True)
        #      return None # Return None on transformation error


        # Check for NaN/Inf in scaled prediction data after transformation
        # This check is now handled within the pipeline's preprocessor step.
        # if np.isnan(X_processed).any() or np.isinf(X_processed).any():
        #     nan_count = np.isnan(X_processed).sum()
        #     inf_count = np.isinf(X_processed).sum()
        #     error_msg = f"Scaled probability prediction data contains NaN ({nan_count}) or Inf ({inf_count}) values after preprocessing. Cannot make probability prediction."
        #     self.logger.critical(error_msg)
        #     return None # Return None on invalid scaled data


        if self.model_type == 'lstm':
             if self.sequence_length <= 0:
                  raise ValueError(f"Invalid sequence_length for LSTM probability prediction: {self.sequence_length}")

             # For LSTM, we still need to manually prepare sequences *after* preprocessing.
             # So, we still need the scaled data. Let's re-add the preprocessing step here
             # but only for the LSTM case.
             try:
                 X_scaled = self.preprocessor.transform(X)
                 # Note: X_scaled is a numpy array after transform
             except ValueError as e:
                  self.logger.error(f"Error transforming input data for LSTM probability prediction: {e}. Likely due to missing features.", exc_info=True)
                  return None # Return None on transformation error
             except Exception as e:
                  self.logger.error(f"An unexpected error occurred during data transformation for LSTM probability prediction: {e}", exc_info=True)
                  return None # Return None on transformation error


             # Check for NaN/Inf in scaled prediction data after transformation
             if np.isnan(X_scaled).any() or np.isinf(X_scaled).any():
                 nan_count = np.isnan(X_scaled).sum()
                 inf_count = np.isinf(X_scaled).sum()
                 error_msg = f"Scaled LSTM probability prediction data contains NaN ({nan_count}) or Inf ({inf_count}) values after preprocessing. Cannot make probability prediction."
                 self.logger.critical(error_msg)
                 return None # Return None on invalid scaled data


             # Prepare sequences for prediction
             dummy_y_mapped = np.zeros(X_scaled.shape[0], dtype=int)
             X_sequences, _ = self._prepare_lstm_sequences(X_scaled, dummy_y_mapped)

             if X_sequences.shape[0] == 0:
                  self.logger.warning("No sequences generated from input data for LSTM probability prediction. Returning None.")
                  return None

             self.logger.info("Making probability predictions with Keras LSTM model on sequences...")
             y_pred_proba_array = self.model.predict(X_sequences)

             self.logger.info("LSTM probability predictions made.")
             # Align probabilities to the original index of X
             # Probabilities correspond to the *end* of each sequence.
             prediction_index = X.index[self.sequence_length - 1:]

             # Create a DataFrame with NaNs and fill in the probabilities at the correct indices
             # Columns should be original labels: -1, 0, 1
             aligned_probabilities_df = pd.DataFrame(np.nan, index=X.index, columns=self.classes.tolist(), dtype=float)

             # Ensure the number of probability rows matches the number of indices to fill
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
                # The pipeline handles preprocessing internally, so pass original X DataFrame
                # *** CORRECTED: Pass original X DataFrame to the pipeline's predict_proba method ***
                y_pred_proba_array = self.pipeline.predict_proba(X)
                self.logger.info(f"{self.model_type} probability predictions made.")
                # Return as DataFrame with original labels as columns
                # Need to handle potential length mismatch if prediction failed for some rows
                if len(y_pred_proba_array) != len(X.index):
                     self.logger.error(f"Probability prediction output length ({len(y_pred_proba_array)}) does not match input length ({len(X.index)}). Prediction failed.")
                     return None # Return None on length mismatch

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
            ImportError: If DataManager is not available.
        """
        if self.model is None and self.pipeline is None:
            raise RuntimeError("No model or pipeline trained/loaded to save.")
        if self.preprocessor is None:
             # Preprocessor is essential for prediction/loading, so it must be available
             raise RuntimeError("Preprocessor is not trained/loaded. Cannot save model.")
        if DataManager is None:
             raise ImportError("DataManager is not available. Cannot save model artifacts.")


        self.logger.info(f"Saving trained {self.model_type} model and metadata for {symbol.upper()} {interval} using DataManager...")

        # Prepare metadata dictionary
        metadata = {
            'model_type': self.model_type,
            # Store the feature columns that the preprocessor was fitted on (processed names)
            'feature_columns_processed': self.feature_columns_processed, # Use processed attribute name
            # Store the original feature columns from the training data
            'feature_columns_original': self.feature_columns_original, # Save original feature names
            'label_map': self.label_map,
            'inverse_label_map': self.inverse_label_map,
            'classes': self.classes.tolist(),
            'model_params': self.model_params, # Store the parameters used to initialize the model
            'sequence_length_bars': self.sequence_length, # Store sequence length for LSTM
            'save_timestamp': datetime.now().isoformat(), # Record save time
            'features_to_use': self.features_to_use # Save the optional feature subset used
        }

        try:
            # Save Metadata using DataManager
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
            # This is a warning, as the model file itself might still be savable, but loading might be an issue
            self.logger.warning("Failed to save model metadata. Model might not be loadable correctly.")


        try:
            if self.model_type == 'lstm':
                 # Save Keras model using DataManager
                 # DataManager.save_model_artifact handles the file extension (.keras)
                 self.dm.save_model_artifact(
                     artifact=self.model,
                     symbol=symbol,
                     interval=interval,
                     model_key=model_key,
                     artifact_type='model' # Use 'model' type for the Keras model file
                 )
                 self.logger.info(f"Keras LSTM model saved successfully via DataManager.")

                 # Save Preprocessor using DataManager
                 # DataManager.save_model_artifact handles the file extension (.joblib)
                 self.dm.save_model_artifact(
                     artifact=self.preprocessor,
                     symbol=symbol,
                     interval=interval,
                     model_key=model_key,
                     artifact_type='preprocessor' # Use 'preprocessor' type for the fitted preprocessor
                 )
                 self.logger.info(f"Preprocessor saved successfully via DataManager.")


            else: # Scikit-learn compatible models (pipeline)
                 # Save pipeline using DataManager
                 # DataManager.save_model_artifact handles the file extension (.joblib)
                 self.dm.save_model_artifact(
                     artifact=self.pipeline,
                     symbol=symbol,
                     interval=interval,
                     model_key=model_key,
                     artifact_type='pipeline' # Use 'pipeline' type for the scikit-learn pipeline
                 )
                 self.logger.info(f"{self.model_type} pipeline saved successfully via DataManager.")

                 # For scikit-learn pipelines, the preprocessor is part of the pipeline,
                 # so we don't need to save it separately.
                 self.logger.debug("Preprocessor is part of the scikit-learn pipeline, not saved separately.")


        except Exception as e:
            self.logger.error(f"Failed to save model artifact(s) via DataManager: {e}", exc_info=True)
            # Re-raise the exception after logging
            raise OSError(f"Failed to save model artifact(s): {e}")


        self.logger.info(f"Model saving process completed for {self.model_type}.")


    def load(self, symbol: str, interval: str, model_key: str) -> 'ModelTrainer':
        """
        Loads a trained model (pipeline or Keras model) and its metadata using DataManager.
        Updates the current ModelTrainer instance with the loaded components.

        Args:
            symbol (str): Trading pair symbol.
            interval (str): Time interval.
            model_key (str): Key for the model configuration in config.params.MODEL_CONFIG.

        Returns:
            ModelTrainer: The current instance, updated with the loaded model and metadata.

        Raises:
            FileNotFoundError: If the model or metadata file is not found (raised by DataManager).
            RuntimeError: If loading fails for other reasons.
            ValueError: If model_key is invalid or metadata is missing crucial info.
            ImportError: If TensorFlow is required but not available, or DataManager is not available.
        """
        # Use instance-specific logger for loading messages
        self.logger.info(f"Loading trained model for {symbol.upper()} {interval} ({self.model_type}) using DataManager...")

        if DataManager is None:
             raise ImportError("DataManager is not available. Cannot load model artifacts.")

        # --- Load Metadata using DataManager ---
        self.logger.info(f"Loading metadata for {self.model_type} model... from {symbol.upper()} {interval} {model_key}")
        try:
            metadata = self.dm.load_model_artifact(
                symbol=symbol,
                interval=interval,
                model_key=model_key,
                artifact_type='metadata'
            )
            self.logger.info("Metadata loaded successfully via DataManager.")

            # Update instance attributes from metadata
            self.model_type = metadata.get('model_type', self.model_type) # Use loaded type, fallback to initialized
            # Load feature_columns_processed from metadata
            self.feature_columns_processed = metadata.get('feature_columns_processed') # Load processed names
            # Load original_feature_names from metadata
            self.feature_columns_original = metadata.get('feature_columns_original') # Load original names
            self.label_map = metadata.get('label_map', self.label_map)
            self.inverse_label_map = metadata.get('inverse_label_map', self.inverse_label_map)
            self.classes = np.array(metadata.get('classes', [-1, 0, 1])) # Default to [-1, 0, 1] if not in metadata
            self.model_params = metadata.get('model_params', {}) # Load saved model parameters
            # Use updated key name for sequence length, fallback to old key or default
            self.sequence_length = metadata.get('sequence_length_bars', metadata.get('sequence_length', 1))
            # Load the optional features_to_use list from metadata
            self.features_to_use = metadata.get('features_to_use')

            # Log warnings if crucial metadata is missing
            if self.feature_columns_processed is None:
                 self.logger.warning("Processed feature columns not found in metadata.")
            if self.feature_columns_original is None:
                 self.logger.warning("Original feature columns not found in metadata.")
            if self.label_map is None or self.inverse_label_map is None or self.classes is None or len(self.classes) == 0:
                 self.logger.warning(f"Label mapping or classes not found or empty in metadata. Using defaults: labels={self.classes}, map={self.label_map}, inverse={self.inverse_label_map}.")


        except FileNotFoundError:
            # Re-raise FileNotFoundError as it's specific and handled by caller
            raise
        except Exception as e:
            self.logger.error(f"Error loading model metadata via DataManager: {e}", exc_info=True)
            raise RuntimeError(f"Failed to load model metadata: {e}")


        # --- Load Model / Pipeline using DataManager ---
        self.logger.info(f"Loading {self.model_type} model artifact...")
        try:
            if self.model_type == 'lstm':
                 if not LSTM_AVAILABLE:
                      raise ImportError("TensorFlow is required to load LSTM model but is not installed.")

                 # Load Keras model using DataManager
                 # DataManager.load_model_artifact handles the file extension (.keras)
                 self.model = self.dm.load_model_artifact(
                     symbol=symbol,
                     interval=interval,
                     model_key=model_key,
                     artifact_type='model' # Use 'model' type for the Keras model file
                 )
                 self.logger.info("Keras LSTM model loaded successfully via DataManager.")

                 # Load Preprocessor using DataManager
                 self.logger.info(f"Attempting to load preprocessor for {self.model_type} model...")
                 try:
                      self.preprocessor = self.dm.load_model_artifact(
                          symbol=symbol,
                          interval=interval,
                          model_key=model_key,
                          artifact_type='preprocessor' # Use 'preprocessor' type
                      )
                      self.logger.info("Preprocessor loaded successfully via DataManager.")
                      # The feature_columns_processed should ideally be loaded from metadata,
                      # but can be inferred from the loaded preprocessor as a fallback.
                      if self.feature_columns_processed is None:
                           if hasattr(self.preprocessor, 'get_feature_names_out'):
                                try:
                                     self.feature_columns_processed = self.preprocessor.get_feature_names_out().tolist()
                                     self.logger.info(f"Inferred processed feature columns from loaded preprocessor: {self.feature_columns_processed}")
                                except Exception as e:
                                     self.logger.warning(f"Error getting feature names out from preprocessor: {e}")
                                     if hasattr(self.preprocessor, 'feature_names_in_'):
                                          # Fallback to feature_names_in_ if get_feature_names_out fails
                                          # Note: feature_names_in_ might be original names or subsetted names depending on preprocessor
                                          self.feature_columns_processed = self.preprocessor.feature_names_in_.tolist() if isinstance(self.preprocessor.feature_names_in_, np.ndarray) else self.preprocessor.feature_names_in_
                                          self.logger.info(f"Inferred processed feature columns from preprocessor.feature_names_in_: {self.feature_columns_processed}")
                                     else:
                                          self.logger.warning("Could not infer processed feature columns from loaded preprocessor. Using feature columns from metadata if available.")
                                          # If feature_columns_processed was None in metadata too, it remains None.
                           elif hasattr(self.preprocessor, 'feature_names_in_'):
                                # Use feature_names_in_ if get_feature_names_out is not available
                                # Note: feature_names_in_ might be original names or subsetted names depending on preprocessor
                                self.feature_columns_processed = self.preprocessor.feature_names_in_.tolist() if isinstance(self.preprocessor.feature_names_in_, np.ndarray) else self.preprocessor.feature_names_in_
                                self.logger.info(f"Inferred processed feature columns from preprocessor.feature_names_in_: {self.feature_columns_processed}")

                           elif self.feature_columns_processed is not None:
                                self.logger.info("Using processed feature columns loaded from metadata.")
                           else:
                                self.logger.warning("Could not infer processed feature columns from preprocessor or metadata.")
                      else:
                           self.logger.info("Using processed feature columns loaded from metadata.")

                      # The original feature names should be loaded from metadata
                      if self.feature_columns_original is None:
                           self.logger.warning("Original feature columns not loaded from metadata.")


                 except FileNotFoundError:
                      self.logger.warning("Preprocessor artifact not found. LSTM predictions/evaluation might fail.")
                      self.preprocessor = None # Set to None if file not found
                 except Exception as e:
                      self.logger.warning(f"Error loading preprocessor via DataManager: {e}", exc_info=True)
                      self.preprocessor = None # Set to None if loading fails


            else: # Load Scikit-learn pipeline using DataManager
                 # DataManager.load_model_artifact handles the file extension (.joblib)
                 self.pipeline = self.dm.load_model_artifact(
                     symbol=symbol,
                     interval=interval,
                     model_key=model_key,
                     artifact_type='pipeline' # Use 'pipeline' type
                 )
                 self.logger.info(f"{self.model_type} pipeline loaded successfully via DataManager.")
                 # Extract preprocessor and model from the loaded pipeline
                 if self.pipeline is not None and len(self.pipeline.steps) > 0:
                      # Assuming preprocessor is the first step
                      self.preprocessor = self.pipeline.steps[0][1]
                      # Assuming model is the last step
                      self.model = self.pipeline.steps[-1][1]

                      # Attempt to infer processed feature columns from the loaded preprocessor if not loaded from metadata
                      if self.feature_columns_processed is None:
                           if hasattr(self.preprocessor, 'get_feature_names_out'):
                                try:
                                     self.feature_columns_processed = self.preprocessor.get_feature_names_out().tolist()
                                     self.logger.info(f"Inferred processed feature columns from loaded preprocessor: {self.feature_columns_processed}")
                                except Exception as e:
                                     self.logger.warning(f"Error getting feature names from preprocessor: {e}", exc_info=True)
                                     if hasattr(self.preprocessor, 'feature_names_in_'):
                                          # Fallback to feature_names_in_ if get_feature_names_out fails
                                          # Note: feature_names_in_ might be original names or subsetted names depending on preprocessor
                                          self.feature_columns_processed = self.preprocessor.feature_names_in_.tolist() if isinstance(self.preprocessor.feature_names_in_, np.ndarray) else self.preprocessor.feature_names_in_
                                          self.logger.info(f"Inferred processed feature columns from preprocessor.feature_names_in_: {self.feature_columns_processed}")
                                     else:
                                          self.logger.warning("Could not infer processed feature columns from preprocessor. Using feature columns from metadata if available.")
                           elif hasattr(self.preprocessor, 'feature_names_in_'):
                                # Use feature_names_in_ if get_feature_names_out is not available
                                # Note: feature_names_in_ might be original names or subsetted names depending on preprocessor
                                self.feature_columns_processed = self.preprocessor.feature_names_in_.tolist() if isinstance(self.preprocessor.feature_names_in_, np.ndarray) else self.preprocessor.feature_names_in_
                                self.logger.info(f"Inferred processed feature columns from preprocessor.feature_names_in_: {self.feature_columns_processed}")

                           elif self.feature_columns_processed is not None:
                                self.logger.info("Using processed feature columns loaded from metadata.")
                           else:
                                self.logger.warning("Could not infer processed feature columns from preprocessor or metadata.")
                      else:
                           self.logger.info("Using processed feature columns loaded from metadata.")

                      # The original feature names should be loaded from metadata
                      if self.feature_columns_original is None:
                           self.logger.warning("Original feature columns not loaded from metadata.")


                 else:
                      self.logger.warning("Loaded pipeline has no steps. Preprocessor and model not extracted.")
                      self.preprocessor = None
                      self.model = None

        except FileNotFoundError:
             # Re-raise FileNotFoundError as it's specific and handled by caller
             raise
        except Exception as e:
            self.logger.error(f"An unexpected error occurred during model artifact loading via DataManager: {e}", exc_info=True)
            raise RuntimeError(f"Failed to load trained model artifact: {e}")


        # Final check for essential components after loading
        if self.preprocessor is None:
             self.logger.error("Preprocessor could not be loaded. Model might not work correctly for prediction/analysis.")
             # Depending on requirements, you might want to raise an error here
             # raise RuntimeError("Failed to load preprocessor.")
        if self.model is None and self.pipeline is None:
             self.logger.error("Model or pipeline could not be loaded. Model is not usable.")
             # Depending on requirements, you might want to raise an error here
             # raise RuntimeError("Failed to load model or pipeline.")
        if self.feature_columns_original is None:
             self.logger.error("Original feature columns could not be loaded. Cannot prepare data for analysis.")
             # Depending on requirements, you might want to raise an error here
             # raise RuntimeError("Failed to load original feature columns.")


        self.logger.info(f"ModelTrainer instance loaded successfully for {self.model_type}.")
        return self

