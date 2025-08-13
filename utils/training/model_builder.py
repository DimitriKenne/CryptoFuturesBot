# utils/training/model_builder.py

import logging
from typing import Optional, Dict, Any, Union
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

# Relative import for config schema, as it's outside the 'training' package
from config.model_config_schema import LSTMParams, RandomForestParams, XGBoostParams

# TF_AVAILABLE and tf should be passed in the constructor.
# No direct import here.
# Assuming they are passed or globally available via app_config in the main training script context.

logger = logging.getLogger(__name__)


class ModelBuilder:
    """
    Builds different machine learning model architectures based on provided configurations.
    Supports RandomForest, XGBoost, and Keras LSTM models.
    """
    def __init__(self, tf_available: bool, tf_module: Optional[Any]):
        """
        Initializes the ModelBuilder with TensorFlow availability status and module.

        Args:
            tf_available (bool): Indicates if TensorFlow is available.
            tf_module (Optional[Any]): The TensorFlow module itself, or None if not available.
        """
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self._tf_available = tf_available
        self._tf = tf_module
        self.logger.info(f"ModelBuilder initialized. TensorFlow available: {self._tf_available}")


    def build_model(
        self,
        model_type: str,
        model_params: Union[RandomForestParams, XGBoostParams, LSTMParams, Dict[str, Any]], # Moved before n_features
        general_config_random_seed: int,
        general_config_n_processors: int,
        n_features: Optional[int] = None # Moved after non-default args
    ) -> Any: # Returns BaseEstimator or KerasModel
        """
        Builds the specified model based on its type and parameters.

        Args:
            model_type (str): The type of model to build ('random_forest', 'xgboost', 'lstm').
            model_params (Union[RandomForestParams, XGBoostParams, LSTMParams, Dict[str, Any]]):
                        Parameters for the specific model. Can be a dataclass instance or a dict.
            general_config_random_seed (int): Random seed from general config.
            general_config_n_processors (int): Number of processors from general config.
            n_features (Optional[int]): Required number of features for LSTM input layer.
                                        Not used for other model types.

        Returns:
            Union[RandomForestClassifier, XGBClassifier, tf.keras.Model]: The built model.

        Raises:
            ValueError: If the model_type is unsupported or parameters are invalid.
            ImportError: If TensorFlow is required but not available.
        """
        self.logger.info(f"Building model: {model_type}")

        if model_type == 'random_forest':
            # Ensure model_params is RandomForestParams type
            rf_params: RandomForestParams = model_params if isinstance(model_params, RandomForestParams) else RandomForestParams(**model_params)

            # Create a mutable dict from dataclass for model init
            model_init_params = rf_params.__dict__.copy()
            model_init_params.pop('class_balancing', None) # Class balancing is handled by imblearn pipeline
            model_init_params.pop('random_state', None) # Handled globally
            model_init_params.pop('n_jobs', None) # Handled globally

            model = RandomForestClassifier(
                random_state=general_config_random_seed,
                n_jobs=general_config_n_processors,
                **model_init_params
            )
            self.logger.info("RandomForestClassifier built.")
            return model

        elif model_type == 'xgboost':
            # Ensure model_params is XGBoostParams type
            xgb_params: XGBoostParams = model_params if isinstance(model_params, XGBoostParams) else XGBoostParams(**model_params)

            # Create a mutable dict from dataclass for model init
            model_init_params = xgb_params.__dict__.copy()
            model_init_params.pop('class_balancing', None) # Class balancing is handled by imblearn pipeline
            model_init_params.pop('random_state', None) # Handled globally
            model_init_params.pop('n_jobs', None) # Handled globally

            # XGBoost specific setup for ternary classification
            final_xgb_params = {
                'objective': 'multi:softmax',
                'num_class': 3,
                'eval_metric': 'mlogloss',
                'random_state': general_config_random_seed,
                'n_jobs': general_config_n_processors,
                **model_init_params
            }
            model = XGBClassifier(**final_xgb_params)
            self.logger.info("XGBClassifier built.")
            return model

        elif model_type == 'lstm':
            if not self._tf_available or self._tf is None:
                raise ImportError("TensorFlow is not available to build LSTM model.")
            if n_features is None or n_features <= 0:
                raise ValueError("n_features must be a positive integer for LSTM model.")

            # Ensure model_params is LSTMParams type
            lstm_params: LSTMParams = model_params if isinstance(model_params, LSTMParams) else LSTMParams(**model_params)

            sequence_length = lstm_params.sequence_length_bars
            n_layers = lstm_params.n_layers
            units_per_layer = lstm_params.units_per_layer
            dropout_rate = lstm_params.dropout_rate
            learning_rate = lstm_params.learning_rate
            clipnorm = lstm_params.clipnorm
            clipvalue = lstm_params.clipvalue

            if sequence_length <= 0:
                raise ValueError("LSTM model parameter 'sequence_length_bars' must be positive.")
            if n_layers <= 0:
                raise ValueError("Number of LSTM layers (n_layers) must be positive.")
            if units_per_layer <= 0:
                raise ValueError("Units per LSTM layer (units_per_layer) must be positive.")
            if not (0.0 <= dropout_rate <= 1.0):
                raise ValueError("Dropout rate must be between 0.0 and 1.0.")
            if learning_rate <= 0:
                raise ValueError("Learning rate must be positive.")

            model = self._tf.keras.models.Sequential()
            model.add(self._tf.keras.layers.Input(shape=(sequence_length, n_features)))

            for i in range(n_layers):
                return_sequences = i < n_layers - 1
                model.add(self._tf.keras.layers.LSTM(units_per_layer, return_sequences=return_sequences))
                if dropout_rate > 0:
                    model.add(self._tf.keras.layers.Dropout(dropout_rate))

            model.add(self._tf.keras.layers.Dense(3, activation='softmax'))

            optimizer_params = {'learning_rate': learning_rate}
            if clipnorm is not None:
                optimizer_params['clipnorm'] = clipnorm
                self.logger.info(f"Using gradient clipping (clipnorm={clipnorm}) in Adam optimizer.")
            if clipvalue is not None:
                optimizer_params['clipvalue'] = clipvalue
                self.logger.info(f"Using gradient clipping (clipvalue={clipvalue}) in Adam optimizer.")

            optimizer = self._tf.keras.optimizers.Adam(**optimizer_params) # type: ignore

            model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

            self.logger.info("LSTM model architecture built and compiled.")
            model.summary(print_fn=lambda x: self.logger.info(x)) # type: ignore
            return model
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
