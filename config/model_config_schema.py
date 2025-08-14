# config/model_config_schema.py

from dataclasses import dataclass, field
from typing import Dict, Any, List, Literal, Optional, Union # Added Union for type hints
import logging

# Conditional Import for Hyperparameter Tuning Distributions
# Need to install scipy: pip install scipy
try:
    from scipy.stats import uniform, randint
    SCIPY_AVAILABLE = True
except ImportError:
    uniform = None
    randint = None
    SCIPY_AVAILABLE = False
    logging.warning("Scipy not found. Hyperparameter tuning distributions (uniform, randint) will not be available.")

# Conditional Import for TensorFlow/Keras for LSTM Availability Check
# These are module-level variables
_TF_AVAILABLE_MODULE_VAR = False # Renamed to avoid confusion with dataclass attribute
_TF_MODULE_VAR = None # Renamed to avoid confusion with dataclass attribute
try:
    import tensorflow as tf
    if tf.config.list_physical_devices('GPU'):
        logging.info("TensorFlow with GPU support is available for model_config_schema.")
        _TF_AVAILABLE_MODULE_VAR = True
    else:
        logging.info("TensorFlow is available, but no GPU found. Using CPU for model_config_schema.")
        _TF_AVAILABLE_MODULE_VAR = True
    _TF_MODULE_VAR = tf # Assign the imported TensorFlow module
except ImportError:
    logging.warning("TensorFlow not found. LSTM model training/loading will not be available.")
except Exception as e:
    logging.warning(f"Error checking TensorFlow availability in model_config_schema: {e}. LSTM model training/loading may not be available.")


@dataclass
class XGBoostParams:
    objective: str = 'multi:softprob'
    num_class: int = 3 # Long, Neutral, Short
    eval_metric: str = 'mlogloss'
    n_estimators: int = 500 # Number of boosting rounds
    learning_rate: float = 0.05
    max_depth: int = 7
    subsample: float = 0.7
    colsample_bytree: float = 0.7
    # Removed use_label_encoder as it's deprecated and can cause warnings
    enable_categorical: bool = True # Use native categorical feature support
    n_jobs: int = -1 # Default to all processors, will be overridden by GeneralConfig.n_processors
    tree_method: str = 'hist' # Faster for large datasets
    early_stopping_rounds: Optional[int] = None # Set to None by default, only active if specified and eval_set provided
    class_balancing: Optional[Union[Literal['undersampling', 'oversampling', 'balanced'], Dict[str, Any]]] = None


@dataclass
class RandomForestParams:
    n_estimators: int = 300
    max_depth: int = 20
    min_samples_leaf: int = 5
    min_samples_split: int = 10
    random_state: int = 42 # Will be overridden by GeneralConfig.random_seed
    n_jobs: int = -1 # Default to all processors, will be overridden by GeneralConfig.n_processors
    oob_score: bool = True # Out-of-bag samples to estimate generalization accuracy
    class_balancing: Optional[Union[Literal['undersampling', 'oversampling', 'balanced'], Dict[str, Any]]] = None


@dataclass
class LSTMParams:
    input_timesteps: int = 5 # Must match 'sequence_length_bars' in FeatureConfig
    n_features: Optional[int] = None # Will be set dynamically based on data shape
    units_per_layer: int = 50 # Number of LSTM units per layer
    n_layers: int = 1 # Number of LSTM layers
    epochs: int = 50
    batch_size: int = 32
    validation_split: float = 0.2
    dropout_rate: float = 0.2 # Renamed from 'dropout' for clarity
    learning_rate: float = 0.001
    clipnorm: Optional[float] = 1.0 # Gradient clipping by norm (e.g., 1.0)
    clipvalue: Optional[float] = None # Gradient clipping by value (e.g., 0.5)
    early_stopping_patience: Optional[int] = 10 # For EarlyStopping
    reduce_lr_on_plateau_factor: Optional[float] = 0.1 # Factor by which LR will be reduced
    reduce_lr_on_plateau_patience: Optional[int] = 5 # Number of epochs with no improvement after which LR will be reduced
    class_balancing: Optional[Union[Literal['balanced'], Dict[str, Any]]] = None # For LSTM, typically 'balanced' or custom weights


@dataclass
class XGBoostTuningParams:
    n_estimators: Any = field(default_factory=lambda: randint(100, 1000) if SCIPY_AVAILABLE else None)
    learning_rate: Any = field(default_factory=lambda: uniform(0.01, 0.2) if SCIPY_AVAILABLE else None)
    max_depth: Any = field(default_factory=lambda: randint(3, 10) if SCIPY_AVAILABLE else None)
    subsample: Any = field(default_factory=lambda: uniform(0.6, 0.4) if SCIPY_AVAILABLE else None)
    colsample_bytree: Any = field(default_factory=lambda: uniform(0.6, 0.4) if SCIPY_AVAILABLE else None)


@dataclass
class RandomForestTuningParams:
    n_estimators: Any = field(default_factory=lambda: randint(100, 500) if SCIPY_AVAILABLE else None)
    max_depth: Any = field(default_factory=lambda: randint(10, 30) if SCIPY_AVAILABLE else None)
    min_samples_leaf: Any = field(default_factory=lambda: randint(1, 10) if SCIPY_AVAILABLE else None)
    min_samples_split: Any = field(default_factory=lambda: randint(2, 20) if SCIPY_AVAILABLE else None)


@dataclass
class ModelConfig:
    """
    Defines configuration parameters for model training.
    """
    # Define a class-level attribute for available model types
    AVAILABLE_MODEL_TYPES: Dict[str, str] = field(
        default_factory=lambda: {
            'xgboost': 'XGBoost Classifier',
            'random_forest': 'Random Forest Classifier',
            'lstm': 'LSTM Neural Network'
        },
        init=False, # This field is not part of the constructor
        repr=False  # Do not include in string representation
    )

    model_type: Literal['xgboost', 'random_forest', 'lstm'] = 'xgboost'
    features_to_use: Optional[List[str]] = None # List of feature names to use, or None for all generated features
    label_column: str = 'label' # Name of the target variable column
    train_test_split_ratio: float = 0.8 # Ratio for train/test split (time-series split)
    scaler_type: Optional[Literal['standard', 'minmax']] = 'standard' # 'standard', 'minmax', None
    pca_enabled: bool = False # Whether to apply PCA dimensionality reduction
    pca_n_components: float = 0.95 # float (0-1, variance explained) or int (number of components)
    tuning_scoring_metric: str = 'f1_macro' # Metric for hyperparameter tuning evaluation

    # Add TF_AVAILABLE and tf as attributes of ModelConfig
    # Initialize them directly from the module-level variables
    TF_AVAILABLE: bool = field(default=_TF_AVAILABLE_MODULE_VAR, init=False, repr=False)
    tf: Any = field(default=_TF_MODULE_VAR, init=False, repr=False)

    # Nested model-specific parameters
    xgboost_params: XGBoostParams = field(default_factory=XGBoostParams)
    random_forest_params: RandomForestParams = field(default_factory=RandomForestParams)
    lstm_params: LSTMParams = field(default_factory=LSTMParams)

    # Nested tuning parameters
    xgboost_tuning_params: XGBoostTuningParams = field(default_factory=XGBoostTuningParams)
    random_forest_tuning_params: RandomForestTuningParams = field(default_factory=RandomForestTuningParams)

    def __post_init__(self):
        if self.model_type not in self.AVAILABLE_MODEL_TYPES: # Validate against the new class attribute
            raise ValueError(f"model_type must be one of {list(self.AVAILABLE_MODEL_TYPES.keys())}.")
        if self.features_to_use is not None and not isinstance(self.features_to_use, list):
            raise TypeError("features_to_use must be a list of strings or None.")
        if not isinstance(self.label_column, str) or not self.label_column:
            raise ValueError("label_column must be a non-empty string.")
        if not isinstance(self.train_test_split_ratio, float) or not (0 < self.train_test_split_ratio < 1):
            raise ValueError("train_test_split_ratio must be a float between 0 and 1 (exclusive).")
        if self.scaler_type not in ['standard', 'minmax', None]:
            raise ValueError("scaler_type must be 'standard', 'minmax', or None.")
        if not isinstance(self.pca_enabled, bool):
            raise TypeError("pca_enabled must be a boolean.")
        if not isinstance(self.pca_n_components, (int, float)) or (isinstance(self.pca_n_components, float) and not (0 < self.pca_n_components <= 1)):
            raise ValueError("pca_n_components must be a positive integer or a float between 0 and 1.")

        # Ensure nested configs are instantiated from their dicts if passed as such
        if isinstance(self.xgboost_params, dict): self.xgboost_params = XGBoostParams(**self.xgboost_params)
        if isinstance(self.random_forest_params, dict): self.random_forest_params = RandomForestParams(**self.random_forest_params)
        if isinstance(self.lstm_params, dict): self.lstm_params = LSTMParams(**self.lstm_params)
        if isinstance(self.xgboost_tuning_params, dict): self.xgboost_tuning_params = XGBoostTuningParams(**self.xgboost_tuning_params)
        if isinstance(self.random_forest_tuning_params, dict): self.random_forest_tuning_params = RandomForestTuningParams(**self.random_forest_tuning_params)


# Default configuration instance
DEFAULT_MODEL_CONFIG = ModelConfig()
