# config/model_config_schema.py

from dataclasses import dataclass, field
from typing import Dict, Any, List, Literal, Optional
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
TF_AVAILABLE = False
try:
    import tensorflow as tf
    if tf.config.list_physical_devices('GPU'):
        logging.info("TensorFlow with GPU support is available for model_config_schema.")
        TF_AVAILABLE = True
    else:
        logging.info("TensorFlow is available, but no GPU found. Using CPU for model_config_schema.")
        TF_AVAILABLE = True
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
    use_label_encoder: bool = False # Suppress warning
    n_jobs: int = -1 # Default to all processors, will be overridden by GeneralConfig.n_processors
    tree_method: str = 'hist' # Faster for large datasets
    early_stopping_rounds: int = 50 # For early stopping during training

@dataclass
class RandomForestParams:
    n_estimators: int = 300
    max_depth: int = 20
    min_samples_leaf: int = 5
    min_samples_split: int = 10
    random_state: int = 42 # Will be overridden by GeneralConfig.random_seed
    n_jobs: int = -1 # Default to all processors, will be overridden by GeneralConfig.n_processors
    oob_score: bool = True # Out-of-bag samples to estimate generalization accuracy

@dataclass
class LSTMParams:
    input_timesteps: int = 5 # Must match 'sequence_length_bars' in FeatureConfig
    n_features: Optional[int] = None # Will be set dynamically based on data shape
    units: int = 50 # Number of LSTM units
    epochs: int = 50
    batch_size: int = 32
    validation_split: float = 0.2
    dropout: float = 0.2
    optimizer: str = 'adam'
    loss: str = 'sparse_categorical_crossentropy'
    metrics: List[str] = field(default_factory=lambda: ['accuracy'])
    patience: int = 10 # For EarlyStopping


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
    model_type: Literal['xgboost', 'random_forest', 'lstm'] = 'xgboost'
    features_to_use: Optional[List[str]] = None # List of feature names to use, or None for all generated features
    label_column: str = 'label' # Name of the target variable column
    train_test_split_ratio: float = 0.8 # Ratio for train/test split (time-series split)
    scaler_type: Optional[Literal['standard', 'minmax']] = 'standard' # 'standard', 'minmax', None
    pca_enabled: bool = False # Whether to apply PCA dimensionality reduction
    pca_n_components: float = 0.95 # float (0-1, variance explained) or int (number of components)
    
    # Nested model-specific parameters
    xgboost_params: XGBoostParams = field(default_factory=XGBoostParams)
    random_forest_params: RandomForestParams = field(default_factory=RandomForestParams)
    lstm_params: LSTMParams = field(default_factory=LSTMParams)

    # Nested tuning parameters
    xgboost_tuning_params: XGBoostTuningParams = field(default_factory=XGBoostTuningParams)
    random_forest_tuning_params: RandomForestTuningParams = field(default_factory=RandomForestTuningParams)
    # LSTM tuning is more complex, not included here by default

    def __post_init__(self):
        if self.model_type not in ['xgboost', 'random_forest', 'lstm']:
            raise ValueError("model_type must be 'xgboost', 'random_forest', or 'lstm'.")
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
        # (This is handled by the overall config loading, but good for direct instantiation)
        if isinstance(self.xgboost_params, dict): self.xgboost_params = XGBoostParams(**self.xgboost_params)
        if isinstance(self.random_forest_params, dict): self.random_forest_params = RandomForestParams(**self.random_forest_params)
        if isinstance(self.lstm_params, dict): self.lstm_params = LSTMParams(**self.lstm_params)
        if isinstance(self.xgboost_tuning_params, dict): self.xgboost_tuning_params = XGBoostTuningParams(**self.xgboost_tuning_params)
        if isinstance(self.random_forest_tuning_params, dict): self.random_forest_tuning_params = RandomForestTuningParams(**self.random_forest_tuning_params)


# Default configuration instance
DEFAULT_MODEL_CONFIG = ModelConfig()
