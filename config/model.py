# config/model_config_schema.py

from dataclasses import dataclass, field
from typing import Dict, Any, List, Literal, Optional, Union
import logging

# Optional: Hyperparameter tuning distributions
try:
    from scipy.stats import uniform, randint
    SCIPY_AVAILABLE = True
except ImportError:
    uniform = None
    randint = None
    SCIPY_AVAILABLE = False

# Optional: TensorFlow/Keras for LSTM
try:
    import tensorflow as tf
    TF_AVAILABLE = True
except ImportError:
    tf = None
    TF_AVAILABLE = False

@dataclass
class XGBoostParams:
    objective: str = 'multi:softprob'
    num_class: int = 3
    eval_metric: str = 'mlogloss'
    n_estimators: int = 500
    learning_rate: float = 0.05
    max_depth: int = 7
    subsample: float = 0.7
    colsample_bytree: float = 0.7
    enable_categorical: bool = True
    n_jobs: int = -1
    tree_method: str = 'hist'
    early_stopping_rounds: Optional[int] = None
    class_balancing: Optional[Union[str, Dict[str, Any]]] = None

@dataclass
class RandomForestParams:
    n_estimators: int = 300
    max_depth: int = 20
    min_samples_leaf: int = 5
    min_samples_split: int = 10
    random_state: int = 42
    n_jobs: int = -1
    oob_score: bool = True
    class_balancing: Optional[Union[str, Dict[str, Any]]] = None

@dataclass
class LSTMParams:
    sequence_length_bars: int = 5
    n_features: Optional[int] = None
    units_per_layer: int = 50
    n_layers: int = 1
    epochs: int = 50
    batch_size: int = 32
    validation_split: float = 0.2
    dropout_rate: float = 0.2
    learning_rate: float = 0.001
    clipnorm: Optional[float] = 1.0
    clipvalue: Optional[float] = None
    early_stopping_patience: Optional[int] = 10
    reduce_lr_on_plateau_factor: Optional[float] = 0.1
    reduce_lr_on_plateau_patience: Optional[int] = 5
    class_balancing: Optional[Union[str, Dict[str, Any]]] = None

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
    Model training configuration.
    """
    AVAILABLE_MODEL_TYPES: Dict[str, str] = field(
        default_factory=lambda: {
            'xgboost': 'XGBoost Classifier',
            'random_forest': 'Random Forest Classifier',
            'lstm': 'LSTM Neural Network'
        },
        init=False, repr=False
    )
    model_type: Literal['xgboost', 'random_forest', 'lstm'] = 'xgboost'
    features_to_use: Optional[List[str]] = None
    label_column: str = 'label'
    train_test_split_ratio: float = 0.8
    scaler_type: Optional[Literal['standard', 'minmax']] = 'standard'
    pca_enabled: bool = False
    pca_n_components: float = 0.95
    tuning_scoring_metric: str = 'f1_macro'

    TF_AVAILABLE: bool = field(default=TF_AVAILABLE, init=False, repr=False)
    tf: Any = field(default=tf if TF_AVAILABLE else None, init=False, repr=False)

    xgboost_params: XGBoostParams = field(default_factory=XGBoostParams)
    random_forest_params: RandomForestParams = field(default_factory=RandomForestParams)
    lstm_params: LSTMParams = field(default_factory=LSTMParams)

    xgboost_tuning_params: XGBoostTuningParams = field(default_factory=XGBoostTuningParams)
    random_forest_tuning_params: RandomForestTuningParams = field(default_factory=RandomForestTuningParams)

# Default config instance
DEFAULT_MODEL_CONFIG = ModelConfig()
