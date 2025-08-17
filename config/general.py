# config/general_config_schema.py

from dataclasses import dataclass

@dataclass
class GeneralConfig:
    """
    Defines general configuration parameters for the project.
    """
    random_seed: int = 42
    n_processors: int = -1 # Number of CPU cores to use. -1 for all available.
    hyperparameter_tuning_n_iter: int = 50 # Number of parameter settings that are sampled
    hyperparameter_tuning_cv_folds: int = 5 # Number of cross-validation splits


# Default configuration instance
DEFAULT_GENERAL_CONFIG = GeneralConfig()
