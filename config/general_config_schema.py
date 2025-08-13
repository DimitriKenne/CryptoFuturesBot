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

    def __post_init__(self):
        if not isinstance(self.random_seed, int):
            raise TypeError("random_seed must be an integer.")
        if not isinstance(self.n_processors, int) or self.n_processors < -1:
            raise ValueError("n_processors must be an integer >= -1.")
        if not isinstance(self.hyperparameter_tuning_n_iter, int) or self.hyperparameter_tuning_n_iter <= 0:
            raise ValueError("hyperparameter_tuning_n_iter must be a positive integer.")
        if not isinstance(self.hyperparameter_tuning_cv_folds, int) or self.hyperparameter_tuning_cv_folds <= 0:
            raise ValueError("hyperparameter_tuning_cv_folds must be a positive integer.")

# Default configuration instance
DEFAULT_GENERAL_CONFIG = GeneralConfig()
