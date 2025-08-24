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
    data_granularity_minutes: int = 5 # Default data granularity in minutes (e.g., 5 for 5m candles, 60 for 1h candles)
    historical_data_lookback: int = 1000 # number of historical data points required for trading
    bot_id: str = "futures_bot_v1" # Unique identifier for this bot instance to tag orders.


    # Parameters for dynamic trade loop interval in live trading
    min_trade_loop_interval_seconds: float = 10.0 # Minimum time the bot will sleep between cycles, regardless of interval
    polling_frequency_factor: float = 0.25 # How often to poll within a candle's interval (e.g., 0.25 means poll every 1/4 of the candle's duration)


# Default configuration instance
DEFAULT_GENERAL_CONFIG = GeneralConfig()

