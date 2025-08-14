# config/params.py

"""
Central configuration file for the trading bot project.

This file now primarily imports and aggregates configurations
from more granular, schema-defined files located in the 'config/' directory.
It provides a single, comprehensive object (AppConfig) for accessing
all project-wide parameters, ensuring consistency across modules.

Sensitive information (API keys/secrets, tokens) are loaded from
environment variables by their respective schema files, reducing direct exposure here.

NOTE: Individual configuration sections have been moved to dedicated schema files:
- config/general_config_schema.py
- config/exchange_config_schema.py
- config/feature_config_schema.py
- config/label_config_schema.py
- config/model_config_schema.py
- config/strategy_config_schema.py
- config/notifier_config_schema.py
- config/backtest_config_schema.py # New: Backtesting specific configurations
"""

import os
import logging
from dotenv import load_dotenv, find_dotenv
from dataclasses import dataclass, field

# --- Load Environment Variables FIRST ---
# This ensures environment variables are available when default_factory lambdas
# in other config schemas attempt to access them.
load_dotenv(find_dotenv())

# Set up a module-level logger for params.py itself
logger = logging.getLogger(__name__)


# --- Global Constants ---
# Define FLOAT_EPSILON for robust floating-point comparisons
# Used to avoid division by zero or issues with very small numbers
FLOAT_EPSILON = 1e-9


# --- Import Configuration Schemas ---
# Import the dataclass definitions and their default instances
from config.general_config_schema import GeneralConfig, DEFAULT_GENERAL_CONFIG
from config.exchange_config_schema import ExchangeConfig, DEFAULT_EXCHANGE_CONFIG
from config.feature_config_schema import FeatureConfig, DEFAULT_FEATURE_CONFIG
from config.label_config_schema import LabelConfig, DEFAULT_LABEL_CONFIG
from config.model_config_schema import ModelConfig, DEFAULT_MODEL_CONFIG
from config.strategy_config_schema import StrategyConfig, DEFAULT_STRATEGY_CONFIG
from config.notifier_config_schema import NotifierConfig, DEFAULT_NOTIFIER_CONFIG
from config.backtest_config_schema import BacktestConfig, DEFAULT_BACKTEST_CONFIG


@dataclass(frozen=True) # Making AppConfig immutable for consistency
class AppConfig:
    """
    Aggregates all application configurations into a single, immutable object.
    Ensures consistency and centralizes access to parameters across the project.
    """
    general: GeneralConfig = field(default_factory=GeneralConfig)
    exchange: ExchangeConfig = field(default_factory=ExchangeConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)
    labeling: LabelConfig = field(default_factory=LabelConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    strategy: StrategyConfig = field(default_factory=StrategyConfig)
    notifier: NotifierConfig = field(default_factory=NotifierConfig)
    backtest: BacktestConfig = field(default_factory=BacktestConfig) # NEW: Add backtest config

    def __post_init__(self):
        """
        Performs post-initialization validation and cross-schema consistency checks.
        Ensures nested dataclasses are correctly instantiated if passed as dicts.
        """
        # Ensure nested configs are instantiated from their dicts if passed as such
        # Use object.__setattr__ for frozen dataclasses
        if isinstance(self.general, dict): object.__setattr__(self, 'general', GeneralConfig(**self.general))
        if isinstance(self.exchange, dict): object.__setattr__(self, 'exchange', ExchangeConfig(**self.exchange))
        if isinstance(self.features, dict): object.__setattr__(self, 'features', FeatureConfig(**self.features))
        if isinstance(self.labeling, dict): object.__setattr__(self, 'labeling', LabelConfig(**self.labeling))
        if isinstance(self.model, dict): object.__setattr__(self, 'model', ModelConfig(**self.model))
        if isinstance(self.strategy, dict): object.__setattr__(self, 'strategy', StrategyConfig(**self.strategy))
        if isinstance(self.notifier, dict): object.__setattr__(self, 'notifier', NotifierConfig(**self.notifier))
        if isinstance(self.backtest, dict): object.__setattr__(self, 'backtest', BacktestConfig(**self.backtest)) # NEW: backtest config


        # Perform cross-schema validation
        self._validate_cross_schema_consistency()
        logger.info("AppConfig initialized and validated successfully.")


    def _validate_cross_schema_consistency(self):
        """
        Performs consistency checks across different configuration schemas.
        """
        # Consistency check for LSTM: Feature sequence length must match model's expected input timesteps
        if self.model.model_type == 'lstm':
            if self.model.lstm_params.input_timesteps != self.features.sequence_length_bars:
                raise ValueError(
                    f"LSTM 'input_timesteps' ({self.model.lstm_params.input_timesteps}) in ModelConfig "
                    f"must match 'sequence_length_bars' ({self.features.sequence_length_bars}) in FeatureConfig."
                )
            # For LSTM, if strategy's lookback for prediction is greater than 1, it implies it expects a sequence of predictions.
            # This is not necessarily an error but a logical consideration.
            # if self.strategy.model_prediction_lookback_bars > 1 and self.strategy.model_prediction_lookback_bars != self.features.sequence_length_bars:
            #     logger.warning("Strategy's 'model_prediction_lookback_bars' is > 1 but does not match feature 'sequence_length_bars'. Ensure this is intended for how predictions are consumed.")

        # Removed the general check:
        # if (self.features.sequence_length_bars != self.strategy.model_prediction_lookback_bars):
        #    raise ValueError(...)
        # This check is incorrect as these two parameters serve different purposes.

        # Cross-validation for trading fee rates (if you wish to enforce consistency)
        # For now, allowing LabelConfig and StrategyConfig to have their own rates.
        # if self.labeling.trading_fee_rate != self.strategy.trading_fee_rate:
        #     logger.warning("Trading fee rates in LabelConfig and StrategyConfig differ. Ensure this is intentional.")


# --- Global Application Configuration Instance ---
# This is the primary instance of the aggregated configuration
# that other modules should import and use.
app_config = AppConfig(
    general=DEFAULT_GENERAL_CONFIG,
    exchange=DEFAULT_EXCHANGE_CONFIG,
    features=DEFAULT_FEATURE_CONFIG,
    labeling=DEFAULT_LABEL_CONFIG,
    model=DEFAULT_MODEL_CONFIG,
    strategy=DEFAULT_STRATEGY_CONFIG,
    notifier=DEFAULT_NOTIFIER_CONFIG,
    backtest=DEFAULT_BACKTEST_CONFIG # Include the new backtest config
)

# You can now access configuration parameters like:
# app_config.general.log_level
# app_config.exchange.api_key
# app_config.features.sma_periods
# app_config.strategy.initial_capital
# app_config.model.model_type
