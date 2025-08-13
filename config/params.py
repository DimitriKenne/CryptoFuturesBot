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
# Used to avoid division by zero or issues with very small numbers.
FLOAT_EPSILON = 1e-9


# --- Import all configuration schemas and their default instances ---
# These imports make the dataclass definitions and their default instances
# available for AppConfig.
from config.general_config_schema import GeneralConfig, DEFAULT_GENERAL_CONFIG
from config.exchange_config_schema import ExchangeConfig, DEFAULT_EXCHANGE_CONFIG
from config.feature_config_schema import FeatureConfig, DEFAULT_FEATURE_CONFIG
from config.label_config_schema import LabelConfig, DEFAULT_LABEL_CONFIG
from config.model_config_schema import ModelConfig, DEFAULT_MODEL_CONFIG
from config.strategy_config_schema import StrategyConfig, DEFAULT_STRATEGY_CONFIG
from config.notifier_config_schema import NotifierConfig, DEFAULT_NOTIFIER_CONFIG
from config.backtest_config_schema import BacktestConfig, DEFAULT_BACKTEST_CONFIG # New import


@dataclass
class AppConfig:
    """
    A comprehensive dataclass aggregating all specific configuration schemas.
    This serves as the single source of truth for all application parameters.
    """
    general: GeneralConfig = field(default_factory=GeneralConfig)
    exchange: ExchangeConfig = field(default_factory=ExchangeConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)
    labeling: LabelConfig = field(default_factory=LabelConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    strategy: StrategyConfig = field(default_factory=StrategyConfig)
    notifier: NotifierConfig = field(default_factory=NotifierConfig)
    backtest: BacktestConfig = field(default_factory=BacktestConfig) # New field for backtesting config

    def __post_init__(self):
        """
        Ensures all nested configuration objects are properly instantiated.
        This handles cases where config might be loaded from a dictionary (e.g., JSON/YAML).
        """
        if isinstance(self.general, dict):
            self.general = GeneralConfig(**self.general)
        if isinstance(self.exchange, dict):
            self.exchange = ExchangeConfig(**self.exchange)
        if isinstance(self.features, dict):
            self.features = FeatureConfig(**self.features)
        if isinstance(self.labeling, dict):
            self.labeling = LabelConfig(**self.labeling)
        if isinstance(self.model, dict):
            self.model = ModelConfig(**self.model)
        if isinstance(self.strategy, dict):
            self.strategy = StrategyConfig(**self.strategy)
        if isinstance(self.notifier, dict):
            self.notifier = NotifierConfig(**self.notifier)
        if isinstance(self.backtest, dict): # New instantiation check
            self.backtest = BacktestConfig(**self.backtest)

        self._validate_cross_schema_consistency()

    def _validate_cross_schema_consistency(self):
        """
        Performs validation checks for parameters that span multiple configuration schemas.
        """
        # Validate that sequence_length_bars in FeatureConfig matches input_timesteps in LSTMParams
        if (self.features.sequence_length_bars != self.strategy.sequence_length_bars):
            raise ValueError(
                f"Mismatch in 'sequence_length_bars': FeatureConfig has {self.features.sequence_length_bars} "
                f"but StrategyConfig has {self.strategy.sequence_length_bars}. These must be consistent."
            )

        if self.model.model_type == 'lstm':
            if self.model.lstm_params.input_timesteps != self.features.sequence_length_bars:
                raise ValueError(
                    f"LSTM 'input_timesteps' ({self.model.lstm_params.input_timesteps}) in ModelConfig "
                    f"must match 'sequence_length_bars' ({self.features.sequence_length_bars}) in FeatureConfig."
                )

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

logger.info("All configuration schemas loaded and aggregated into app_config.")
