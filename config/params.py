# config/params.py

"""
Central configuration aggregator for the trading bot.

- Imports config dataclasses from schema files in 'config/'.
- Provides a single AppConfig object for project-wide access.
- Sensitive info (API keys, tokens) is loaded via environment variables in schema files.
- All config validation is handled in config/validator.py.
"""

import os
import logging
from typing import Any, Dict
from dotenv import load_dotenv, find_dotenv
from dataclasses import dataclass, field, is_dataclass

# Load environment variables first
load_dotenv(find_dotenv())

logger = logging.getLogger(__name__)

# Global constant for float comparisons
FLOAT_EPSILON = 1e-9

# Import config schemas (short names, e.g., general.py, exchange.py, trading.py)
from config.general import GeneralConfig, DEFAULT_GENERAL_CONFIG
from config.exchange import ExchangeConfig, DEFAULT_EXCHANGE_CONFIG
from config.feature import FeatureConfig, DEFAULT_FEATURE_CONFIG
from config.label import LabelConfig, DEFAULT_LABEL_CONFIG
from config.model import ModelConfig, DEFAULT_MODEL_CONFIG
from config.notifier import NotifierConfig, DEFAULT_NOTIFIER_CONFIG
from config.trading import TradingConfig, DEFAULT_TRADING_CONFIG  # Merged strategy + backtest

@dataclass(frozen=True)
class AppConfig:
    """
    Aggregates all configs into a single, immutable object.
    """
    general: GeneralConfig = field(default_factory=GeneralConfig)
    exchange: ExchangeConfig = field(default_factory=ExchangeConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)
    labeling: LabelConfig = field(default_factory=LabelConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    notifier: NotifierConfig = field(default_factory=NotifierConfig)
    trading: TradingConfig = field(default_factory=TradingConfig)  # Merged config

    def __post_init__(self):
        # If any config is passed as dict, instantiate the dataclass
        for attr, cls in [
            ('general', GeneralConfig),
            ('exchange', ExchangeConfig),
            ('features', FeatureConfig),
            ('labeling', LabelConfig),
            ('model', ModelConfig),
            ('notifier', NotifierConfig),
            ('trading', TradingConfig),
        ]:
            val = getattr(self, attr)
            if isinstance(val, dict):
                object.__setattr__(self, attr, cls(**val))
        # Validation is handled in config/validator.py
        
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'AppConfig':
        """
        Creates an AppConfig instance from a dictionary,
        recursively instantiating nested dataclasses from their
        corresponding dictionary keys.
        """
        kwargs = {}
        for field_name, field_type in cls.__annotations__.items():
            # Get the actual class if it's a field with a default_factory
            actual_type = field_type.__args__[0] if getattr(field_type, '__origin__', None) is field else field_type
            
            # Check if the field is a nested dataclass
            if is_dataclass(actual_type):
                # Use a specific from_dict method if it exists, otherwise instantiate
                if 'from_dict' in dir(actual_type) and isinstance(data.get(field_name), dict):
                    kwargs[field_name] = actual_type.from_dict(data.get(field_name, {}))
                else:
                    kwargs[field_name] = actual_type(**data.get(field_name, {}))
            else:
                # For non-dataclass fields, just use the value from the dictionary
                if field_name in data:
                    kwargs[field_name] = data[field_name]
        
        return cls(**kwargs)

# Global config instance
app_config = AppConfig(
    general=DEFAULT_GENERAL_CONFIG,
    exchange=DEFAULT_EXCHANGE_CONFIG,
    features=DEFAULT_FEATURE_CONFIG,
    labeling=DEFAULT_LABEL_CONFIG,
    model=DEFAULT_MODEL_CONFIG,
    notifier=DEFAULT_NOTIFIER_CONFIG,
    trading=DEFAULT_TRADING_CONFIG
)

# Usage example:
# app_config.general.random_seed
# app_config.exchange.api_key
# app_config.features.sma_periods
# app_config.trading.risk.initial_capital
# app_config.model.model_type
