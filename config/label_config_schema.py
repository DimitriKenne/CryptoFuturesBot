# config/label_config_schema.py

from dataclasses import dataclass, field
from typing import Dict, Any, List, Literal, Type
import importlib

# Define individual strategy configs first
@dataclass
class Strategy1Config:
    """Configuration for Strategy1 (formerly Triple Barrier)."""
    profit_multiplier: float = 2.0
    stop_loss_multiplier: float = 1.0
    future_return_window: int = 200 # Look-forward window for barriers
    vol_adj_lookback: int = 14 # ATR period for volatility adjustment
    num_price_bars: int = 100 # Default bars for price sampling if needed for ATR
    
@dataclass
class Strategy2Config:
    """Configuration for Strategy2 (formerly Net Forward Return Quantile)."""
    quantile_threshold_long: float = 0.75
    quantile_threshold_short: float = 0.25
    future_return_window: int = 150
    return_type: Literal['log_returns', 'simple_returns'] = 'log_returns'
    
@dataclass
class Strategy3Config:
    """Configuration for Strategy3 (formerly Future Range Dominance)."""
    future_return_window: int = 150 # Renamed from f_window_range for consistency
    long_ratio_quantile_pct: float = 75.0 # New: for the threshold of the long dominance ratio
    short_ratio_quantile_pct: float = 75.0 # New: for the threshold of the short dominance ratio
    min_profit_threshold: float = 0.005 # Minimum required net profit (e.g., 0.5% after fees)
    
@dataclass
class Strategy4Config:
    """Configuration for Strategy4 (formerly Swing Pivot / Clustering)."""
    n_clusters: int = 3
    features_for_clustering: List[str] = field(default_factory=list) # This will be populated dynamically
    pca_n_components: float = 0.95 # or int e.g. 10
    cluster_to_label_mapping: Dict[int, int] = field(default_factory=lambda: {
        1: 1,  # Example: Cluster 1 -> Long
        0: 0,  # Example: Cluster 0 -> Neutral
        2: -1  # Example: Cluster 2 -> Short
    })
    future_return_window: int = 150

# --- Dynamically populate mapping of strategy keys to their config dataclass types ---
_STRATEGY_CONFIG_CLASS_MAP: Dict[str, Type[Any]] = {}

for i in range(1, 5): # Adjust range if you have more strategies
    strategy_config_name = f'Strategy{i}Config'
    strategy_key = f'strategy_{i}'
    try:
        config_class = globals()[strategy_config_name]
        _STRATEGY_CONFIG_CLASS_MAP[strategy_key] = config_class
    except KeyError:
        raise RuntimeError(f"Strategy config class '{strategy_config_name}' not defined in label_config_schema.py.")


@dataclass
class LabelConfig:
    """
    Defines the configuration parameters for the LabelGenerator class.
    """
    label_type: Literal['strategy_1', 'strategy_2', 'strategy_3', 'strategy_4'] = 'strategy_1' # Set default here
    min_holding_period: int = 1 # Minimum number of bars a signal should persist

    # Nested configurations for each strategy - these now rely on __post_init__ for dynamic typing
    strategy_1: Strategy1Config = field(default_factory=Strategy1Config)
    strategy_2: Strategy2Config = field(default_factory=Strategy2Config)
    strategy_3: Strategy3Config = field(default_factory=Strategy3Config)
    strategy_4: Strategy4Config = field(default_factory=Strategy4Config)

    # General parameters used by labeling or dependent modules (e.g., trading fees)
    trading_fee_rate: float = 0.0005
    slippage_tolerance_pct: float = 0.0005

    def __post_init__(self):
        """
        Perform validation and dynamically instantiate nested strategy configs from dicts.
        """
        if self.min_holding_period < 1:
            raise ValueError("min_holding_period must be at least 1.")
        
        for strategy_key, strategy_config_class in _STRATEGY_CONFIG_CLASS_MAP.items():
            current_value = getattr(self, strategy_key)
            if isinstance(current_value, dict):
                setattr(self, strategy_key, strategy_config_class(**current_value))
            elif not isinstance(current_value, strategy_config_class):
                # This check ensures that if the field was assigned something other than a dict
                # (e.g., from a direct instantiation of LabelConfig where a nested config
                # was already an object), it's the correct type.
                pass # Already handled by dataclasses default_factory or valid dict conversion
        
        if not isinstance(self.trading_fee_rate, (int, float)) or self.trading_fee_rate < 0:
            raise ValueError("trading_fee_rate must be a non-negative number.")
        if not isinstance(self.slippage_tolerance_pct, (int, float)) or self.slippage_tolerance_pct < 0:
            raise ValueError("slippage_tolerance_pct must be a non-negative number.")


# Default configuration instance
DEFAULT_LABEL_CONFIG = LabelConfig()
