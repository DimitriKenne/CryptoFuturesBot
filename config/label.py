# config/label_config_schema.py

from dataclasses import dataclass, field
from typing import Dict, Any, List, Literal, Type

# --- Individual labeling strategy configs ---
@dataclass
class LabelingStrategy1Config:
    """Triple Barrier labeling. All '_pct' fields are percentages (0-100)."""
    profit_multiplier_pct: float = 200.0      # e.g. 200 for 2x ATR
    stop_loss_multiplier_pct: float = 100.0   # e.g. 100 for 1x ATR
    future_return_window: int = 200
    vol_adj_lookback: int = 14
    num_price_bars: int = 100

@dataclass
class LabelingStrategy2Config:
    """Net Forward Return Quantile labeling. All '_pct' fields are percentages (0-100)."""
    quantile_threshold_long_pct: float = 25.0
    quantile_threshold_short_pct: float = 25.0
    future_return_window: int = 300
    # return_type: Literal['log_returns', 'simple_returns'] = 'simple_returns'

@dataclass
class LabelingStrategy3Config:
    """Future Range Dominance labeling. All '_pct' fields are percentages (0-100)."""
    future_return_window: int = 150
    long_ratio_quantile_pct: float = 50.0
    short_ratio_quantile_pct: float = 50.0
    min_profit_threshold_pct: float = 0.5     # e.g. 0.5 for 0.5%

@dataclass
class LabelingStrategy4Config:
    """Swing Pivot / Clustering labeling. All '_pct' fields are percentages (0-100)."""
    n_clusters: int = 3
    features_for_clustering: List[str] = field(default_factory=list)
    pca_n_components_pct: float = 95.0        # e.g. 95 for 0.95
    cluster_to_label_mapping: Dict[int, int] = field(default_factory=lambda: {1: 1, 0: 0, 2: -1})
    future_return_window: int = 150

# --- Dynamically populate mapping of labeling strategy keys to their config dataclass types ---
_LABELING_STRATEGY_CONFIG_CLASS_MAP: Dict[str, Type[Any]] = {
    'labeling_strategy_1': LabelingStrategy1Config,
    'labeling_strategy_2': LabelingStrategy2Config,
    'labeling_strategy_3': LabelingStrategy3Config,
    'labeling_strategy_4': LabelingStrategy4Config,
}

@dataclass
class LabelConfig:
    """
    Labeling configuration. All '_pct' fields are percentages (0-100).
    """
    labeling_strategy_type: Literal[
        'labeling_strategy_1',
        'labeling_strategy_2',
        'labeling_strategy_3',
        'labeling_strategy_4'
    ] = 'labeling_strategy_1'
    min_holding_period: int = 1

    labeling_strategy_1: LabelingStrategy1Config = field(default_factory=LabelingStrategy1Config)
    labeling_strategy_2: LabelingStrategy2Config = field(default_factory=LabelingStrategy2Config)
    labeling_strategy_3: LabelingStrategy3Config = field(default_factory=LabelingStrategy3Config)
    labeling_strategy_4: LabelingStrategy4Config = field(default_factory=LabelingStrategy4Config)

    trading_fee_pct: float = 0.05             # e.g. 0.05 for 0.05%
    slippage_tolerance_pct: float = 0.01      # e.g. 0.01 for 0.01%
    analysis_future_horizons: List[int] = field(default_factory=lambda: [10, 30, 60, 100, 150])

# Default configuration instance
DEFAULT_LABEL_CONFIG = LabelConfig()
