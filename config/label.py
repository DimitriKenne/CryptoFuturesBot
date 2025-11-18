# config/label_config_schema.py

from dataclasses import dataclass, field
from typing import Dict, Any, List, Literal, Type

# --- Individual labeling strategy configs ---
@dataclass
class LabelingStrategy1Config:
    """
    Triple Barrier labeling with fixed % for TP/SL. All '_pct' fields are percentages (0-100).
    Supports both long and short directions.
    """
    take_profit_pct: float = 4.0      # TP barrier as percent (e.g. 6 for +6%)
    stop_loss_pct: float = 1.5        # SL barrier as percent (e.g. 3 for -3%)
    lookahead_bars: int = 150          # Number of bars to look ahead for barrier hit


@dataclass
class LabelingStrategy2Config:
    """Net Forward Return Quantile labeling. All '_pct' fields are percentages (0-100)."""
    quantile_threshold_long_pct: float = 30.0
    quantile_threshold_short_pct: float = 30.0
    future_return_window: int = 150
    # return_type: Literal['log_returns', 'simple_returns'] = 'simple_returns'

@dataclass
class LabelingStrategy3Config:
    """
    HTF Volatility Filtered Labeling (Fixed R/R, Daily Expiration). (NEW STRATEGY)
    1. Filters on HTF Volatility (Range).
    2. Labels trades hitting fixed TP/SL percentage before HTF bar close.
    """
    volatility_quantile_pct: float = 30.0 # Only label LTF bars in HTF bars with volatility >= this percentile (0-100)
    take_profit_pct: float = 3.0        # Fixed Take Profit barrier as percent (e.g. 2.0 for +2%)
    stop_loss_pct: float = 1.0          # Fixed Stop Loss barrier as percent (e.g. 1.0 for -1%)
    htf_timeframe: str = "1d"           # The High Time Frame to calculate volatility and expiry against

@dataclass
class LabelingStrategy4Config:
    """Swing Pivot / Clustering labeling. All '_pct' fields are percentages (0-100)."""
    n_clusters: int = 3
    features_for_clustering: List[str] = field(default_factory=list)
    pca_n_components_pct: float = 95.0        # e.g. 95 for 0.95
    cluster_to_label_mapping: Dict[int, int] = field(default_factory=lambda: {1: 1, 0: 0, 2: -1})
    future_return_window: int = 150

@dataclass
class LabelingStrategy5Config:
    """
    HTF-LTF Context Quantile labeling for regime alignment.
    All '_pct' fields are percentages (0-100).
    """
    htf_timeframe: str = "1d"           # Higher timeframe, e.g. "1d" for daily
    bullish_quantile_pct: float = 50.0  # Quantile for bullish regime (e.g. 75 for 75th percentile)
    bearish_quantile_pct: float = 50.0  # Quantile for bearish regime (e.g. 25 for 25th percentile)


# --- Dynamically populate mapping of labeling strategy keys to their config dataclass types ---
_LABELING_STRATEGY_CONFIG_CLASS_MAP: Dict[str, Type[Any]] = {
    'labeling_strategy_1': LabelingStrategy1Config,
    'labeling_strategy_2': LabelingStrategy2Config,
    'labeling_strategy_3': LabelingStrategy3Config,
    'labeling_strategy_4': LabelingStrategy4Config,
    'labeling_strategy_5': LabelingStrategy5Config
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
        'labeling_strategy_4',
        'labeling_strategy_5'
    ] = 'labeling_strategy_1'
    min_holding_period: int = 5

    labeling_strategy_1: LabelingStrategy1Config = field(default_factory=LabelingStrategy1Config)
    labeling_strategy_2: LabelingStrategy2Config = field(default_factory=LabelingStrategy2Config)
    labeling_strategy_3: LabelingStrategy3Config = field(default_factory=LabelingStrategy3Config)
    labeling_strategy_4: LabelingStrategy4Config = field(default_factory=LabelingStrategy4Config)
    labeling_strategy_5: LabelingStrategy5Config = field(default_factory=LabelingStrategy5Config)

    trading_fee_pct: float = 0.05             # e.g. 0.05 for 0.05%
    slippage_tolerance_pct: float = 0.01      # e.g. 0.01 for 0.01%
    analysis_future_horizons: List[int] = field(default_factory=lambda: [10, 30, 60, 100, 150, 300])

# Default configuration instance
DEFAULT_LABEL_CONFIG = LabelConfig()
