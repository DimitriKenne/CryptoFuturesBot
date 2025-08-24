# config/feature.py

from dataclasses import dataclass, field
from typing import List, Literal, Optional, Dict, Any
import logging 

# Optional: TA-Lib for specific indicators/patterns
try:
    import talib
    TALIB_AVAILABLE_RUNTIME = True
except ImportError:
    talib = None
    TALIB_AVAILABLE_RUNTIME = False

# Optional: 'ta' library for common technical indicators
try:
    import ta
    TA_LIB_AVAILABLE_RUNTIME = True
except ImportError:
    ta = None
    TA_LIB_AVAILABLE_RUNTIME = False

@dataclass
class TemporalValidationConfig:
    """Temporal safety validation settings."""
    enabled: bool = True
    warning_correlation_threshold: float = 0.3
    error_correlation_threshold: float = 0.6

@dataclass
class FeatureConfig:
    """
    Feature engineering configuration.
    """
    # Technical Indicator Periods (Use lists for multiple periods where beneficial)
    sma_periods: List[int] = field(default_factory=lambda: [10, 20, 50, 100])
    ema_periods: List[int] = field(default_factory=lambda: [10, 14, 20, 50, 100, 200])
    rsi_periods: List[int] = field(default_factory=lambda: [7, 14, 28, 50, 150])
    bollinger_periods: List[int] = field(default_factory=lambda: [20, 30, 40, 150])
    atr_periods: List[int] = field(default_factory=lambda: [5, 14, 20, 50, 150])
    stochastic_periods: List[int] = field(default_factory=lambda: [14, 28])
    ao_periods: List[int] = field(default_factory=lambda: [5, 34])
    cci_periods: List[int] = field(default_factory=lambda: [14, 20, 40])
    mfi_periods: List[int] = field(default_factory=lambda: [14, 28])
    volume_periods: List[int] = field(default_factory=lambda: [10, 20, 30])

    # Other Feature Settings
    pivot_point_calculation_period: Literal['daily', 'weekly', 'monthly'] = 'daily'
    pivot_point_method: Literal['standard'] = 'standard'
    support_resistance_periods: List[int] = field(default_factory=lambda: [30, 50, 100])
    candlestick_patterns: List[str] = field(default_factory=lambda: [
        'hammer', 'engulfing', 'doji', 'evening_star', 'morning_star',
        'harami', 'shooting_star', 'dark_cloud_cover', 'piercing_pattern'
    ])
    fvg_lookback_bars: int = 3
    z_score_periods: List[int] = field(default_factory=lambda: [20, 30, 40])
    adr_periods: List[int] = field(default_factory=lambda: [1, 2])
    trend_strength_periods: List[int] = field(default_factory=lambda: [20, 50])

    # Swing Pivots & Breakout Parameters
    swing_pivot_left_bars: int = 15
    swing_pivot_right_bars: int = 15
    volume_oscillator_short_ema: int = 5
    volume_oscillator_long_ema: int = 10
    volume_threshold: float = 20.0

    # Volatility Regime
    volatility_regime_col_name: str = "volatility_regime"

    # Temporal Safety Validation
    temporal_validation: TemporalValidationConfig = field(default_factory=TemporalValidationConfig)

    # Lagged & Differenced Features
    lagged_features: Dict[str, List[int]] = field(default_factory=lambda: {
        'close': [1, 2, 3],
        'volume': [1, 2],
        'high': [1],
        'low': [1]
    })
    differenced_features: Dict[str, List[int]] = field(default_factory=lambda: {
        'close': [1],
        'volume': [1],
    })

    # NaN Handling
    remove_nan_rows: bool = True

    # --- REMOVED: sequence_length_bars is now in ModelConfig ---
    
    # Library Availability Flags
    talib_available: bool = field(default=TALIB_AVAILABLE_RUNTIME, init=False, repr=False)
    ta_lib_available: bool = field(default=TA_LIB_AVAILABLE_RUNTIME, init=False, repr=False)

    # --- REMOVED: __post_init__ is redundant with validator.py ---

# Default configuration instance
DEFAULT_FEATURE_CONFIG = FeatureConfig()