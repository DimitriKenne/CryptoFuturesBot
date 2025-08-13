# config/feature_config_schema.py

from dataclasses import dataclass, field
from typing import List, Literal, Any, Dict
# Removed sys and Path imports as they are no longer necessary for this file
# This file defines a schema and default config, it doesn't need to manipulate sys.path

@dataclass
class TemporalValidationConfig:
    """
    Configuration for temporal safety validation within feature engineering.
    """
    enabled: bool = True
    warning_correlation_threshold: float = 0.3
    error_correlation_threshold: float = 0.5

@dataclass
class FeatureConfig:
    """
    Defines the configuration parameters for the FeatureEngineer class.

    This dataclass provides a structured and validated way to manage
    all periods and settings for various technical indicators and custom features.
    """
    # General Feature Parameters
    sma_periods: List[int] = field(default_factory=lambda: [10, 20, 50, 100, 200])
    ema_periods: List[int] = field(default_factory=lambda: [10, 20, 50, 100, 200])
    rsi_periods: List[int] = field(default_factory=lambda: [14, 21, 30])
    bollinger_periods: List[int] = field(default_factory=lambda: [20, 30])
    atr_periods: List[int] = field(default_factory=lambda: [14])
    stochastic_periods: List[int] = field(default_factory=lambda: [14])
    ao_periods: List[int] = field(default_factory=lambda: [5, 34]) # Awesome Oscillator periods [short, long]
    cci_periods: List[int] = field(default_factory=lambda: [20])
    mfi_periods: List[int] = field(default_factory=lambda: [14])
    volume_periods: List[int] = field(default_factory=lambda: [20, 50]) # Used for CMF and MFI windows
    support_resistance_periods: List[int] = field(default_factory=lambda: [50, 100, 200]) # Periods for S/R
    candlestick_patterns: List[str] = field(default_factory=list) # List of TA-Lib pattern names
    fvg_lookback_bars: int = 2 # Fair Value Gap lookback
    z_score_periods: List[int] = field(default_factory=lambda: [30, 60]) # Z-score periods for close price
    adr_periods: List[int] = field(default_factory=lambda: [14]) # Average Daily Range periods
    trend_strength_periods: List[int] = field(default_factory=lambda: [20, 50]) # Trend strength periods [short_sma, long_sma]

    # Pivot Point Configuration (Standard Pivots based on daily/weekly/monthly OHLC)
    pivot_point_calculation_period: Literal['daily', 'weekly', 'monthly'] = 'daily'
    pivot_point_method: Literal['standard', 'fibonacci', 'woodie', 'camarilla'] = 'standard' # Only 'standard' currently implemented

    # Pine Script style Swing Pivot Configuration
    swing_pivot_left_bars: int = 15
    swing_pivot_right_bars: int = 15

    # Volume Breakout and Oscillator Configuration
    volume_oscillator_short_ema: int = 5
    volume_oscillator_long_ema: int = 10
    volume_threshold: float = 20.0 # Threshold for volume breakout detection (e.g., percentage above normal)

    # Lagged Features (simple shifts)
    lagged_features: Dict[str, List[int]] = field(default_factory=lambda: {
        'close': [1, 2, 3],
        'volume': [1, 2],
        'high': [1],
        'low': [1]
    })

    # Differenced Features
    differenced_features: Dict[str, List[int]] = field(default_factory=lambda: {
        'close': [1],
        'volume': [1],
    })

    # Temporal Safety Validation Configuration
    temporal_validation: TemporalValidationConfig = field(default_factory=TemporalValidationConfig)

    # NaN Handling
    remove_nan_rows: bool = True # Whether to remove rows with NaNs at the end of feature engineering

    # Sequence Length (MUST match model and strategy config if using sequence models like LSTM)
    sequence_length_bars: int = 5


    def __post_init__(self):
        """
        Perform additional validation after initialization.
        This allows for cross-field validation not easily done with simple type hints.
        """
        # Validate AO periods length and order
        if len(self.ao_periods) != 2:
            raise ValueError(f"AO periods must be a list of exactly two integers [short, long], but got {self.ao_periods}")
        if not (0 < self.ao_periods[0] < self.ao_periods[1]):
            raise ValueError(f"AO short period ({self.ao_periods[0]}) must be less than AO long period ({self.ao_periods[1]}) and both must be positive.")

        # Validate Trend Strength periods length and order
        if len(self.trend_strength_periods) != 2:
            raise ValueError(f"Trend strength periods must be a list of exactly two integers [short, long], but got {self.trend_strength_periods}")
        if not (0 < self.trend_strength_periods[0] < self.trend_strength_periods[1]):
            raise ValueError(f"Trend strength short period ({self.trend_strength_periods[0]}) must be less than trend strength long period ({self.trend_strength_periods[1]}) and both must be positive.")
        
        # Validate Volume Oscillator EMA periods
        if not (0 < self.volume_oscillator_short_ema < self.volume_oscillator_long_ema):
            raise ValueError(f"Volume oscillator short EMA ({self.volume_oscillator_short_ema}) must be less than long EMA ({self.volume_oscillator_long_ema}) and both must be positive.")

        # Validate that all period lists contain positive integers
        for attr in ['sma_periods', 'ema_periods', 'rsi_periods', 'bollinger_periods', 'atr_periods',
                     'stochastic_periods', 'cci_periods', 'mfi_periods', 'volume_periods',
                     'support_resistance_periods', 'z_score_periods', 'adr_periods']:
            if not all(isinstance(p, int) and p > 0 for p in getattr(self, attr)):
                raise ValueError(f"All periods in '{attr}' must be positive integers.")
        
        # Validate single positive integer values
        for attr in ['fvg_lookback_bars', 'swing_pivot_left_bars', 'swing_pivot_right_bars', 'sequence_length_bars']:
            val = getattr(self, attr)
            if not isinstance(val, int) or val <= 0:
                raise ValueError(f"'{attr}' must be a positive integer.")
        
        # Validate volume_threshold type
        if not isinstance(self.volume_threshold, (int, float)):
            raise ValueError("'volume_threshold' must be a number.")

        # Validate candlestick_patterns elements (ensure they are strings, optional non-empty check)
        if not all(isinstance(p, str) for p in self.candlestick_patterns):
            raise ValueError("'candlestick_patterns' must be a list of strings.")

        # Validate lagged_features configuration
        for col, lags in self.lagged_features.items():
            if not isinstance(col, str) or not col:
                raise ValueError(f"Lagged feature column name must be a non-empty string, got '{col}'.")
            if not (isinstance(lags, list) and all(isinstance(l, int) and l > 0 for l in lags)):
                raise ValueError(f"Lags for '{col}' must be a list of positive integers, got {lags}.")

        # Validate differenced_features configuration
        for col, orders in self.differenced_features.items():
            if not isinstance(col, str) or not col:
                raise ValueError(f"Differenced feature column name must be a non-empty string, got '{col}'.")
            if not (isinstance(orders, list) and all(isinstance(o, int) and o > 0 for o in orders)):
                raise ValueError(f"Differencing orders for '{col}' must be a list of positive integers, got {orders}.")

# --- Default Feature Configuration Values ---
# This dictionary will be used to initialize the FeatureConfig dataclass
# when no specific configuration is provided.
DEFAULT_FEATURE_CONFIG = {
    # Technical Indicator Periods (Use lists for multiple periods where beneficial)
    'sma_periods': [10, 20, 50, 100],   # Simple Moving Averages (Standard range for context)
    'ema_periods': [10, 14, 20, 50, 100, 200], # Exponential Moving Averages (Added 200 for longer-term context)
    'rsi_periods': [7, 14, 28, 50, 150],        # Relative Strength Index (Shorter to medium-long for 5m)
    'bollinger_periods': [20, 30, 40, 150],     # Bollinger Bands (Range around common values)
    'atr_periods': [5, 14, 20, 50, 150],           # Average True Range (Standard and variants for volatility)
    'stochastic_periods': [14, 28],      # Stochastic Oscillator %K (Standard and double)
    'ao_periods': [5, 34],                  # Awesome Oscillator (Standard periods)
    'cci_periods': [14, 20, 40],           # Commodity Channel Index (Faster, standard, slower)
    'mfi_periods': [14, 28],             # Money Flow Index (Standard and double)
    'volume_periods': [10, 20, 30],        # Period for Volume-based indicators (e.g., CMF, OBV calculation window if used)

    # Other Feature Settings
    'pivot_point_calculation_period': 'daily', # 'daily', 'weekly', 'monthly'
    'pivot_point_method': 'standard', # 'standard', 'fibonacci', 'woodie', 'camarilla' (only 'standard' implemented for now)
    'support_resistance_periods': [30, 50, 100], # list[int]: Lookback for simple S/R levels (Intraday relevant ranges).
    'candlestick_patterns': [               # list[str]: Patterns to detect (uses talib).
        'hammer', 'engulfing', 'doji', 'evening_star', 'morning_star',
        'harami', 'shooting_star', 'dark_cloud_cover', 'piercing_pattern'
    ],
    'fvg_lookback_bars': 3,                 # int: Lookback for Fair Value Gap detection (standard 3-candle is i vs i-2).
    'z_score_periods': [20, 30, 40],        # list[int]: Periods for Z-score calculation.
    'adr_periods': [14, 28],               # list[int]: Average 5m Range period (different lookbacks for recent volatility).

    # Derived Features
    'trend_strength_periods': [20, 50],     # list[int]: Short/long periods for trend strength (based on SMAs).
    
    # Swing Pivots & Breakout Parameters
    'swing_pivot_left_bars': 15,
    'swing_pivot_right_bars': 15,
    'volume_oscillator_short_ema': 5,
    'volume_oscillator_long_ema': 10,
    'volume_threshold': 20.0, # Threshold for volume oscillator for breakout confirmation (float)

    # Temporal Safety Validation (for detecting lookahead bias during feature engineering)
    # IMPORTANT: Initialize this directly as a TemporalValidationConfig object
    'temporal_validation': TemporalValidationConfig(
        enabled=True,
        warning_correlation_threshold=0.3,
        error_correlation_threshold=0.5
    ),

    # NEW: Lagged Features (simple shifts)
    'lagged_features': {
        'close': [1, 2, 3],
        'volume': [1, 2],
        'high': [1],
        'low': [1]
    },

    # NEW: Differenced Features
    'differenced_features': {
        'close': [1],
        'volume': [1],
    },

    # Sequence Length (MUST match model and strategy config if using sequence models like LSTM)
    'sequence_length_bars': 5,
}
