# config/feature_config_schema.py

from dataclasses import dataclass, field
from typing import List, Literal, Optional, Dict, Any

@dataclass
class TemporalValidationConfig:
    """Configuration for temporal safety validation."""
    enabled: bool = True
    warning_correlation_threshold: float = 0.3
    error_correlation_threshold: float = 0.5

    def __post_init__(self):
        if not isinstance(self.enabled, bool):
            raise TypeError("Temporal validation 'enabled' must be a boolean.")
        if not isinstance(self.warning_correlation_threshold, float) or not (0 <= self.warning_correlation_threshold <= 1):
            raise ValueError("warning_correlation_threshold must be a float between 0 and 1.")
        if not isinstance(self.error_correlation_threshold, float) or not (0 <= self.error_correlation_threshold <= 1):
            raise ValueError("error_correlation_threshold must be a float between 0 and 1.")
        if self.warning_correlation_threshold > self.error_correlation_threshold:
            raise ValueError("warning_correlation_threshold cannot be greater than error_correlation_threshold.")


@dataclass
class FeatureConfig:
    """
    Defines configuration parameters for the FeaturesEngineer class,
    including technical indicator periods and other feature settings.
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
    pivot_point_method: Literal['standard'] = 'standard' # Only 'standard' implemented for now
    support_resistance_periods: List[int] = field(default_factory=lambda: [30, 50, 100])
    candlestick_patterns: List[str] = field(default_factory=lambda: [
        'hammer', 'engulfing', 'doji', 'evening_star', 'morning_star',
        'harami', 'shooting_star', 'dark_cloud_cover', 'piercing_pattern'
    ])
    fvg_lookback_bars: int = 3
    z_score_periods: List[int] = field(default_factory=lambda: [20, 30, 40])
    adr_periods: List[int] = field(default_factory=lambda: [14, 28])

    # Derived Features
    trend_strength_periods: List[int] = field(default_factory=lambda: [20, 50])
    
    # Swing Pivots & Breakout Parameters
    swing_pivot_left_bars: int = 15
    swing_pivot_right_bars: int = 15
    volume_oscillator_short_ema: int = 5
    volume_oscillator_long_ema: int = 10
    volume_threshold: float = 20.0 # Threshold for volume oscillator for breakout confirmation

    # New: Column name for the volatility regime feature created by FeatureEngineer
    volatility_regime_col_name: str = "volatility_regime" # Added this line

    # Temporal Safety Validation (for detecting lookahead bias during feature engineering)
    temporal_validation: TemporalValidationConfig = field(default_factory=TemporalValidationConfig)

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

    # NaN Handling
    remove_nan_rows: bool = True # Whether to remove rows with NaNs at the end of feature engineering

    # Sequence Length (MUST match model and strategy config if using sequence models like LSTM)
    sequence_length_bars: int = 5


    def __post_init__(self):
        # Basic validation for list types
        for attr in ['sma_periods', 'ema_periods', 'rsi_periods', 'bollinger_periods', 'atr_periods',
                     'stochastic_periods', 'ao_periods', 'cci_periods', 'mfi_periods', 'volume_periods',
                     'support_resistance_periods', 'z_score_periods', 'adr_periods', 'trend_strength_periods']:
            if not isinstance(getattr(self, attr), list) or not all(isinstance(x, int) and x > 0 for x in getattr(self, attr)):
                raise ValueError(f"'{attr}' must be a list of positive integers.")

        if self.pivot_point_calculation_period not in ['daily', 'weekly', 'monthly']:
            raise ValueError("pivot_point_calculation_period must be 'daily', 'weekly', or 'monthly'.")
        if self.pivot_point_method not in ['standard']: # Extend as more methods are implemented
            raise ValueError("pivot_point_method must be 'standard'.")
        if not isinstance(self.candlestick_patterns, list) or not all(isinstance(x, str) for x in self.candlestick_patterns):
            raise ValueError("'candlestick_patterns' must be a list of strings.")
        if not isinstance(self.fvg_lookback_bars, int) or self.fvg_lookback_bars <= 0:
            raise ValueError("'fvg_lookback_bars' must be a positive integer.")
        
        # Validate swing pivot parameters
        if not isinstance(self.swing_pivot_left_bars, int) or self.swing_pivot_left_bars <= 0:
            raise ValueError("swing_pivot_left_bars must be a positive integer.")
        if not isinstance(self.swing_pivot_right_bars, int) or self.swing_pivot_right_bars <= 0:
            raise ValueError("swing_pivot_right_bars must be a positive integer.")
        if not isinstance(self.volume_oscillator_short_ema, int) or self.volume_oscillator_short_ema <= 0:
            raise ValueError("volume_oscillator_short_ema must be a positive integer.")
        if not isinstance(self.volume_oscillator_long_ema, int) or self.volume_oscillator_long_ema <= 0:
            raise ValueError("volume_oscillator_long_ema must be a positive integer.")
        if self.volume_oscillator_short_ema >= self.volume_oscillator_long_ema:
            raise ValueError("volume_oscillator_short_ema must be less than volume_oscillator_long_ema.")
        if not isinstance(self.volume_threshold, (int, float)) or self.volume_threshold < 0:
            raise ValueError("volume_threshold must be a non-negative number.")

        # Temporal Validation config: Ensure nested dataclass is instantiated
        if isinstance(self.temporal_validation, dict):
            self.temporal_validation = TemporalValidationConfig(**self.temporal_validation)
        elif not isinstance(self.temporal_validation, TemporalValidationConfig):
            raise TypeError("temporal_validation must be a dictionary or TemporalValidationConfig instance.")

        if not isinstance(self.sequence_length_bars, int) or self.sequence_length_bars <= 0:
            raise ValueError("'sequence_length_bars' must be a positive integer.")
        
        if not isinstance(self.remove_nan_rows, bool):
            raise TypeError("'remove_nan_rows' must be a boolean.")

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


# Default configuration instance
DEFAULT_FEATURE_CONFIG = FeatureConfig()
