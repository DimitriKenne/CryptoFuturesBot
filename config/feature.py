# config/feature_config_schema.py

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
    pivot_point_method: Literal['standard'] = 'standard' # Only 'standard' implemented for now
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
    remove_nan_rows: bool = False # Whether to remove rows with NaNs at the end of feature engineering

    # Sequence Length (MUST match model and strategy config if using sequence models like LSTM)
    sequence_length_bars: int = 5

    # Library Availability Flags (Runtime determined)
    talib_available: bool = field(default=TALIB_AVAILABLE_RUNTIME, init=False, repr=False)
    ta_lib_available: bool = field(default=TA_LIB_AVAILABLE_RUNTIME, init=False, repr=False)

    def __post_init__(self):
        # Temporal Validation config: Ensure nested dataclass is instantiated
        # This part remains here as it handles the instantiation of a nested dataclass.
        if isinstance(self.temporal_validation, dict):
            self.temporal_validation = TemporalValidationConfig(**self.temporal_validation)
        elif not isinstance(self.temporal_validation, TemporalValidationConfig):
            raise TypeError("temporal_validation must be a dictionary or TemporalValidationConfig instance.")
        
        # Import validate_feature_config here to avoid circular dependencies at module level
        # if validator.py also imports FeatureConfig. This ensures the import happens
        # only when needed and after FeatureConfig is fully defined.
        try:
            from config.validator import validate_feature_config
        except ImportError as e:
            logging.critical(f"Failed to import validate_feature_config from config.validator: {e}. Ensure validator.py exists and is accessible.", exc_info=True)
            raise

        # Call the external validation function
        validate_feature_config(self)

# Default configuration instance
DEFAULT_FEATURE_CONFIG = FeatureConfig()