# utils/feature_engineering/feature_name_generator.py

from typing import List
import logging
from pathlib import Path
import sys

# Add project root to Python path for imports
PROJECT_ROOT = Path(__file__).parent.parent.parent # utils/feature_engineering is two levels down from project root
sys.path.append(str(PROJECT_ROOT))

try:
    # Import FeatureConfig and DEFAULT_FEATURE_CONFIG (now a dataclass instance)
    from config.feature_config_schema import FeatureConfig, DEFAULT_FEATURE_CONFIG
except ImportError as e:
    logging.error(f"Failed to import FeatureConfig from config.feature_config_schema: {e}")
    raise

logger = logging.getLogger(__name__)

# Define FLOAT_EPSILON consistent with FeatureEngineer
# FLOAT_EPSILON = 1e-9
# Import FLOAT_EPSILON from the central constants file (config.params)
from config.params import FLOAT_EPSILON

def generate_feature_names(config: FeatureConfig) -> List[str]:
    """
    Generates a comprehensive list of all feature names that would be produced
    by the FeatureEngineer class based on the provided FeatureConfig.

    This function reflects the naming conventions and feature types implemented
    in FeatureEngineer, including base technical indicators, custom patterns,
    pivot points, derived features, lagged features, and differenced features.

    Args:
        config (FeatureConfig): An instance of FeatureConfig containing all
                                parameters for feature engineering.

    Returns:
        List[str]: A list of strings, where each string is the name of a feature.
    """
    feature_names = []

    # --- 1. Price Transformations ---
    feature_names.extend(['log_returns', 'typical_price', 'mid_price', 'body_range', 'open_close_diff', 'high_low_diff'])

    # --- 2. Momentum Indicators ---
    for period in config.rsi_periods:
        feature_names.append(f'rsi_{period}')
    
    stoch_d_period = 3 # Hardcoded in FeatureEngineer, so consistent here
    for period_k in config.stochastic_periods:
        feature_names.append(f'stoch_k_{period_k}')
        feature_names.append(f'stoch_d_{period_k}')

    if len(config.ao_periods) == 2:
        feature_names.append('ao')

    for period in config.cci_periods:
        feature_names.append(f'cci_{period}')

    for period in config.mfi_periods:
        feature_names.append(f'mfi_{period}')


    # --- 3. Trend Indicators ---
    for period in config.sma_periods:
        feature_names.append(f'sma_{period}')

    for period in config.ema_periods:
        feature_names.append(f'ema_{period}')

    feature_names.extend(['macd', 'macd_signal', 'macd_diff'])


    # --- 4. Volatility Indicators ---
    for period in config.atr_periods:
        feature_names.append(f'atr_{period}')

    for period in config.bollinger_periods:
        feature_names.append(f'bb_upper_{period}')
        feature_names.append(f'bb_lower_{period}')
        feature_names.append(f'bb_width_{period}')


    # --- 5. Volume Indicators ---
    feature_names.append('obv')
    for period in config.volume_periods:
        feature_names.append(f'cmf_{period}')
        feature_names.append(f'mfi_{period}')
    feature_names.append('volume_osc') # Volume Oscillator

    # --- 6. Statistical Features ---
    for period in config.z_score_periods:
        feature_names.append(f'z_score_{period}')
    for period in config.adr_periods:
        feature_names.append(f'adr_{period}')

    # --- 7. Custom Pattern and FVG Features ---
    for pattern in config.candlestick_patterns:
        feature_names.append(f'pattern_{pattern}_signal')
    feature_names.append('fvg')

    # --- 8. Pivot Point Features (Standard and Swing) ---
    # Standard Pivots
    standard_pivot_bases = ['pp', 'r1', 's1', 'r2', 's2', 'r3', 's3']
    if config.pivot_point_method == 'standard':
        feature_names.extend(standard_pivot_bases)

    # Swing Pivots
    feature_names.extend(['swing_high_pivot', 'swing_low_pivot'])

    # --- 9. Derived Features (depend on earlier features) ---
    feature_names.append('trend_strength')
    feature_names.append('volatility_regime')
    feature_names.append('pattern_cluster')

    # Normalized distance to Standard Pivots and binary above/below
    if config.pivot_point_method == 'standard' and config.atr_periods:
        for p_col in standard_pivot_bases:
            feature_names.append(f'dist_to_{p_col}_norm')
            feature_names.append(f'is_above_{p_col}')
            feature_names.append(f'is_below_{p_col}')
    
    # Normalized distance to Swing Pivots and binary above/below
    if config.atr_periods: # Swing pivots also rely on ATR for normalization
        feature_names.extend([
            'dist_to_swing_high_norm', 'dist_to_swing_low_norm',
            'is_above_swing_high', 'is_below_swing_low'
        ])

    # Normalized Support/Resistance Distances (now derived features)
    if config.atr_periods: # These also rely on ATR for normalization
        for period in config.support_resistance_periods:
            feature_names.append(f'dist_to_support_{period}') # Raw distance
            feature_names.append(f'dist_to_resistance_{period}') # Raw distance
            feature_names.append(f'dist_to_support_norm_{period}') # Normalized distance
            feature_names.append(f'dist_to_resistance_norm_{period}') # Normalized distance
    else: # If ATR not configured, raw S/R distances might still be useful
        for period in config.support_resistance_periods:
            feature_names.append(f'resistance_{period}') # Base S/R levels
            feature_names.append(f'support_{period}') # Base S/R levels
            # These raw distances might still be generated even if not normalized
            feature_names.append(f'dist_to_support_{period}')
            feature_names.append(f'dist_to_resistance_{period}')


    # --- 10. Breakout Features ---
    feature_names.extend([
        'is_support_broken_strong_vol', 'is_resistance_broken_strong_vol',
        'is_bull_wick_at_resistance', 'is_bear_wick_at_support'
    ])

    # --- 11. Lagged Features ---
    for col_name, lags in config.lagged_features.items():
        for lag in lags:
            feature_names.append(f'{col_name}_lag_{lag}')

    # --- 12. Differenced Features ---
    for col_name, orders in config.differenced_features.items():
        for order in orders:
            feature_names.append(f'{col_name}_diff_{order}')

    # Filter out duplicates (if any) and return
    return sorted(list(set(feature_names)))

