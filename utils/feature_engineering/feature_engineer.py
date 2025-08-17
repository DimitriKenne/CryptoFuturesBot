# utils/feature_engineering/feature_engineer.py

import sys
import pandas as pd
import numpy as np
import logging
from pathlib import Path
from typing import Optional, Dict, Any, List
import copy

# Add project root to Python path for imports
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.append(str(PROJECT_ROOT))

# --- Import configuration from central params.py ---
try:
    # Import the aggregated app_config object
    from config.params import app_config, FLOAT_EPSILON
    # Import FeatureConfig and TemporalValidationConfig for type hints and instantiation
    from config.feature import FeatureConfig, TemporalValidationConfig
    # Import the new modular processors
    from utils.feature_engineering.technical_indicator_calculator import TechnicalIndicatorCalculator
    from utils.feature_engineering.indicator_feature_processor import IndicatorFeatureProcessor
    from utils.feature_engineering.price_action_feature_processor import PriceActionFeatureProcessor
    # Assuming TemporalSafetyError is defined in a custom exceptions.py file
    from utils.exceptions import TemporalSafetyError
except ImportError as e:
    logging.error(f"Failed to import necessary modules: {e}")
    raise

logger = logging.getLogger(__name__)

class FeatureEngineer:
    """
    Engineers technical, statistical, and price action features from OHLCV data.
    Acts as an orchestrator for specialized feature processors.
    Includes temporal safety checks to prevent lookahead bias.
    """

    def __init__(self, config: Optional[FeatureConfig] = None):
        """
        Initializes the FeatureEngineer with a FeatureConfig object.

        Args:
            config (Optional[FeatureConfig]): Configuration for feature parameters.
                If None, uses app_config.features (from config/params.py).
        """
        if config is None:
            self.config: FeatureConfig = copy.deepcopy(app_config.features)
        elif isinstance(config, dict):
            cfg_dict = copy.deepcopy(config)
            if 'temporal_validation' in cfg_dict and isinstance(cfg_dict['temporal_validation'], dict):
                cfg_dict['temporal_validation'] = TemporalValidationConfig(**cfg_dict['temporal_validation'])
            self.config: FeatureConfig = FeatureConfig(**cfg_dict)
        elif isinstance(config, FeatureConfig):
            self.config: FeatureConfig = copy.deepcopy(config)
        else:
            raise TypeError("Config must be a FeatureConfig instance or a dictionary, not " + str(type(config)))

        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.info("FeatureEngineer initialized with feature configuration.")
        self.logger.info(f"Temporal safety validation enabled: {self.config.temporal_validation.enabled}")

        # Initialize modular feature processors
        self.indicator_processor = IndicatorFeatureProcessor(config=self.config)
        self.price_action_processor = PriceActionFeatureProcessor(config=self.config)

    @property
    def required_lookback(self) -> int:
        """
        Calculates the maximum lookback required across all types of feature engineering.
        This is the number of *previous* bars needed to calculate features for the latest bar.
        """
        # Sum of maximum lookbacks from each sub-processor, plus a small buffer
        max_lookback = max(
            self.indicator_processor.required_lookback,
            self.price_action_processor.required_lookback
        )
        # Add 1 for the current bar itself and potentially one more for shifts if the base data is already shifted
        return max_lookback + 2 # A small buffer

    def _validate_dataframe(self, df: pd.DataFrame):
        """
        Validates the input DataFrame structure and integrity.
        """
        required_cols = {'open', 'high', 'low', 'close', 'volume'}
        missing = required_cols - set(df.columns)
        if missing:
            self.logger.error(f"Input DataFrame missing required columns: {missing}")
            raise ValueError(f"Input DataFrame missing required columns: {missing}")

        if not isinstance(df.index, pd.DatetimeIndex):
             self.logger.error("Input DataFrame index is not a DatetimeIndex.")
             raise ValueError("Input DataFrame index must be a pandas DatetimeIndex.")

        if df.index.is_monotonic_increasing is False:
            self.logger.error("Input DataFrame index is not monotonically increasing.")
            raise ValueError("DataFrame index must be time-sorted")

        # Check for NaNs and non-finite values only in required OHLCV columns
        for col in required_cols:
            if df[col].isnull().any():
                self.logger.warning(f"Input DataFrame contains NaN values in required column: {col}. This might affect feature calculations.")
            if not np.isfinite(df[col]).all():
                 self.logger.warning(f"Input DataFrame contains Inf or non-finite values in required column: {col}. This might affect feature calculations.")

        self.logger.debug("Input DataFrame validated successfully.")


    def _add_price_transformations(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Adds basic price transformations (e.g., log returns, typical price) to the DataFrame.
        These are calculated based on past data to ensure temporal safety.
        """
        df_transformed = pd.DataFrame(index=df.index)
        # Shift close prices for log returns to prevent lookahead
        # log_returns is usually based on (current / previous) or (current / future)
        # For current bar `t`, we use `close[t-1] / close[t-2]`
        df_transformed['log_returns'] = np.log(df['close'].shift(1) / df['close'].shift(2))
        
        # Typical price of the *previous* bar
        df_transformed['typical_price'] = (df['high'].shift(1) + df['low'].shift(1) + df['close'].shift(1)) / 3

        # Add other essential price differences based on *shifted* data
        df_transformed['mid_price'] = (df['high'].shift(1) + df['low'].shift(1)) / 2
        df_transformed['body_range'] = df['high'].shift(1) - df['low'].shift(1)
        df_transformed['open_close_diff'] = df['close'].shift(1) - df['open'].shift(1)
        df_transformed['high_low_diff'] = df['high'].shift(1) - df['low'].shift(1)

        self.logger.debug("Price transformations added.")
        return df_transformed


    def _add_lagged_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Adds lagging features for specified columns and periods (simple shifts).
        These are applied after all other features are generated.
        """
        df_lagged = pd.DataFrame(index=df.index)
        for col_name, lags in self.config.lagged_features.items():
            if col_name in df.columns:
                for lag in lags:
                    df_lagged[f'{col_name}_lag_{lag}'] = df[col_name].shift(lag)
            else:
                self.logger.warning(f"Column '{col_name}' not found for lagging features. Skipping.")
        self.logger.debug("Lagged features added.")
        return df_lagged

    def _add_differenced_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Adds differenced features for specified columns and orders.
        These are applied after all other features are generated.
        """
        df_differenced = pd.DataFrame(index=df.index)
        for col_name, orders in self.config.differenced_features.items():
            if col_name in df.columns:
                for order in orders:
                    df_differenced[f'{col_name}_diff_{order}'] = df[col_name].diff(periods=order)
            else:
                self.logger.warning(f"Column '{col_name}' not found for differencing. Skipping.")
        self.logger.debug("Differenced features added.")
        return df_differenced

    def _validate_temporal_safety(self, df: pd.DataFrame) -> List[str]:
        """
        Performs enhanced temporal safety validation by checking correlation
        between each feature and the *next* period's close price change.
        Identifies and returns a list of features that violate the thresholds.
        Now checks all generated general features.
        """
        cfg = self.config.temporal_validation
        warn_threshold = cfg.warning_correlation_threshold
        error_threshold = cfg.error_correlation_threshold

        if 'close' not in df.columns:
            self.logger.error("Cannot perform temporal safety validation: 'close' column is missing.")
            return []

        # Calculate the next period's close price change, shifted for validation against current features
        next_close_change = df['close'].pct_change().shift(-1)
        
        violating_features = []
        ohlcv_cols = {'open', 'high', 'low', 'close', 'volume'}

        # Columns that are binary/categorical and thus correlation check is not directly applicable
        # or columns that are direct outputs of current price patterns and don't need a shift.
        # This list should be kept up-to-date with all non-numeric or current-bar-dependent features.
        cols_to_skip_correlation = [
            'fvg', 'volatility_regime', 'pattern_cluster',
            'pp', 'r1', 's1', 'r2', 's2', 'r3', 's3', # Standard pivot levels
            'swing_high_pivot', 'swing_low_pivot', # Swing pivot levels
            # Binary flags related to pivots/SR/breaks - these are already based on current vs past levels
            'is_above_pp', 'is_below_pp', 'is_above_r1', 'is_below_r1',
            'is_above_s1', 'is_below_s1', 'is_above_r2', 'is_below_r2',
            'is_above_s2', 'is_below_s2', 'is_above_r3', 'is_below_r3',
            'is_above_s3', 'is_below_s3', # Added missing _s3
            'is_above_swing_high', 'is_below_swing_low',
            'is_support_broken_strong_vol', 'is_resistance_broken_strong_vol',
            'is_bull_wick_at_resistance', 'is_bear_wick_at_support'
        ]
        
        # Filter features to check for correlation: only numeric features not in OHLCV and not in skip list
        feature_cols = [col for col in df.columns 
                        if col not in ohlcv_cols and 
                           col != next_close_change.name and 
                           col not in cols_to_skip_correlation and
                           pd.api.types.is_numeric_dtype(df[col])]

        for col in feature_cols:
            x = df[col]
            y = next_close_change
            
            # Drop NaN values for accurate correlation calculation
            valid = pd.notna(x) & pd.notna(y)
            x_valid = x[valid]
            y_valid = y[valid]

            if len(x_valid) < 2 or x_valid.std() < FLOAT_EPSILON or y_valid.std() < FLOAT_EPSILON:
                self.logger.debug(f"Skipping temporal validation for {col}: Insufficient valid data points ({len(x_valid)}) or near-zero standard deviation after filtering NaNs.")
                continue

            try:
                corr = abs(x_valid.corr(y_valid))
                if pd.isna(corr):
                     self.logger.debug(f"Correlation is NaN for {col}, likely due to insufficient variability after filtering NaNs.")
                     continue
                if corr > error_threshold:
                    self.logger.error(f"Temporal safety violation: Feature '{col}' correlation with next close change ({corr:.4f}) exceeds error threshold ({error_threshold}).")
                    violating_features.append(col)
                elif corr > warn_threshold:
                    self.logger.warning(f"Temporal safety warning: Feature '{col}' correlation with next close change ({corr:.4f}) exceeds warning threshold ({warn_threshold}).")
            except Exception as e:
                self.logger.warning(f"Could not compute correlation for feature '{col}': {e}")

        self.logger.info(f"Temporal safety validation complete. {len(violating_features)} features violated the error threshold.")
        return violating_features


    def process(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Processes raw OHLCV data to generate all technical, statistical, and price action features.
        Initial rows affected by lookback periods will contain NaN values.
        Performs temporal safety checks if enabled.
        """
        self.logger.info("Starting general feature engineering process.")
        self._validate_dataframe(df)
        
        df_processed = df.copy()

        # 1. Add basic price transformations (e.g., log returns, typical price)
        df_price_transforms = self._add_price_transformations(df)
        df_processed = pd.concat([df_processed, df_price_transforms], axis=1)

        # 2. Add technical and statistical indicators, including derived ones
        # This MUST run before price action features that depend on indicators like volume_osc.
        df_tech_stats = self.indicator_processor.add_all_technical_and_derived_features(df_processed)
        df_processed = pd.concat([df_processed, df_tech_stats], axis=1)

        # 3. Add price action features (patterns, pivots, S/R, breakouts)
        # Now, df_processed contains OHLCV, price transforms, AND technical indicators,
        # so price action features (e.g., breakouts) will have access to all their dependencies.
        df_price_action = self.price_action_processor.add_all_price_action_features(df_processed)
        df_processed = pd.concat([df_processed, df_price_action], axis=1)

        # 4. Add lagged and differenced features (applied to all features generated so far)
        df_lagged = self._add_lagged_features(df_processed)
        df_processed = pd.concat([df_processed, df_lagged], axis=1)

        df_differenced = self._add_differenced_features(df_processed)
        df_processed = pd.concat([df_processed, df_differenced], axis=1)

        df_with_nan = df_processed.copy()

        # Convert appropriate columns to nullable integer types (for binary/categorical features)
        # This list should ideally be dynamic or clearly defined based on expected outputs
        categorical_cols = [
            'fvg', 'volatility_regime', 'pattern_cluster',
            'is_above_pp', 'is_below_pp', 'is_above_r1', 'is_below_r1',
            'is_above_s1', 'is_below_s1', 'is_above_r2', 'is_below_r2',
            'is_above_s2', 'is_below_s2', 'is_above_r3', 'is_below_r3',
            'is_above_s3', 'is_below_s3',
            'is_above_swing_high', 'is_below_swing_low',
            'is_support_broken_strong_vol', 'is_resistance_broken_strong_vol',
            'is_bull_wick_at_resistance', 'is_bear_wick_at_support'
        ]
        # Add candlestick pattern signals
        for pattern in self.config.candlestick_patterns:
            categorical_cols.append(f'pattern_{pattern}_signal')

        for col in categorical_cols:
             if col in df_with_nan.columns:
                  # Ensure conversion only for columns that are actually binary (0, 1, -1)
                  # and are not already float-based sums (like pattern_cluster)
                  if df_with_nan[col].dropna().isin([0, 1, -1]).all(): # Check if values are binary/ternary
                    df_with_nan.loc[:, col] = df_with_nan[col].astype(pd.Int8Dtype())
                  else:
                    self.logger.debug(f"Column '{col}' contains values outside of [0, 1, -1] or NaNs. Not casting to Int8Dtype.")
             else:
                  self.logger.debug(f"Categorical column '{col}' not found in DataFrame to cast type.")

        self.logger.info(f"General feature engineering complete. DataFrame shape (including NaNs): {df_with_nan.shape}")

        # Conditionally perform temporal safety validation
        if self.config.temporal_validation.enabled:
            self.logger.info("Performing temporal safety validation on general features...")
            violating_features = self._validate_temporal_safety(df_with_nan)
            if violating_features:
                error_msg = f"Temporal safety violations detected in general features: {', '.join(violating_features)}"
                self.logger.error(error_msg)
                raise TemporalSafetyError(error_msg, features=violating_features)
            else:
                self.logger.info("Temporal safety validation passed for general features.")
        else:
            self.logger.info("Temporal safety validation skipped as per configuration.")

        return df_with_nan

