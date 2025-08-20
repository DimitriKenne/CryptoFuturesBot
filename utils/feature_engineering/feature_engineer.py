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
    from config.params import app_config, FLOAT_EPSILON
    from config.feature import FeatureConfig, TemporalValidationConfig
    from utils.feature_engineering.technical_indicator_calculator import TechnicalIndicatorCalculator
    from utils.feature_engineering.indicator_feature_processor import IndicatorFeatureProcessor
    from utils.feature_engineering.price_action_feature_processor import PriceActionFeatureProcessor
    from utils.exceptions import TemporalSafetyError
except ImportError as e:
    logging.error(f"Failed to import necessary modules: {e}")
    raise

logger = logging.getLogger(__name__)

def log_nan_stats(df_before, df_after, stage_name, logger):
    new_cols = [col for col in df_after.columns if col not in df_before.columns]
    if not new_cols:
        logger.info(f"{stage_name}: No new features added.")
        return
    nan_counts = df_after[new_cols].isna().sum()
    total_nan_rows = df_after[new_cols].isna().any(axis=1).sum()
    max_nan = nan_counts.max()
    logger.info(
        f"{stage_name}: {len(new_cols)} new features. "
        f"Max NaN in any feature: {max_nan}. "
        f"Rows with any NaN: {total_nan_rows}."
    )
    # Move details to debug level
    logger.debug(f"{stage_name}: New columns: {new_cols}")
    logger.debug(f"{stage_name}: NaN counts per new column:\n{nan_counts}")

class FeatureEngineer:
    """
    Orchestrates the feature engineering process. It manages the flow of data
    through specialized sub-processors (IndicatorFeatureProcessor, PriceActionFeatureProcessor)
    to ensure features are generated in a dependency-aware manner.
    Includes temporal safety checks to prevent lookahead bias.
    """

    def __init__(self, config: Optional[FeatureConfig] = None):
        """
        Initializes the FeatureEngineer with a FeatureConfig object.

        Args:
            config (Optional[FeatureConfig]): Configuration for feature parameters.
                If None, uses app_config.features (from config/params.py).
                If a dictionary is passed, instantiates FeatureConfig.
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

        # Initialize modular feature processors - they receive the same config
        self.indicator_processor = IndicatorFeatureProcessor(config=self.config)
        self.price_action_processor = PriceActionFeatureProcessor(config=self.config)

    @property
    def required_lookback(self) -> int:
        """
        Calculates the maximum lookback required across all types of feature engineering.
        This is the number of *previous* bars needed to calculate features for the latest bar.
        This calculation needs to consider the maximum lookback of *all* individual features
        and their dependencies, including any internal shifts within sub-processors.
        For simplicity, we can defer to the sub-processors and add a buffer.
        """
        # Get maximum lookback from individual processors (which should account for their internal shifts)
        max_sub_processor_lookback = max(
            self.indicator_processor.required_lookback,
            self.price_action_processor.required_lookback
        )
        
        # Consider additional lookback for FVG (if not handled by price_action_processor's lookback)
        fvg_lookback = self.config.fvg_lookback_bars if self.config.fvg_lookback_bars > 0 else 0

        # Consider lagged and differenced features that are applied at the end
        max_lag = 0
        if self.config.lagged_features:
            max_lag = max([max(lags) for lags in self.config.lagged_features.values()] + [0])
        max_diff = 0
        if self.config.differenced_features:
            max_diff = max([max(orders) for orders in self.config.differenced_features.values()] + [0])

        # The overall required lookback is the maximum of all these, plus a small buffer
        # Added +2 as a general safety margin for potential shifts or calculations involving multiple past bars.
        calculated_lookback = max(max_sub_processor_lookback, fvg_lookback, max_lag, max_diff) + 2 

        self.logger.debug(f"Calculated required lookback for FeatureEngineer: {calculated_lookback} bars.")
        return calculated_lookback + 800


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
        
        # Log returns use previous two close prices
        df_transformed['log_returns'] = np.log(df['close'].shift(1) / df['close'].shift(2))
        
        # Typical price of the *previous* bar
        df_transformed['typical_price'] = (df['high'].shift(1) + df['low'].shift(1) + df['close'].shift(1)) / 3

        # Other essential price differences based on *shifted* data
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

    def _log_nan_stats(self, df_before, df_after, stage_name):
        logger = self.logger
        new_cols = [col for col in df_after.columns if col not in df_before.columns]
        if not new_cols:
            logger.info(f"{stage_name}: No new features added.")
            return
        nan_counts = df_after[new_cols].isna().sum()
        total_nan_rows = df_after[new_cols].isna().any(axis=1).sum()
        max_nan = nan_counts.max()
        logger.info(
            f"{stage_name}: {len(new_cols)} new features. "
            f"Max NaN in any feature: {max_nan}. "
            f"Rows with any NaN: {total_nan_rows}."
        )
        # Move details to debug level
        logger.debug(f"{stage_name}: New columns: {new_cols}")
        logger.debug(f"{stage_name}: NaN counts per new column:\n{nan_counts}")


    def _validate_temporal_safety(self, df):
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
            'is_above_s3', 'is_below_s3',
            'is_above_swing_high', 'is_below_swing_low',
            'is_support_broken_strong_vol', 'is_resistance_broken_strong_vol',
            'is_bull_wick_at_resistance', 'is_bear_wick_at_support'
        ]
        # Add candlestick pattern signals to the skip list
        for pattern in self.config.candlestick_patterns:
            cols_to_skip_correlation.append(f'pattern_{pattern}_signal')
        
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


    def process(self, df: pd.DataFrame, interval: str = None) -> pd.DataFrame:
        """
        Orchestrates the processing of raw OHLCV data to generate all technical,
        statistical, and price action features in a dependency-aware manner.
        Accepts interval for correct ADR calculation.
        """
        self.logger.info("🚦 Starting general feature engineering process.")
        self._validate_dataframe(df)
        
        # Use interval from argument, else from config, else default to '1d'
        interval_to_use = interval or getattr(self.config, "interval", "1d")

        df_processed = df.copy()

        # --- Stage 1: Basic Price Transformations and Core Indicators ---
        self.logger.info("Stage 1: Adding basic price transformations and core technical indicators.")
        df_before = df_processed.copy()
        df_price_transforms = self._add_price_transformations(df)
        df_processed = pd.concat([df_processed, df_price_transforms], axis=1)
        log_nan_stats(df_before, df_processed, "Stage 1 - Price Transformations", self.logger)

        df_before = df_processed.copy()
        # Pass interval to indicator processor for ADR calculation
        df_core_indicators = self.indicator_processor.add_core_technical_indicators(df_processed, interval=interval_to_use)
        df_processed = pd.concat([df_processed, df_core_indicators], axis=1)
        log_nan_stats(df_before, df_processed, "Stage 1 - Core Indicators", self.logger)

        df_before = df_processed.copy()
        df_patterns_fvg = self.price_action_processor.add_custom_pattern_features(df_processed)
        df_processed = pd.concat([df_processed, df_patterns_fvg], axis=1)
        log_nan_stats(df_before, df_processed, "Stage 1 - Patterns/FVG", self.logger)
        self.logger.info("Stage 1 complete. DataFrame shape: %s", df_processed.shape)

        # --- Stage 2: Mid-Level Price Action Features (Pivots, S/R) ---
        self.logger.info("Stage 2: Adding pivot point and support/resistance features.")
        df_before = df_processed.copy()
        df_pivots = self.price_action_processor.add_pivot_point_features(df_processed)
        df_processed = pd.concat([df_processed, df_pivots], axis=1)
        log_nan_stats(df_before, df_processed, "Stage 2 - Pivot Points", self.logger)

        df_before = df_processed.copy()
        df_sr = self.price_action_processor.add_support_resistance_features(df_processed)
        df_processed = pd.concat([df_processed, df_sr], axis=1)
        log_nan_stats(df_before, df_processed, "Stage 2 - Support/Resistance", self.logger)
        self.logger.info("Stage 2 complete. DataFrame shape: %s", df_processed.shape)

        # --- Stage 3: Derived Features and Breakouts ---
        self.logger.info("Stage 3: Adding derived features and breakout patterns.")
        df_before = df_processed.copy()
        df_derived_indicators = self.indicator_processor.add_derived_features(df_processed)
        df_processed = pd.concat([df_processed, df_derived_indicators], axis=1)
        log_nan_stats(df_before, df_processed, "Stage 3 - Derived Indicators", self.logger)

        df_before = df_processed.copy()
        df_breaks = self.price_action_processor.add_breakout_features(df_processed)
        df_processed = pd.concat([df_processed, df_breaks], axis=1)
        log_nan_stats(df_before, df_processed, "Stage 3 - Breakouts", self.logger)
        self.logger.info("Stage 3 complete. DataFrame shape: %s", df_processed.shape)

        # --- Stage 4: Lagged and Differenced Features ---
        self.logger.info("Stage 4: Adding lagged and differenced features.")
        df_before = df_processed.copy()
        df_lagged = self._add_lagged_features(df_processed)
        df_processed = pd.concat([df_processed, df_lagged], axis=1)
        log_nan_stats(df_before, df_processed, "Stage 4 - Lagged Features", self.logger)

        df_before = df_processed.copy()
        df_differenced = self._add_differenced_features(df_processed)
        df_processed = pd.concat([df_processed, df_differenced], axis=1)
        log_nan_stats(df_before, df_processed, "Stage 4 - Differenced Features", self.logger)
        self.logger.info("Stage 4 complete. DataFrame shape: %s", df_processed.shape)

        # --- Stage 5: Final Type Conversions, NaN Handling & Temporal Validation ---
        self.logger.info("Stage 5: Performing final type conversions and temporal validation.")
        df_with_nan = df_processed.copy()

        # Convert appropriate columns to nullable integer types (for binary/categorical features)
        # These lists are constructed dynamically based on config and expected outputs
        categorical_cols = ['fvg', 'volatility_regime', 'pattern_cluster'] # pattern_cluster is now sum, can be int/float
        
        # Standard pivot binary columns
        standard_pivot_binary_cols = []
        if self.config.pivot_point_method == 'standard':
            for p_col in ['pp', 'r1', 's1', 'r2', 's2', 'r3', 's3']:
                standard_pivot_binary_cols.append(f'is_above_{p_col}')
                standard_pivot_binary_cols.append(f'is_below_{p_col}')
        categorical_cols.extend([col for col in standard_pivot_binary_cols if col in df_with_nan.columns])

        # Swing pivot and breakout binary columns
        swing_pivot_binary_cols = [
            'is_above_swing_high', 'is_below_swing_low',
            'is_support_broken_strong_vol', 'is_resistance_broken_strong_vol',
            'is_bull_wick_at_resistance', 'is_bear_wick_at_support'
        ]
        categorical_cols.extend([col for col in swing_pivot_binary_cols if col in df_with_nan.columns])

        # Add candlestick pattern signals to the list for type casting
        for pattern in self.config.candlestick_patterns:
            col_name = f'pattern_{pattern}_signal'
            if col_name not in categorical_cols: # Avoid duplicates if defined elsewhere
                categorical_cols.append(col_name)

        for col in categorical_cols:
             if col in df_with_nan.columns:
                  # Ensure conversion only for columns that are actually binary (0, 1, -1)
                  # and are not already float-based sums (like pattern_cluster)
                  # Also, check if the column isn't all NaNs before attempting conversion
                  if not df_with_nan[col].dropna().empty and df_with_nan[col].dropna().isin([0, 1, -1]).all():
                    df_with_nan.loc[:, col] = df_with_nan[col].astype(pd.Int8Dtype())
                  else:
                    self.logger.debug(f"Column '{col}' contains values outside of [0, 1, -1], is all NaNs, or is not applicable for Int8Dtype conversion. Not casting.")
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

        # Drop initial rows with NaNs caused by lookback periods, if configured
        if self.config.remove_nan_rows:
            original_shape = df_with_nan.shape
            ohlcv_and_index_cols = ['open', 'high', 'low', 'close', 'volume', df_with_nan.index.name if df_with_nan.index.name else '']
            feature_cols_for_nan_check = [col for col in df_with_nan.columns if col not in ohlcv_and_index_cols]
            df_final = df_with_nan.dropna(subset=feature_cols_for_nan_check)
            self.logger.info(f"Removed {original_shape[0] - df_final.shape[0]} rows with NaNs from feature columns. Final shape: {df_final.shape}")
        else:
            df_final = df_with_nan
            self.logger.info("NaN row removal skipped as per configuration.")

        self.logger.info(
            f"Feature engineering summary: {df_final.shape[1]} features, "
            f"{df_final.shape[0]} rows after NaN removal."
        )
        self.logger.info("Feature engineering process completed.")
        return df_final
