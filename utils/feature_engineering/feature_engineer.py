# utils/feature_engineering/feature_engineer.py

import sys
import pandas as pd
import talib
import numpy as np
import logging
from pathlib import Path
from typing import Optional, Dict, Any, List
import copy # Import copy for deepcopy

# Add project root to Python path for imports
# Adjusted PROJECT_ROOT calculation for nested folder structure (utils/feature_engineering is two levels down from project root)
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.append(str(PROJECT_ROOT))

# Import configuration and custom exception
try:
    # Import FeatureConfig, TemporalValidationConfig, and DEFAULT_FEATURE_CONFIG (now a dataclass instance)
    from config.feature_config_schema import FeatureConfig, TemporalValidationConfig, DEFAULT_FEATURE_CONFIG
    
    # Import TechnicalIndicatorCalculator from its location within feature_engineering
    from utils.feature_engineering.technical_indicator_calculator import TechnicalIndicatorCalculator
    
    # Assuming TemporalSafetyError is defined in a custom exceptions.py file
    from utils.exceptions import TemporalSafetyError
except ImportError as e:
    logging.error(f"Failed to import necessary modules: {e}")
    raise # Re-raise the exception to stop execution if essential imports fail

# Set up logger for this module
logger = logging.getLogger(__name__)

# Define FLOAT_EPSILON for robust floating-point comparisons
# FLOAT_EPSILON = 1e-9
# Import FLOAT_EPSILON from the central constants file (config.params)
from config.params import FLOAT_EPSILON

class FeatureEngineer:
    """
    Engineers technical and statistical features from OHLCV data.
    Focuses on generating general-purpose features independent of specific
    trading strategy execution logic or labeling methods.
    Calculates indicators for lists of periods as defined in FEATURE_CONFIG.
    Includes temporal safety checks to prevent lookahead bias.
    """

    def __init__(self, config: Optional[FeatureConfig] = None):
        """
        Initializes the FeatureEngineer with a FeatureConfig object.
        
        Args:
            config (Optional[FeatureConfig]): Configuration for feature parameters.
                                               If None, defaults to a deep copy of
                                               config.feature_config_schema.DEFAULT_FEATURE_CONFIG.
                                               If a dictionary is passed (e.g., from merging),
                                               it will be used to instantiate FeatureConfig.
        """
        if config is None:
            # Use a deep copy of the default dataclass instance
            self.config: FeatureConfig = copy.deepcopy(DEFAULT_FEATURE_CONFIG)
        elif isinstance(config, dict):
            # If a dictionary is provided (e.g., from merged config),
            # ensure nested TemporalValidationConfig is handled and then instantiate FeatureConfig
            cfg_dict = copy.deepcopy(config)
            if 'temporal_validation' in cfg_dict and isinstance(cfg_dict['temporal_validation'], dict):
                cfg_dict['temporal_validation'] = TemporalValidationConfig(**cfg_dict['temporal_validation'])
            self.config: FeatureConfig = FeatureConfig(**cfg_dict)
        elif isinstance(config, FeatureConfig):
            # If an FeatureConfig instance is passed, use a deep copy of it
            self.config: FeatureConfig = copy.deepcopy(config)
        else:
            raise TypeError("Config must be a FeatureConfig instance or a dictionary, not " + str(type(config)))


        logger.info("FeatureEngineer initialized with general feature configuration.")
        logger.info(f"Temporal safety validation enabled: {self.config.temporal_validation.enabled}")

    @property
    def required_lookback(self) -> int:
        """
        Calculates the minimum number of data points required for feature engineering
        based on the largest lookback period configured for any indicator.
        This is the number of *previous* bars needed to calculate features for the latest bar.
        """
        period_sizes = []

        single_period_keys = ['fvg_lookback_bars', 'swing_pivot_left_bars', 'swing_pivot_right_bars', 'sequence_length_bars']
        for key in single_period_keys:
             value = getattr(self.config, key)
             if isinstance(value, int) and value > 0:
                 period_sizes.append(value)

        list_period_keys = [
            'sma_periods', 'ema_periods', 'rsi_periods', 'bollinger_periods',
            'atr_periods', 'stochastic_periods', 'ao_periods', 'cci_periods',
            'mfi_periods', 'volume_periods', 'support_resistance_periods',
            'z_score_periods', 'adr_periods', 'trend_strength_periods'
        ]
        for key in list_period_keys:
            values = getattr(self.config, key)
            if isinstance(values, list):
                for period in values:
                    if isinstance(period, int) and period > 0:
                        period_sizes.append(period)

        period_sizes.append(self.config.swing_pivot_left_bars + self.config.swing_pivot_right_bars + 1)
        period_sizes.append(self.config.volume_oscillator_long_ema)

        if self.config.lagged_features:
            max_lag = max([max(lags) for lags in self.config.lagged_features.values()] + [0])
            period_sizes.append(max_lag)

        if self.config.differenced_features:
            max_diff_order = max([max(orders) for orders in self.config.differenced_features.values()] + [0])
            period_sizes.append(max_diff_order)

        max_period_size = max(period_sizes) if period_sizes else 0
        calculated_lookback = max_period_size + 2 # +1 for current bar, +1 for shift

        logger.debug(f"Calculated required lookback for FeatureEngineer: {calculated_lookback} bars (max_period: {max_period_size}).")
        return calculated_lookback


    def _validate_dataframe(self, df: pd.DataFrame):
        """
        Validates the input DataFrame structure and integrity.
        """
        required_cols = {'open', 'high', 'low', 'close', 'volume'}
        missing = required_cols - set(df.columns)
        if missing:
            logger.error(f"Input DataFrame missing required columns: {missing}")
            raise ValueError(f"Input DataFrame missing required columns: {missing}")

        if not isinstance(df.index, pd.DatetimeIndex):
             logger.error("Input DataFrame index is not a DatetimeIndex.")
             raise ValueError("Input DataFrame index must be a pandas DatetimeIndex.")

        if df.index.is_monotonic_increasing is False:
            logger.error("Input DataFrame index is not monotonically increasing.")
            raise ValueError("DataFrame index must be time-sorted")

        for col in required_cols:
            if df[col].isnull().any():
                logger.error(f"Input DataFrame contains NaN values in column: {col}")
                raise ValueError(f"Input contains NaN values in column: {col}")
            if not np.isfinite(df[col]).all():
                 logger.error(f"Input DataFrame contains Inf or non-finite values in column: {col}")
                 raise ValueError(f"Input contains Inf or non-finite values in column: {col}")

        logger.debug("Input DataFrame validated successfully.")


    def _add_price_transformations(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Adds basic price transformations (e.g., log returns, typical price) to the DataFrame.
        These are calculated based on past data to ensure temporal safety.
        """
        df_transformed = pd.DataFrame(index=df.index)
        # Shift close prices for log returns to prevent lookahead
        # log_returns is usually based on (current / previous) or (current / future)
        # Here we calculate log_returns of *previous* bar from its previous
        # For current bar `t`, we use `close[t-1] / close[t-2]`
        df_transformed['log_returns'] = np.log(df['close'].shift(1) / df['close'].shift(2))
        
        # Typical price of the *previous* bar
        df_transformed['typical_price'] = (df['high'].shift(1) + df['low'].shift(1) + df['close'].shift(1)) / 3

        # Add other essential price differences based on *shifted* data
        df_transformed['mid_price'] = (df['high'].shift(1) + df['low'].shift(1)) / 2
        df_transformed['body_range'] = df['high'].shift(1) - df['low'].shift(1)
        df_transformed['open_close_diff'] = df['close'].shift(1) - df['open'].shift(1)
        df_transformed['high_low_diff'] = df['high'].shift(1) - df['low'].shift(1)

        logger.debug("Price transformations added.")
        return df_transformed


    def _add_momentum_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Adds various momentum indicators (RSI, Stochastic, Awesome Oscillator, CCI, MFI).
        Calculations are based on past data for temporal safety.
        """
        df_momentum = pd.DataFrame(index=df.index)

        # Shift input data for temporal safety
        df_shifted = df[['open', 'high', 'low', 'close', 'volume']].shift(1)
        shifted_high = df_shifted['high']
        shifted_low = df_shifted['low']
        shifted_close = df_shifted['close']
        shifted_volume = df_shifted['volume']

        for period in self.config.rsi_periods:
             df_momentum[f'rsi_{period}'] = TechnicalIndicatorCalculator.calculate_rsi(shifted_close, period)

        stoch_d_period = 3 # This is often a fixed standard for the %D line
        for period_k in self.config.stochastic_periods:
            stoch_results = TechnicalIndicatorCalculator.calculate_stochastic_oscillator(
                high_prices=shifted_high, low_prices=shifted_low, close_prices=shifted_close,
                window=period_k, smooth_window=stoch_d_period
            )
            df_momentum[f'stoch_k_{period_k}'] = stoch_results['stoch']
            df_momentum[f'stoch_d_{period_k}'] = stoch_results['stoch_signal']

        if len(self.config.ao_periods) == 2:
             ao_result = TechnicalIndicatorCalculator.calculate_awesome_oscillator(
                 high_prices=shifted_high, low_prices=shifted_low,
                 window1=self.config.ao_periods[0], window2=self.config.ao_periods[1]
             )
             df_momentum['ao'] = ao_result
        else:
             logger.warning(f"AO periods not correctly configured as a pair (expected 2, got {len(self.config.ao_periods)}): {self.config.ao_periods}. Skipping AO feature.")
             df_momentum['ao'] = np.nan

        for period in self.config.cci_periods:
            df_momentum[f'cci_{period}'] = TechnicalIndicatorCalculator.calculate_cci(
                high_prices=shifted_high, low_prices=shifted_low, close_prices=shifted_close, window=period
            )

        for period in self.config.mfi_periods:
             df_momentum[f'mfi_{period}'] = TechnicalIndicatorCalculator.calculate_mfi(
                 high_prices=shifted_high, low_prices=shifted_low, close_prices=shifted_close,
                 volume=shifted_volume, window=period
             )

        logger.debug("Momentum indicators added.")
        return df_momentum


    def _add_trend_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Adds various trend indicators (SMA, EMA, MACD).
        Calculations are based on past data for temporal safety.
        """
        df_trend = pd.DataFrame(index=df.index)

        shifted_close = df['close'].shift(1)

        for period in self.config.sma_periods:
            df_trend[f'sma_{period}'] = TechnicalIndicatorCalculator.calculate_sma(shifted_close, period)

        for period in self.config.ema_periods:
            df_trend[f'ema_{period}'] = TechnicalIndicatorCalculator.calculate_ema(shifted_close, period)

        macd_results = TechnicalIndicatorCalculator.calculate_macd(shifted_close)
        df_trend['macd'] = macd_results['macd']
        df_trend['macd_signal'] = macd_results['macd_signal']
        df_trend['macd_diff'] = macd_results['macd_diff']

        logger.debug("Trend indicators added.")
        return df_trend


    def _add_volatility_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Adds various volatility indicators (ATR, Bollinger Bands).
        Calculations are based on past data for temporal safety.
        """
        df_volatility = pd.DataFrame(index=df.index)

        # Shift input data for temporal safety
        df_shifted = df[['high', 'low', 'close']].shift(1)
        shifted_high = df_shifted['high']
        shifted_low = df_shifted['low']
        shifted_close = df_shifted['close']

        for period in self.config.atr_periods:
             # ATR needs (high, low, close) of *previous* bar, use shifted_data
             df_volatility[f'atr_{period}'] = TechnicalIndicatorCalculator.calculate_atr(
                 high_prices=shifted_high,
                 low_prices=shifted_low,
                 close_prices=shifted_close,
                 window=period
             )
             # Log status of ATR calculation
             if f'atr_{period}' in df_volatility.columns:
                 num_nans = df_volatility[f'atr_{period}'].isnull().sum()
                 num_zeros = (df_volatility[f'atr_{period}'] == 0).sum()
                 logger.debug(f"ATR column 'atr_{period}' status: {num_nans} NaNs, {num_zeros} zeros out of {len(df_volatility)} rows.")
             else:
                 logger.debug(f"ATR column 'atr_{period}' was not created.")

        for period in self.config.bollinger_periods:
            bb_results = TechnicalIndicatorCalculator.calculate_bollinger_bands(shifted_close, period)
            df_volatility[f'bb_upper_{period}'] = bb_results['hband']
            df_volatility[f'bb_lower_{period}'] = bb_results['lband']
            df_volatility[f'bb_width_{period}'] = bb_results['wband'] # Width is often used as a feature

        logger.debug("Volatility indicators added.")
        return df_volatility


    def _add_volume_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Adds various volume indicators (OBV, CMF, Volume Oscillator).
        Calculations are based on past data for temporal safety.
        """
        df_volume = pd.DataFrame(index=df.index)

        # Shift input data for temporal safety
        df_shifted = df[['high', 'low', 'close', 'volume']].shift(1)
        shifted_high = df_shifted['high']
        shifted_low = df_shifted['low']
        shifted_close = df_shifted['close']
        shifted_volume = df_shifted['volume']

        df_volume['obv'] = TechnicalIndicatorCalculator.calculate_obv(shifted_close, shifted_volume)

        for period in self.config.volume_periods:
             df_volume[f'cmf_{period}'] = TechnicalIndicatorCalculator.calculate_cmf(
                 high_prices=shifted_high, low_prices=shifted_low, close_prices=shifted_close,
                 volume=shifted_volume, window=period
             )
             df_volume[f'mfi_{period}'] = TechnicalIndicatorCalculator.calculate_mfi(
                 high_prices=shifted_high, low_prices=shifted_low, close_prices=shifted_close,
                 volume=shifted_volume, window=period
             )

        # Volume oscillator based on shifted volume for temporal safety
        df_volume['volume_osc'] = TechnicalIndicatorCalculator.calculate_volume_oscillator(
            volume=df['volume'].shift(1),
            short_ema_window=self.config.volume_oscillator_short_ema,
            long_ema_window=self.config.volume_oscillator_long_ema
        )

        logger.debug("Volume indicators added.")
        return df_volume


    def _add_statistical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Adds statistical features like Z-scores and Average Daily Range (ADR).
        """
        df_stats = pd.DataFrame(index=df.index)

        shifted_close = df['close'].shift(1)

        for period in self.config.z_score_periods:
            df_stats[f'z_score_{period}'] = TechnicalIndicatorCalculator.calculate_z_score(shifted_close, period)

        for period in self.config.adr_periods:
            # Resample OHLCV data to daily frequency for ADR calculation, then shift for temporal safety
            # ADR is typically calculated on daily (or higher interval) ranges
            # Here we take the high/low of the *previous* day's aggregated data
            resampled_df = df.resample('D').agg({'high': 'max', 'low': 'min'})
            daily_adr = TechnicalIndicatorCalculator.calculate_adr(
                high_prices=resampled_df['high'].shift(1), # Shift aggregated daily high
                low_prices=resampled_df['low'].shift(1),  # Shift aggregated daily low
                window=period
            )
            
            # Reindex back to original frequency, forward-filling to propagate daily ADR to intraday bars
            df_stats[f'adr_{period}'] = daily_adr.reindex(df.index, method='ffill')

        logger.debug("Statistical features added.")
        return df_stats


    def _add_custom_pattern_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Adds Fair Value Gap (FVG) and Candlestick Pattern detection features.
        These are calculated based on *current* bar data (no shift for pattern recognition itself,
        as patterns are identified on the completed bar).
        """
        df_patterns_fvg = pd.DataFrame(index=df.index)

        # Use current bar's OHLC for pattern detection
        current_open = df['open']
        current_high = df['high']
        current_low = df['low']
        current_close = df['close']

        pattern_map = {
            'hammer': talib.CDLHAMMER,
            'engulfing': talib.CDLENGULFING,
            'doji': talib.CDLDOJI,
            'evening_star': talib.CDLEVENINGSTAR,
            'morning_star': talib.CDLMORNINGSTAR,
            'harami': talib.CDLHARAMI,
            'shooting_star': talib.CDLSHOOTINGSTAR,
            'dark_cloud_cover': talib.CDLDARKCLOUDCOVER,
            'piercing_pattern': talib.CDLPIERCING
        }

        for pattern in self.config.candlestick_patterns:
            if pattern in pattern_map:
                pattern_values = TechnicalIndicatorCalculator.detect_candlestick_pattern(
                    open_prices=current_open,
                    high_prices=current_high,
                    low_prices=current_low,
                    close_prices=current_close,
                    pattern_func=pattern_map[pattern]
                )
                # Convert TA-Lib's 100/-100/0 output to 1/-1/0 for consistency
                df_patterns_fvg[f'pattern_{pattern}_signal'] = np.where(pattern_values == 100, 1, np.where(pattern_values == -100, -1, 0))
            else:
                logger.warning(f"Configured pattern '{pattern}' is not supported by the current TA-Lib implementation or not recognized. Skipping.")

        # Fair Value Gap (FVG) - requires lookback relative to current bar
        if self.config.fvg_lookback_bars >= 1: # For FVG(3), it's close[t-1] and close[t-2] vs current.
                                             # fvg_lookback_bars = 1 would mean prev_candle's high/low
            # Bullish FVG: Current candle's low > high of the candle `fvg_lookback_bars` ago
            bullish_fvg_at_t = (df['low'] > df['high'].shift(self.config.fvg_lookback_bars))
            # Bearish FVG: Current candle's high < low of the candle `fvg_lookback_bars` ago
            bearish_fvg_at_t = (df['high'] < df['low'].shift(self.config.fvg_lookback_bars))

            df_patterns_fvg['fvg'] = 0 # Default to neutral
            df_patterns_fvg.loc[bullish_fvg_at_t, 'fvg'] = 1
            df_patterns_fvg.loc[bearish_fvg_at_t, 'fvg'] = -1
        else:
            logger.warning(f"FVG lookback bars ({self.config.fvg_lookback_bars}) is too small. Skipping FVG feature.")
            df_patterns_fvg['fvg'] = 0 # Ensure column exists even if skipped

        logger.debug("Custom pattern features (candlestick and FVG) added.")
        return df_patterns_fvg


    def _add_pivot_point_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Adds Standard Pivot Points and Pine Script-style Swing Pivots.
        Standard Pivots are based on previous period's OHLC (e.g., previous day).
        Swing Pivots are defined by local highs/lows over a specified window.
        """
        df_pivots = pd.DataFrame(index=df.index)

        # --- Standard Pivot Points (PP, R1/S1, R2/S2, R3/S3) ---
        # Calculation based on the *previous* period's (e.g., previous day's) OHLC.
        # This inherently ensures temporal safety if resampled data is shifted.
        resample_rule = None
        if self.config.pivot_point_calculation_period == 'daily':
            resample_rule = 'D'
        elif self.config.pivot_point_calculation_period == 'weekly':
            resample_rule = 'W'
        elif self.config.pivot_point_calculation_period == 'monthly':
            resample_rule = 'M'
        else:
            logger.error(f"Unsupported pivot_point_calculation_period: {self.config.pivot_point_calculation_period}. Skipping standard pivot features.")

        if resample_rule:
            # Aggregate OHLC for the previous period
            # Use .shift(1) on the aggregated data to get the *previous* period's OHLC
            prev_period_ohlc = df[['open', 'high', 'low', 'close']].resample(resample_rule).agg({
                'open': 'first',
                'high': 'max',
                'low': 'min',
                'close': 'last'
            }).shift(1) # Shift by 1 to get previous period

            # Drop NaNs from aggregated data to avoid calculating pivots on incomplete periods
            valid_periods = prev_period_ohlc.dropna()

            if not valid_periods.empty:
                # Calculate Standard Pivot Points
                # PP = (High + Low + Close) / 3
                # R1 = (2 * PP) - Low
                # S1 = (2 * PP) - High
                # ... and so on for R2/S2, R3/S3
                pp = (valid_periods['high'] + valid_periods['low'] + valid_periods['close']) / 3
                r1 = (2 * pp) - valid_periods['low']
                s1 = (2 * pp) - valid_periods['high']
                r2 = pp + (valid_periods['high'] - valid_periods['low'])
                s2 = pp - (valid_periods['high'] - valid_periods['low'])
                r3 = valid_periods['high'] + (pp - valid_periods['low'])
                s3 = valid_periods['low'] - (pp - valid_periods['high'])

                temp_pivots = pd.DataFrame({
                    'pp': pp, 'r1': r1, 's1': s1, 'r2': r2, 's2': s2, 'r3': r3, 's3': s3
                }, index=valid_periods.index)
                
                # Reindex back to original frequency and forward-fill values
                df_pivots_standard = temp_pivots.reindex(df.index, method='ffill')
                df_pivots = pd.concat([df_pivots, df_pivots_standard], axis=1)
                logger.debug(f"Standard Pivot points calculated using {self.config.pivot_point_calculation_period} aggregation and {self.config.pivot_point_method} method.")
            else:
                logger.warning(f"No valid previous period data found for {self.config.pivot_point_calculation_period} pivot point calculation. Skipping standard pivot features.")
                # Ensure columns exist even if empty
                for col in ['pp', 'r1', 's1', 'r2', 's2', 'r3', 's3']:
                    df_pivots[col] = np.nan

        # --- Swing Pivots (Pine Script style local high/low) ---
        # These identify a bar as a swing high/low if it's the highest/lowest
        # within 'left_bars' to its left and 'right_bars' to its right.
        # To ensure temporal safety, these are calculated for `df.shift(1)` data
        # and then results are mapped back to the original index.
        left_bars = self.config.swing_pivot_left_bars
        right_bars = self.config.swing_pivot_right_bars
        
        # Calculate on shifted data
        df_shifted_for_swing = df[['high', 'low']].shift(1)
        
        n = len(df_shifted_for_swing)
        swing_highs_temp = pd.Series(np.nan, index=df_shifted_for_swing.index)
        swing_lows_temp = pd.Series(np.nan, index=df_shifted_for_swing.index)

        # Iterate through the *shifted* DataFrame to find swing points
        for i in range(n):
            if i >= left_bars and i + right_bars < n:
                # Check for Swing High on shifted high
                window_highs = df_shifted_for_swing['high'].iloc[i - left_bars : i + right_bars + 1]
                if df_shifted_for_swing['high'].iloc[i] == window_highs.max():
                    swing_highs_temp.iloc[i] = df_shifted_for_swing['high'].iloc[i]

                # Check for Swing Low on shifted low
                window_lows = df_shifted_for_swing['low'].iloc[i - left_bars : i + right_bars + 1]
                if df_shifted_for_swing['low'].iloc[i] == window_lows.min():
                    swing_lows_temp.iloc[i] = df_shifted_for_swing['low'].iloc[i]

        df_swing_pivots_temp = pd.DataFrame(index=df.index)
        # Forward fill the identified swing points, and then shift *again*
        # by `right_bars + 1` to ensure we are only using *past* swing pivots.
        df_swing_pivots_temp['swing_high_pivot'] = swing_highs_temp.ffill().shift(right_bars + 1)
        df_swing_pivots_temp['swing_low_pivot'] = swing_lows_temp.ffill().shift(right_bars + 1)
        
        df_pivots = pd.concat([df_pivots, df_swing_pivots_temp], axis=1)

        logger.debug("Pivot point features added.")
        return df_pivots


    def _add_support_resistance_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculates simple support and resistance levels based on rolling highest high and lowest low.
        These are calculated based on *past* data for temporal safety.
        """
        df_sr = pd.DataFrame(index=df.index)
        
        # Shifted high/low for temporal safety
        shifted_high = df['high'].shift(1)
        shifted_low = df['low'].shift(1)

        for period in self.config.support_resistance_periods:
            # Resistance: Highest high over the period (shifted to avoid lookahead)
            df_sr[f'resistance_{period}'] = shifted_high.rolling(window=period, min_periods=1).max()
            # Support: Lowest low over the period (shifted to avoid lookahead)
            df_sr[f'support_{period}'] = shifted_low.rolling(window=period, min_periods=1).min()
        
        logger.debug("Support and Resistance features added.")
        return df_sr


    def _add_breakout_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Detects support/resistance breaks and wick patterns based on Pine Script logic.
        Requires 'swing_high_pivot', 'swing_low_pivot', 'volume_osc' features to be present.
        These are calculated based on current bar's attributes vs. lagged pivot/S/R levels.
        """
        df_breaks = pd.DataFrame(index=df.index)

        # Ensure necessary columns are available. These columns are expected to be generated
        # by previous steps and present in the combined 'df' passed to this method.
        required_cols = ['swing_high_pivot', 'swing_low_pivot', 'volume_osc', 'open', 'high', 'low', 'close']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            logger.warning(f"Missing required columns for breakout features: {missing_cols}. Skipping breakout detection.")
            # Ensure the output columns exist with default values (0 for binary)
            for col in ['is_support_broken_strong_vol', 'is_resistance_broken_strong_vol',
                        'is_bull_wick_at_resistance', 'is_bear_wick_at_support']:
                df_breaks[col] = 0
            return df_breaks

        # Use current bar's OHLC and previously calculated pivot/oscillator values
        current_open = df['open']
        current_close = df['close']
        current_high = df['high']
        current_low = df['low']

        high_pivot = df['swing_high_pivot']
        low_pivot = df['swing_low_pivot']
        volume_osc = df['volume_osc']

        # --- Resistance Breakout with Strong Volume ---
        # Condition: Current close > previous high pivot AND previous close <= high pivot AND strong volume
        # Adjusted logic for current candle crossing previous high pivot
        is_resistance_crossover = (df['close'].shift(1) <= high_pivot) & (df['close'] > high_pivot)
        is_strong_bullish_body = (current_close > current_open) & ((current_close - current_open) / (current_high - current_low + FLOAT_EPSILON) > 0.6)

        df_breaks['is_resistance_broken_strong_vol'] = (
            is_resistance_crossover &
            is_strong_bullish_body &
            (volume_osc > self.config.volume_threshold)
        ).astype(int)

        # --- Support Breakout with Strong Volume ---
        # Condition: Current close < previous low pivot AND previous close >= low pivot AND strong volume
        is_support_crossunder = (df['close'].shift(1) >= low_pivot) & (df['close'] < low_pivot)
        is_strong_bearish_body = (current_close < current_open) & ((current_open - current_close) / (current_high - current_low + FLOAT_EPSILON) > 0.6)

        df_breaks['is_support_broken_strong_vol'] = (
            is_support_crossunder &
            is_strong_bearish_body &
            (volume_osc > self.config.volume_threshold)
        ).astype(int)

        # --- Bullish Wick at Resistance (e.g., rejection from resistance) ---
        # Condition: Price touched/crossed resistance and reversed, forming a long upper wick
        # Simplified: if current high > high_pivot AND current close is significantly below high AND long upper wick
        # (current high - max(current_open, current_close)) > (max(current_open, current_close) - current_low)
        is_long_upper_wick = (current_high - current_close) > (current_close - current_low) # Simple approx for upper wick prominence
        df_breaks['is_bull_wick_at_resistance'] = (
            (current_high > high_pivot) & # Price went above resistance
            (current_close < high_pivot) & # But closed below resistance
            is_long_upper_wick # And has a long upper wick
        ).astype(int)

        # --- Bearish Wick at Support (e.g., bounce from support) ---
        # Condition: Price touched/crossed support and reversed, forming a long lower wick
        is_long_lower_wick = (current_close - current_low) > (current_high - current_close) # Simple approx for lower wick prominence
        df_breaks['is_bear_wick_at_support'] = (
            (current_low < low_pivot) & # Price went below support
            (current_close > low_pivot) & # But closed above support
            is_long_lower_wick # And has a long lower wick
        ).astype(int)


        logger.debug("Breakout and wick features added.")
        return df_breaks


    def _add_all_derived_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Creates all secondary and tertiary features from base indicators and other features.
        Ensures temporal safety for derived features.
        These features might use the *current* values of primary indicators, as long as
        those primary indicators themselves were generated from *past* OHLCV data.
        """
        df_derived = pd.DataFrame(index=df.index)

        # --- Trend Strength ---
        if len(self.config.trend_strength_periods) == 2:
            short_period, long_period = self.config.trend_strength_periods
            short_ema_col = f'ema_{short_period}'
            long_ema_col = f'ema_{long_period}'

            # Check if base EMA features exist
            if short_ema_col in df.columns and long_ema_col in df.columns:
                long_ema_series = df[long_ema_col]
                # Avoid division by zero by replacing 0 with NaN or a small epsilon for ratio calculation
                safe_long_ema = long_ema_series.replace(0, np.nan) 
                
                # Trend strength as percentage difference between short and long EMA
                df_derived['trend_strength'] = (df[short_ema_col] - safe_long_ema) / (safe_long_ema + FLOAT_EPSILON)
            else:
                logger.warning(f"Base EMA features '{short_ema_col}' or '{long_ema_col}' missing for trend strength calculation. Skipping.")
                df_derived['trend_strength'] = np.nan # Ensure column exists

        else:
            logger.warning(f"Trend strength periods not correctly configured as a pair (expected 2, got {len(self.config.trend_strength_periods)}): {self.config.trend_strength_periods}. Skipping trend_strength feature.")
            df_derived['trend_strength'] = np.nan # Ensure column exists


        # --- Volatility Regime (categorical) ---
        # This uses ATR, which is derived from shifted OHLC, so it's temporally safe.
        if self.config.atr_periods:
            atr_period_for_regime = min(self.config.atr_periods) # Use the smallest ATR for consistency
            atr_col_name_for_regime = f'atr_{atr_period_for_regime}'

            if atr_col_name_for_regime in df.columns:
                atr_series = df[atr_col_name_for_regime]
                atr_series_dropna = atr_series.dropna().copy()
                
                if atr_series_dropna.shape[0] >= 3 and len(atr_series_dropna.unique()) >= 2:
                    try:
                        # Qcut assigns labels 0, 1, 2 for low, medium, high volatility
                        df_derived.loc[atr_series_dropna.index, 'volatility_regime'] = pd.qcut(
                            atr_series_dropna,
                            q=3,
                            labels=False, # Use integer labels 0, 1, 2
                            duplicates='drop' # Handle cases with identical quantiles
                        ).astype(pd.Int8Dtype())
                    except Exception as e:
                        logger.warning(f"Could not compute volatility regime with qcut: {e}. Filling with NaN.", exc_info=True)
                        df_derived['volatility_regime'] = pd.NA # Use pd.NA for nullable integer dtype
                else:
                    logger.warning("Insufficient unique ATR values or data points to compute volatility regime. Filling with NaN.")
                    df_derived['volatility_regime'] = pd.NA
            else:
                logger.warning(f"'{atr_col_name_for_regime}' feature missing for volatility regime calculation. Skipping.")
                df_derived['volatility_regime'] = pd.NA # Ensure column exists
        else:
            logger.warning("No ATR periods configured. Cannot compute volatility regime. Skipping.")
            df_derived['volatility_regime'] = pd.NA # Ensure column exists


        # --- Pattern Clustering (Simple Sum of Binary Patterns) ---
        # This feature combines the candlestick pattern signals into a single score.
        pattern_cols = [f"pattern_{p}_signal" for p in self.config.candlestick_patterns]
        # Filter to only include columns that actually exist in the DataFrame
        existing_pattern_cols = [col for col in pattern_cols if col in df.columns]

        if not existing_pattern_cols:
             logger.warning("No configured pattern signal columns found in DataFrame for pattern clustering. Skipping feature.")
             df_derived['pattern_cluster'] = np.nan
        else:
            # Sum the binary pattern signals. A higher positive sum indicates more bullish patterns.
            df_derived['pattern_cluster'] = df[existing_pattern_cols].sum(axis=1)


        # --- Relative distance to Standard Pivot Points (normalized by ATR) ---
        # And binary features for being above/below pivots
        if self.config.pivot_point_method == 'standard' and self.config.atr_periods:
            atr_period_for_norm = min(self.config.atr_periods)
            atr_col = f'atr_{atr_period_for_norm}'

            # Ensure ATR column exists and is not all NaN/zero before normalization
            if atr_col in df.columns and not df[atr_col].isnull().all() and (df[atr_col].abs() > FLOAT_EPSILON).any():
                standard_pivot_cols = ['pp', 'r1', 's1', 'r2', 's2', 'r3', 's3']
                for p_col in standard_pivot_cols:
                    if p_col in df.columns:
                        # Distance to pivot / ATR
                        df_derived[f'dist_to_{p_col}_norm'] = (df['close'] - df[p_col]) / (df[atr_col] + FLOAT_EPSILON)
                        # Is current close above/below pivot?
                        df_derived[f'is_above_{p_col}'] = (df['close'] > df[p_col]).astype(pd.Int8Dtype())
                        df_derived[f'is_below_{p_col}'] = (df['close'] < df[p_col]).astype(pd.Int8Dtype())
                    else:
                        logger.debug(f"Standard pivot point column '{p_col}' not found for relative distance calculation.")
            else:
                logger.warning(f"ATR column '{atr_col}' not available or all NaN/near-zero for Standard Pivot Point normalization. Skipping normalized Standard Pivot Point distances and binary flags.")
        else:
            logger.warning("Standard pivot points not configured or ATR not available for normalized pivot point distances. Skipping.")


        # --- Relative distance to Swing Pivots (normalized by ATR) ---
        # And binary features for being above/below swing pivots
        if self.config.atr_periods:
            atr_period_for_norm = min(self.config.atr_periods)
            atr_col = f'atr_{atr_period_for_norm}'
            
            # Ensure ATR and swing pivot columns exist
            if atr_col in df.columns and not df[atr_col].isnull().all() and (df[atr_col].abs() > FLOAT_EPSILON).any():
                if 'swing_high_pivot' in df.columns and 'swing_low_pivot' in df.columns:
                    df_derived['dist_to_swing_high_norm'] = (df['close'] - df['swing_high_pivot']) / (df[atr_col] + FLOAT_EPSILON)
                    df_derived['dist_to_swing_low_norm'] = (df['close'] - df['swing_low_pivot']) / (df[atr_col] + FLOAT_EPSILON)
                    # Binary flags for current close relative to swing pivots
                    df_derived['is_above_swing_high'] = (df['close'] > df['swing_high_pivot']).astype(pd.Int8Dtype())
                    df_derived['is_below_swing_low'] = (df['close'] < df['swing_low_pivot']).astype(pd.Int8Dtype())
                else:
                    logger.warning("Swing pivot columns ('swing_high_pivot', 'swing_low_pivot') not found for relative distance calculation. Skipping.")
            else:
                logger.warning(f"ATR column '{atr_col}' not available or all NaN/near-zero for Swing Pivot normalization. Skipping normalized Swing Pivot distances and binary flags.")
        else:
            logger.warning("ATR not available for normalized swing pivot distances. Skipping.")

        # --- Support/Resistance Distances (raw and normalized) ---
        # These features now explicitly use shifted close/S/R to prevent lookahead from their raw calculation
        for period in self.config.support_resistance_periods:
            if f'support_{period}' in df.columns and f'resistance_{period}' in df.columns:
                # Raw distances: `close_prev - support_level` and `resistance_level - close_prev`
                df_derived[f'dist_to_support_{period}'] = df['close'].shift(1) - df[f'support_{period}']
                df_derived[f'dist_to_resistance_{period}'] = df[f'resistance_{period}'] - df['close'].shift(1)

                if self.config.atr_periods:
                    atr_period_for_norm = min(self.config.atr_periods)
                    atr_col = f'atr_{atr_period_for_norm}'

                    if atr_col in df.columns and not df[atr_col].isnull().all() and (df[atr_col].abs() > FLOAT_EPSILON).any():
                        # Normalized distances: `raw_distance / ATR`
                        # Need to handle potential NaNs from shifted data in dist_to_support/resistance
                        valid_dist_sup = df_derived[f'dist_to_support_{period}'].notna() & df[atr_col].notna() & (df[atr_col].abs() > FLOAT_EPSILON)
                        valid_dist_res = df_derived[f'dist_to_resistance_{period}'].notna() & df[atr_col].notna() & (df[atr_col].abs() > FLOAT_EPSILON)

                        df_derived.loc[valid_dist_sup, f'dist_to_support_norm_{period}'] = \
                            df_derived.loc[valid_dist_sup, f'dist_to_support_{period}'] / (df.loc[valid_dist_sup, atr_col] + FLOAT_EPSILON)
                        
                        df_derived.loc[valid_dist_res, f'dist_to_resistance_norm_{period}'] = \
                            df_derived.loc[valid_dist_res, f'dist_to_resistance_{period}'] / (df.loc[valid_dist_res, atr_col] + FLOAT_EPSILON)
                    else:
                        logger.warning(f"ATR column '{atr_col}' not available or all NaN/near-zero for S/R normalization for period {period}. Skipping normalized S/R distance.")
                        df_derived[f'dist_to_support_norm_{period}'] = np.nan
                        df_derived[f'dist_to_resistance_norm_{period}'] = np.nan
                else:
                    logger.warning(f"No ATR periods configured. Skipping normalized S/R distance for period {period}.")
                    df_derived[f'dist_to_support_norm_{period}'] = np.nan
                    df_derived[f'dist_to_resistance_norm_{period}'] = np.nan
            else:
                logger.warning(f"Base S/R features 'support_{period}' or 'resistance_{period}' not found. Skipping S/R distance calculations for this period.")
                # Ensure columns exist with NaNs if not calculated
                df_derived[f'dist_to_support_{period}'] = np.nan
                df_derived[f'dist_to_resistance_{period}'] = np.nan
                df_derived[f'dist_to_support_norm_{period}'] = np.nan
                df_derived[f'dist_to_resistance_norm_{period}'] = np.nan

        logger.debug("Derived features added.")
        return df_derived


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
                logger.warning(f"Column '{col_name}' not found for lagging features. Skipping.")
        logger.debug("Lagged features added.")
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
                logger.warning(f"Column '{col_name}' not found for differencing. Skipping.")
        logger.debug("Differenced features added.")
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
            logger.error("Cannot perform temporal safety validation: 'close' column is missing.")
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
                logger.debug(f"Skipping temporal validation for {col}: Insufficient valid data points ({len(x_valid)}) or near-zero standard deviation after filtering NaNs.")
                continue

            try:
                corr = abs(x_valid.corr(y_valid))
                if pd.isna(corr):
                     logger.debug(f"Correlation is NaN for {col}, likely due to insufficient variability after filtering NaNs.")
                     continue
                if corr > error_threshold:
                    logger.error(f"Temporal safety violation: Feature '{col}' correlation with next close change ({corr:.4f}) exceeds error threshold ({error_threshold}).")
                    violating_features.append(col)
                elif corr > warn_threshold:
                    logger.warning(f"Temporal safety warning: Feature '{col}' correlation with next close change ({corr:.4f}) exceeds warning threshold ({warn_threshold}).")
            except Exception as e:
                logger.warning(f"Could not compute correlation for feature '{col}': {e}")

        logger.info(f"Temporal safety validation complete. {len(violating_features)} features violated the error threshold.")
        return violating_features


    def process(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Processes raw OHLCV data to generate general-purpose technical and
        statistical features for configured periods.
        Initial rows affected by lookback periods will contain NaN values.
        Performs temporal safety checks if enabled.
        """
        logger.info("Starting general feature engineering process.")
        self._validate_dataframe(df)
        
        # Ensure that we are working with a fresh copy to avoid side effects
        df_processed = df.copy()

        # Generate base features that use shifted data to ensure temporal safety
        # These methods are designed to produce features for the current bar based on past information
        df_price_transforms = self._add_price_transformations(df)
        df_momentum = self._add_momentum_indicators(df)
        df_trend = self._add_trend_indicators(df)
        df_volatility = self._add_volatility_indicators(df)
        df_volume = self._add_volume_indicators(df)
        df_stats = self._add_statistical_features(df)
        # Candlestick patterns and FVG are based on the current completed bar, not future.
        df_custom_patterns = self._add_custom_pattern_features(df) 
        df_pivots = self._add_pivot_point_features(df)
        df_sr = self._add_support_resistance_features(df) # Add Support/Resistance before derived features that might use it

        # Concatenate all base features. The NaNs due to shifting are expected here.
        df_processed = pd.concat([
            df_processed,
            df_price_transforms,
            df_momentum,
            df_trend,
            df_volatility,
            df_volume,
            df_stats,
            df_custom_patterns,
            df_pivots,
            df_sr
        ], axis=1)

        # Derived features can now use the columns generated above.
        # Ensure they also handle temporal safety by deriving from already temporally safe features.
        df_derived = self._add_all_derived_features(df_processed) # Pass df_processed which contains all prior features
        df_processed = pd.concat([df_processed, df_derived], axis=1)

        # Breakout features also use already generated features
        df_breaks = self._add_breakout_features(df_processed) # Pass df_processed
        df_processed = pd.concat([df_processed, df_breaks], axis=1)

        # Lagged and Differenced features are applied to the *already generated* features
        # They will introduce further NaNs at the beginning based on their shift periods.
        df_lagged = self._add_lagged_features(df_processed)
        df_processed = pd.concat([df_processed, df_lagged], axis=1)

        df_differenced = self._add_differenced_features(df_processed)
        df_processed = pd.concat([df_processed, df_differenced], axis=1)

        df_with_nan = df_processed.copy()

        # Convert appropriate columns to nullable integer types (for binary/categorical features)
        categorical_cols = ['fvg', 'volatility_regime', 'pattern_cluster'] # pattern_cluster is now sum, can be int/float
        
        standard_pivot_binary_cols = []
        for p_col in ['pp', 'r1', 's1', 'r2', 's2', 'r3', 's3']:
            standard_pivot_binary_cols.append(f'is_above_{p_col}')
            standard_pivot_binary_cols.append(f'is_below_{p_col}')
        categorical_cols.extend([col for col in standard_pivot_binary_cols if col in df_with_nan.columns])

        swing_pivot_binary_cols = [
            'is_above_swing_high', 'is_below_swing_low',
            'is_support_broken_strong_vol', 'is_resistance_broken_strong_vol',
            'is_bull_wick_at_resistance', 'is_bear_wick_at_support'
        ]
        categorical_cols.extend([col for col in swing_pivot_binary_cols if col in df_with_nan.columns])

        for col in categorical_cols:
             if col in df_with_nan.columns:
                  # Ensure conversion only for columns that are actually binary (0, 1, -1)
                  # and are not already float-based sums (like pattern_cluster)
                  if df_with_nan[col].dropna().isin([0, 1, -1]).all(): # Check if values are binary/ternary
                    df_with_nan.loc[:, col] = df_with_nan[col].astype(pd.Int8Dtype())
                  else:
                    logger.debug(f"Column '{col}' contains values outside of [0, 1, -1] or NaNs. Not casting to Int8Dtype.")
             else:
                  logger.debug(f"Categorical column '{col}' not found in DataFrame to cast type.")

        logger.info(f"General feature engineering complete. DataFrame shape (including NaNs): {df_with_nan.shape}")

        # Conditionally perform temporal safety validation
        if self.config.temporal_validation.enabled:
            logger.info("Performing temporal safety validation on general features...")
            violating_features = self._validate_temporal_safety(df_with_nan)
            if violating_features:
                error_msg = f"Temporal safety violations detected in general features: {', '.join(violating_features)}"
                logger.error(error_msg)
                raise TemporalSafetyError(error_msg, features=violating_features)
            else:
                logger.info("Temporal safety validation passed for general features.")
        else:
            logger.info("Temporal safety validation skipped as per configuration.")

        return df_with_nan

