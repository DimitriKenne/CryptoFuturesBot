# utils/feature_engineering/price_action_feature_processor.py

import pandas as pd
import numpy as np
import logging
import talib # Assume talib is imported and available if self.config.talib_available is True
from typing import Dict, Any, List, Optional

# --- Import configuration ---
try:
    from config.feature import FeatureConfig
    from config.params import FLOAT_EPSILON
    from utils.feature_engineering.technical_indicator_calculator import TechnicalIndicatorCalculator
except ImportError as e:
    logging.critical(f"Failed to import necessary modules: {e}")
    raise

logger = logging.getLogger(__name__)

# Runtime check for TA-Lib availability
talib_available = False
try:
    import talib as _talib
    talib = _talib
    talib_available = True
except ImportError:
    logger.warning("TA-Lib not available. Candlestick patterns may be unavailable.")
    talib = None # Explicitly set to None


class PriceActionFeatureProcessor:
    """
    Calculates features related to price action, such as candlestick patterns,
    Fair Value Gaps (FVG), Pivot Points, Support/Resistance levels, and Breakouts.
    """

    def __init__(self, config: FeatureConfig):
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.info("PriceActionFeatureProcessor initialized.")

    def add_custom_pattern_features(self, df: pd.DataFrame) -> pd.DataFrame:
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

        if talib_available:
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
                    self.logger.warning(f"Configured pattern '{pattern}' is not supported by the current TA-Lib implementation or not recognized. Skipping.")
            self.logger.debug("Candlestick patterns added.")
        else:
            self.logger.warning("TA-Lib not available. Skipping candlestick pattern detection.")
            # Ensure columns exist with NaNs if skipped
            for pattern in self.config.candlestick_patterns:
                df_patterns_fvg[f'pattern_{pattern}_signal'] = np.nan


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
            self.logger.warning(f"FVG lookback bars ({self.config.fvg_lookback_bars}) is too small. Skipping FVG feature.")
            df_patterns_fvg['fvg'] = 0 # Ensure column exists even if skipped

        self.logger.debug("Custom pattern features (FVG) added.")
        return df_patterns_fvg


    def add_pivot_point_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Adds Standard Pivot Points and Pine Script-style Swing Pivots.
        Standard Pivots are based on previous period's OHLC (e.g., previous day).
        Swing Pivots are defined by local highs/lows over a specified window.
        """
        df_pivots = pd.DataFrame(index=df.index)

        # --- Standard Pivot Points (PP, R1/S1, R2/S2, R3/S3) ---
        resample_rule = None
        if self.config.pivot_point_calculation_period == 'daily':
            resample_rule = 'D'
        elif self.config.pivot_point_calculation_period == 'weekly':
            resample_rule = 'W'
        elif self.config.pivot_point_calculation_period == 'monthly':
            resample_rule = 'M'
        else:
            self.logger.error(f"Unsupported pivot_point_calculation_period: {self.config.pivot_point_calculation_period}. Skipping standard pivot features.")

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
                self.logger.debug(f"Standard Pivot points calculated using {self.config.pivot_point_calculation_period} aggregation and {self.config.pivot_point_method} method.")
            else:
                self.logger.warning(f"No valid previous period data found for {self.config.pivot_point_calculation_period} pivot point calculation. Skipping standard pivot features.")
                # Ensure columns exist even if empty
                for col in ['pp', 'r1', 's1', 'r2', 's2', 'r3', 's3']:
                    df_pivots[col] = np.nan

        # --- Swing Pivots (Pine Script style local high/low) ---
        left_bars = self.config.swing_pivot_left_bars
        right_bars = self.config.swing_pivot_right_bars
        
        # Calculate on shifted data to ensure temporal safety
        df_shifted_for_swing = df[['high', 'low']].shift(1)
        
        n = len(df_shifted_for_swing)
        swing_highs_temp = pd.Series(np.nan, index=df_shifted_for_swing.index)
        swing_lows_temp = pd.Series(np.nan, index=df_shifted_for_swing.index)

        for i in range(n):
            if i >= left_bars and i + right_bars < n:
                window_highs = df_shifted_for_swing['high'].iloc[i - left_bars : i + right_bars + 1]
                if df_shifted_for_swing['high'].iloc[i] == window_highs.max():
                    swing_highs_temp.iloc[i] = df_shifted_for_swing['high'].iloc[i]

                window_lows = df_shifted_for_swing['low'].iloc[i - left_bars : i + right_bars + 1]
                if df_shifted_for_swing['low'].iloc[i] == window_lows.min():
                    swing_lows_temp.iloc[i] = df_shifted_for_swing['low'].iloc[i]

        df_swing_pivots_temp = pd.DataFrame(index=df.index)
        # Forward fill and then shift by `right_bars + 1` to ensure we are using *past* swing pivots.
        df_swing_pivots_temp['swing_high_pivot'] = swing_highs_temp.ffill().shift(right_bars + 1)
        df_swing_pivots_temp['swing_low_pivot'] = swing_lows_temp.ffill().shift(right_bars + 1)
        
        df_pivots = pd.concat([df_pivots, df_swing_pivots_temp], axis=1)

        self.logger.debug("Pivot point features added.")
        return df_pivots


    def add_support_resistance_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculates simple support and resistance levels based on rolling highest high and lowest low.
        These are calculated based on *past* data for temporal safety.
        """
        df_sr = pd.DataFrame(index=df.index)
        
        # Shifted high/low for temporal safety
        shifted_high = df['high'].shift(1)
        shifted_low = df['low'].shift(1)

        for period in self.config.support_resistance_periods:
            df_sr[f'resistance_{period}'] = shifted_high.rolling(window=period, min_periods=1).max()
            df_sr[f'support_{period}'] = shifted_low.rolling(window=period, min_periods=1).min()
        
        self.logger.debug("Support and Resistance features added.")
        return df_sr


    def add_breakout_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Detects support/resistance breaks and wick patterns based on Pine Script logic.
        Requires 'swing_high_pivot', 'swing_low_pivot', 'volume_osc' (from IndicatorFeatureProcessor)
        and basic OHLC data to be present in the input DataFrame.
        """
        df_breaks = pd.DataFrame(index=df.index)

        # Ensure necessary columns are available. These columns are expected to be generated
        # by previous steps and present in the combined 'df' passed to this method.
        # volume_osc is generated by IndicatorFeatureProcessor, so it must be passed in `df`.
        required_cols = ['swing_high_pivot', 'swing_low_pivot', 'volume_osc', 'open', 'high', 'low', 'close']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            self.logger.warning(f"Missing required columns for breakout features: {missing_cols}. Skipping breakout detection.")
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
        is_resistance_crossover = (df['close'].shift(1) <= high_pivot) & (df['close'] > high_pivot)
        is_strong_bullish_body = (current_close > current_open) & ((current_close - current_open) / (current_high - current_low + FLOAT_EPSILON) > 0.6)

        df_breaks['is_resistance_broken_strong_vol'] = (
            is_resistance_crossover &
            is_strong_bullish_body &
            (volume_osc > self.config.volume_threshold)
        ).astype(int)

        # --- Support Breakout with Strong Volume ---
        is_support_crossunder = (df['close'].shift(1) >= low_pivot) & (df['close'] < low_pivot)
        is_strong_bearish_body = (current_close < current_open) & ((current_open - current_close) / (current_high - current_low + FLOAT_EPSILON) > 0.6)

        df_breaks['is_support_broken_strong_vol'] = (
            is_support_crossunder &
            is_strong_bearish_body &
            (volume_osc > self.config.volume_threshold)
        ).astype(int)

        # --- Bullish Wick at Resistance (e.g., rejection from resistance) ---
        is_long_upper_wick = (current_high - current_close) > (current_close - current_low)
        df_breaks['is_bull_wick_at_resistance'] = (
            (current_high > high_pivot) &
            (current_close < high_pivot) &
            is_long_upper_wick
        ).astype(int)

        # --- Bearish Wick at Support (e.g., bounce from support) ---
        is_long_lower_wick = (current_close - current_low) > (current_high - current_close)
        df_breaks['is_bear_wick_at_support'] = (
            (current_low < low_pivot) &
            (current_close > low_pivot) &
            is_long_lower_wick
        ).astype(int)

        self.logger.debug("Breakout and wick features added.")
        return df_breaks
