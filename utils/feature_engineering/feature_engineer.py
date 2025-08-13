# utils/feature_engineering/feature_engineer.py

import sys
import pandas as pd
import talib
import numpy as np
import logging
from pathlib import Path
from typing import Optional, Dict, Any, List

# Add project root to Python path for imports
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.append(str(PROJECT_ROOT))

# Import configuration and custom exception
try:
    # Import DEFAULT_FEATURE_CONFIG and schema from config/feature_config_schema
    from config.feature_config_schema import FeatureConfig, TemporalValidationConfig, DEFAULT_FEATURE_CONFIG
    
    # Import TechnicalIndicatorCalculator from its new location
    from utils.feature_engineering.technical_indicator_calculator import TechnicalIndicatorCalculator
    
    # Assuming TemporalSafetyError is defined in a custom exceptions.py file
    # This path remains the same as utils is still a parent directory
    from utils.exceptions import TemporalSafetyError
except ImportError as e:
    logging.error(f"Failed to import necessary modules: {e}")
    raise

# Set up logger for this module
logger = logging.getLogger(__name__)

# Define FLOAT_EPSILON for robust floating-point comparisons
FLOAT_EPSILON = 1e-9

class FeatureEngineer:
    """
    Engineers technical and statistical features from OHLCV data.
    Focuses on generating general-purpose features independent of specific
    trading strategy execution logic or labeling methods.
    Calculates indicators for lists of periods as defined in FEATURE_CONFIG.
    Includes temporal safety checks to prevent lookahead bias.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initializes the FeatureEngineer with a configuration dictionary.
        The configuration is validated against the FeatureConfig schema.

        Args:
            config (Optional[Dict[str, Any]]): Configuration for feature parameters.
                                               If None, defaults to config.feature_config_schema.DEFAULT_FEATURE_CONFIG.
        """
        if config is None:
            self.config: FeatureConfig = FeatureConfig(**DEFAULT_FEATURE_CONFIG.copy())
        else:
            # Ensure temporal_validation is correctly instantiated if it's a dict in the custom config
            if isinstance(config.get('temporal_validation'), dict):
                config['temporal_validation'] = TemporalValidationConfig(**config['temporal_validation'])
            self.config: FeatureConfig = FeatureConfig(**config)

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
        calculated_lookback = max_period_size + 2

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
        shifted_close = df['close'].shift(1)

        df_transformed['log_returns'] = np.log(shifted_close / df['close'].shift(2))
        df_transformed['typical_price'] = (df['high'].shift(1) + df['low'].shift(1) + shifted_close) / 3

        logger.debug("Price transformations added.")
        return df_transformed


    def _add_momentum_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Adds various momentum indicators (RSI, Stochastic, Awesome Oscillator, CCI, MFI).
        Calculations are based on past data for temporal safety.
        """
        df_momentum = pd.DataFrame(index=df.index)

        df_shifted = df[['open', 'high', 'low', 'close', 'volume']].shift(1)
        shifted_high = df_shifted['high']
        shifted_low = df_shifted['low']
        shifted_close = df_shifted['close']
        shifted_volume = df_shifted['volume']

        for period in self.config.rsi_periods:
             df_momentum[f'rsi_{period}'] = TechnicalIndicatorCalculator.calculate_rsi(shifted_close, period)

        stoch_d_period = 3
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
             logger.warning(f"AO periods not correctly configured as a pair: {self.config.ao_periods}. Skipping AO feature.")
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

        df_shifted = df[['high', 'low', 'close']].shift(1)
        shifted_high = df_shifted['high']
        shifted_low = df_shifted['low']
        shifted_close = df_shifted['close']

        for period in self.config.atr_periods:
             if len(df) >= period:
                  df_volatility[f'atr_{period}'] = TechnicalIndicatorCalculator.calculate_atr(
                      high_prices=shifted_high,
                      low_prices=shifted_low,
                      close_prices=shifted_close,
                      window=period
                  )
             else:
                  logger.warning(f"DataFrame too short ({len(df)} bars) for ATR period {period}. Filling 'atr_{period}' with NaN.")
                  df_volatility[f'atr_{period}'] = np.nan

        if self.config.atr_periods:
            for period in self.config.atr_periods:
                atr_col_name = f'atr_{period}'
                if atr_col_name in df_volatility.columns:
                    num_nans = df_volatility[atr_col_name].isnull().sum()
                    num_zeros = (df_volatility[atr_col_name] == 0).sum()
                    logger.debug(f"ATR column '{atr_col_name}' status: {num_nans} NaNs, {num_zeros} zeros out of {len(df_volatility)} rows.")
                else:
                    logger.debug(f"ATR column '{atr_col_name}' was not created.")

        for period in self.config.bollinger_periods:
            bb_results = TechnicalIndicatorCalculator.calculate_bollinger_bands(shifted_close, period)
            df_volatility[f'bb_upper_{period}'] = bb_results['hband']
            df_volatility[f'bb_lower_{period}'] = bb_results['lband']
            df_volatility[f'bb_width_{period}'] = bb_results['wband']

        logger.debug("Volatility indicators added.")
        return df_volatility


    def _add_volume_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Adds various volume indicators (OBV, CMF, Volume Oscillator).
        Calculations are based on past data for temporal safety.
        """
        df_volume = pd.DataFrame(index=df.index)

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
            resampled_df = df.resample('D').agg({'high': 'max', 'low': 'min'})
            daily_adr = TechnicalIndicatorCalculator.calculate_adr(
                high_prices=resampled_df['high'], low_prices=resampled_df['low'], window=period
            ).shift(1)
            
            df_stats[f'adr_{period}'] = daily_adr.reindex(df.index, method='ffill')

        logger.debug("Statistical features added.")
        return df_stats


    def _add_custom_pattern_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Adds Fair Value Gap (FVG) and Candlestick Pattern detection features.
        These are calculated based on current bar data.
        """
        df_patterns_fvg = pd.DataFrame(index=df.index)

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
                df_patterns_fvg[f'pattern_{pattern}_signal'] = pattern_values.apply(
                    lambda x: 1 if x > 0 else (-1 if x < 0 else 0)
                )
            else:
                logger.warning(f"Configured pattern '{pattern}' is not supported by the current implementation.")

        bullish_fvg_at_t = (df['low'] > df['high'].shift(self.config.fvg_lookback_bars))
        bearish_fvg_at_t = (df['high'] < df['low'].shift(self.config.fvg_lookback_bars))

        df_patterns_fvg['fvg'] = 0
        df_patterns_fvg.loc[bullish_fvg_at_t, 'fvg'] = 1
        df_patterns_fvg.loc[bearish_fvg_at_t, 'fvg'] = -1

        logger.debug("Custom pattern features (candlestick and FVG) added.")
        return df_patterns_fvg


    def _add_pivot_point_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Adds Standard Pivot Points and Pine Script-style Swing Pivots.
        """
        df_pivots = pd.DataFrame(index=df.index)

        rule = None
        if self.config.pivot_point_calculation_period == 'daily':
            rule = 'D'
        elif self.config.pivot_point_calculation_period == 'weekly':
            rule = 'W'
        elif self.config.pivot_point_calculation_period == 'monthly':
            rule = 'M'
        else:
            logger.error(f"Unsupported pivot_point_calculation_period: {self.config.pivot_point_calculation_period}")

        if rule:
            agg_ohlc = df[['open', 'high', 'low', 'close']].resample(rule).agg({
                'open': 'first',
                'high': 'max',
                'low': 'min',
                'close': 'last'
            })
            prev_period_ohlc = agg_ohlc.shift(1)
            valid_periods = prev_period_ohlc.dropna()

            if not valid_periods.empty:
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
                df_pivots_standard = temp_pivots.reindex(df.index, method='ffill')
                df_pivots_standard.dropna(inplace=True)
                df_pivots = pd.concat([df_pivots, df_pivots_standard], axis=1)
                logger.debug(f"Standard Pivot points calculated using {self.config.pivot_point_calculation_period} aggregation and {self.config.pivot_point_method} method.")
            else:
                logger.warning(f"No valid previous period data found for {self.config.pivot_point_calculation_period} pivot point calculation. Skipping standard pivot features.")
                for col in ['pp', 'r1', 's1', 'r2', 's2', 'r3', 's3']:
                    df_pivots[col] = np.nan

        n = len(df)
        swing_highs = pd.Series(np.nan, index=df.index)
        swing_lows = pd.Series(np.nan, index=df.index)

        left_bars = self.config.swing_pivot_left_bars
        right_bars = self.config.swing_pivot_right_bars

        for i in range(n):
            if i >= left_bars and i + right_bars < n:
                window_highs = df['high'].iloc[i - left_bars : i + right_bars + 1]
                if df['high'].iloc[i] == window_highs.max():
                    is_pivot_high = True
                    for j in range(1, left_bars + 1):
                        if df['high'].iloc[i - j] >= df['high'].iloc[i]:
                            is_pivot_high = False
                            break
                    if is_pivot_high:
                        for j in range(1, right_bars + 1):
                            if df['high'].iloc[i + j] >= df['high'].iloc[i]:
                                is_pivot_high = False
                                break
                    if is_pivot_high:
                        swing_highs.iloc[i] = df['high'].iloc[i]

                window_lows = df['low'].iloc[i - left_bars : i + right_bars + 1]
                if df['low'].iloc[i] == window_lows.min():
                    is_pivot_low = True
                    for j in range(1, left_bars + 1):
                        if df['low'].iloc[i - j] <= df['low'].iloc[i]:
                            is_pivot_low = False
                            break
                    if is_pivot_low:
                        for j in range(1, right_bars + 1):
                            if df['low'].iloc[i + j] <= df['low'].iloc[i]:
                                is_pivot_low = False
                                break
                    if is_pivot_low:
                        swing_lows.iloc[i] = df['low'].iloc[i]

        df_swing_pivots = pd.DataFrame(index=df.index)
        df_swing_pivots['swing_high_pivot'] = swing_highs.ffill().shift(right_bars + 1)
        df_swing_pivots['swing_low_pivot'] = swing_lows.ffill().shift(right_bars + 1)
        df_pivots = pd.concat([df_pivots, df_swing_pivots], axis=1)

        logger.debug("Pivot point features added.")
        return df_pivots


    def _add_support_resistance_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculates simple support and resistance levels based on rolling highest high and lowest low.
        These are calculated based on past data for temporal safety.
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
        """
        df_breaks = pd.DataFrame(index=df.index)

        required_cols = ['swing_high_pivot', 'swing_low_pivot', 'volume_osc', 'open', 'high', 'low', 'close']
        if not all(col in df.columns for col in required_cols):
            logger.warning("Missing required columns for breakout features. Skipping breakout detection.")
            for col in ['is_support_broken_strong_vol', 'is_resistance_broken_strong_vol',
                        'is_bull_wick_at_resistance', 'is_bear_wick_at_support']:
                df_breaks[col] = 0
            return df_breaks

        close_prev = df['close'].shift(1)
        open_prev = df['open'].shift(1)
        high_prev = df['high'].shift(1)
        low_prev = df['low'].shift(1)

        high_pivot = df['swing_high_pivot']
        low_pivot = df['swing_low_pivot']
        volume_osc = df['volume_osc']

        is_resistance_crossover = (close_prev <= high_pivot) & (df['close'] > high_pivot)
        is_strong_bullish_body = (df['close'] > df['open']) & ((df['close'] - df['open']) / (df['high'] - df['low'] + FLOAT_EPSILON) > 0.6)

        df_breaks['is_resistance_broken_strong_vol'] = (
            is_resistance_crossover &
            is_strong_bullish_body &
            (volume_osc > self.config.volume_threshold)
        ).astype(int)

        is_support_crossunder = (close_prev >= low_pivot) & (df['close'] < low_pivot)
        is_strong_bearish_body = (df['close'] < df['open']) & ((df['open'] - df['close']) / (df['high'] - df['low'] + FLOAT_EPSILON) > 0.6)

        df_breaks['is_support_broken_strong_vol'] = (
            is_support_crossunder &
            is_strong_bearish_body &
            (volume_osc > self.config.volume_threshold)
        ).astype(int)

        is_bull_wick_condition = (df['open'] - df['low']) > (df['close'] - df['open'])
        df_breaks['is_bull_wick_at_resistance'] = (
            is_resistance_crossover &
            is_bull_wick_condition
        ).astype(int)

        is_bear_wick_condition = (df['open'] - df['close']) < (df['high'] - df['open'])
        df_breaks['is_bear_wick_at_support'] = (
            is_support_crossunder &
            is_bear_wick_condition
        ).astype(int)

        logger.debug("Breakout and wick features added.")
        return df_breaks


    def _add_all_derived_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Creates all secondary and tertiary features from base indicators and other features.
        Ensures temporal safety for derived features.
        """
        df_derived = pd.DataFrame(index=df.index)

        # Trend Strength
        if len(self.config.trend_strength_periods) == 2:
            short_period, long_period = self.config.trend_strength_periods
            short_sma_col = f'sma_{short_period}'
            long_sma_col = f'sma_{long_period}'

            if short_sma_col not in df.columns or long_sma_col not in df.columns:
                 logger.error(f"Base SMA features '{short_sma_col}' or '{long_sma_col}' missing for trend strength calculation.")
                 df_derived['trend_strength'] = np.nan
            else:
                long_sma_t = df[long_sma_col].replace(0, np.nan)
                df_derived['trend_strength'] = (df[short_sma_col] - long_sma_t) / long_sma_t
        else:
            logger.warning(f"Trend strength periods not correctly configured as a pair: {self.config.trend_strength_periods}. Skipping trend_strength feature.")
            df_derived['trend_strength'] = np.nan

        # Volatility Regime (categorical)
        if self.config.atr_periods:
            atr_period_for_regime = min(self.config.atr_periods)
            atr_col_name_for_regime = f'atr_{atr_period_for_regime}'

            if atr_col_name_for_regime not in df.columns:
                 logger.error(f"'{atr_col_name_for_regime}' feature missing for volatility regime calculation.")
                 df_derived['volatility_regime'] = pd.NA
            else:
                atr_t = df[atr_col_name_for_regime]
                atr_t_dropna = atr_t.dropna().copy()
                if atr_t_dropna.shape[0] >= 3 and len(atr_t_dropna.unique()) >= 2:
                    try:
                        df_derived.loc[atr_t_dropna.index, 'volatility_regime'] = pd.qcut(
                            atr_t_dropna,
                            q=3,
                            labels=False,
                            duplicates='drop'
                        ).astype(pd.Int8Dtype())
                    except Exception as e:
                        logger.warning(f"Could not compute volatility regime: {e}. Filling with NaN.", exc_info=True)
                        df_derived['volatility_regime'] = pd.NA
                else:
                     logger.warning("Insufficient data points or unique ATR values to compute volatility regime. Filling with NaN.")
                     df_derived['volatility_regime'] = pd.NA
        else:
            logger.warning("No ATR periods configured. Cannot compute volatility regime. Filling with NaN.")
            df_derived['volatility_regime'] = pd.NA

        # Pattern Clustering
        pattern_cols = [f"pattern_{p}_signal" for p in self.config.candlestick_patterns]
        existing_pattern_cols = [col for col in pattern_cols if col in df.columns]

        if not existing_pattern_cols:
             logger.warning("No configured pattern signal columns found in DataFrame for pattern clustering. Skipping feature.")
             df_derived['pattern_cluster'] = np.nan
        else:
            df_derived['pattern_cluster'] = df[existing_pattern_cols].sum(axis=1)

        # Relative distance to Standard Pivot Points (normalized by ATR)
        if self.config.pivot_point_method == 'standard' and self.config.atr_periods:
            atr_period_for_norm = min(self.config.atr_periods)
            atr_col = f'atr_{atr_period_for_norm}'

            if atr_col in df.columns and not df[atr_col].isnull().all() and (df[atr_col] > FLOAT_EPSILON).any():
                standard_pivot_cols = ['pp', 'r1', 's1', 'r2', 's2', 'r3', 's3']
                for p_col in standard_pivot_cols:
                    if p_col in df.columns:
                        df_derived[f'dist_to_{p_col}_norm'] = (df['close'] - df[p_col]) / df[atr_col]
                        df_derived[f'is_above_{p_col}'] = (df['close'] > df[p_col]).astype(int)
                        df_derived[f'is_below_{p_col}'] = (df['close'] < df[p_col]).astype(int)
                    else:
                        logger.debug(f"Standard pivot point column '{p_col}' not found for relative distance calculation.")
            else:
                logger.warning(f"ATR column '{atr_col}' not available or all NaN/zero for Standard Pivot Point normalization. Skipping normalized Standard Pivot Point distances.")
        else:
            logger.warning("Standard pivot points not configured or ATR not available for normalized pivot point distances. Skipping.")

        # Relative distance to Swing Pivots (normalized by ATR)
        if self.config.atr_periods:
            atr_period_for_norm = min(self.config.atr_periods)
            atr_col = f'atr_{atr_period_for_norm}'
            if atr_col in df.columns and not df[atr_col].isnull().all() and (df[atr_col] > FLOAT_EPSILON).any():
                if 'swing_high_pivot' in df.columns and 'swing_low_pivot' in df.columns:
                    df_derived['dist_to_swing_high_norm'] = (df['close'] - df['swing_high_pivot']) / df[atr_col]
                    df_derived['dist_to_swing_low_norm'] = (df['close'] - df['swing_low_pivot']) / df[atr_col]
                    df_derived['is_above_swing_high'] = (df['close'] > df['swing_high_pivot']).astype(int)
                    df_derived['is_below_swing_low'] = (df['close'] < df['swing_low_pivot']).astype(int)
                else:
                    logger.warning("Swing pivot columns not found for relative distance calculation. Skipping.")
            else:
                logger.warning(f"ATR column '{atr_col}' not available or all NaN/zero for Swing Pivot normalization. Skipping normalized Swing Pivot distances.")
        else:
            logger.warning("ATR not available for normalized swing pivot distances. Skipping.")

        # Support/Resistance Levels Normalized (This section expects base S/R levels to already exist)
        for period in self.config.support_resistance_periods:
            # Check for the existence of base S/R features before calculating distances
            if f'support_{period}' in df.columns and f'resistance_{period}' in df.columns:
                df_derived[f'dist_to_support_{period}'] = df['close'].shift(1) - df[f'support_{period}']
                df_derived[f'dist_to_resistance_{period}'] = df[f'resistance_{period}'] - df['close'].shift(1)

                num_nans_dist_sup = df_derived[f'dist_to_support_{period}'].isnull().sum()
                num_nans_dist_res = df_derived[f'dist_to_resistance_{period}'].isnull().sum()
                logger.debug(f"S/R Distance (Period {period}) status: dist_to_support_{period} has {num_nans_dist_sup} NaNs. dist_to_resistance_{period} has {num_nans_dist_res} NaNs.")
                if num_nans_dist_sup < len(df_derived):
                    logger.debug(f"Sample of dist_to_support_{period}: {df_derived[f'dist_to_support_{period}'].dropna().head().tolist()}")
                if num_nans_dist_res < len(df_derived):
                    logger.debug(f"Sample of dist_to_resistance_{period}: {df_derived[f'dist_to_resistance_{period}'].dropna().head().tolist()}")

                if self.config.atr_periods:
                    atr_period_for_norm = min(self.config.atr_periods)
                    atr_col = f'atr_{atr_period_for_norm}'

                    if atr_col in df.columns and not df[atr_col].isnull().all() and (df[atr_col] > FLOAT_EPSILON).any():
                        combined_valid_mask_sup = (df_derived[f'dist_to_support_{period}'].notna()) & \
                                                  (df[atr_col].notna()) & \
                                                  (df[atr_col] > FLOAT_EPSILON)

                        combined_valid_mask_res = (df_derived[f'dist_to_resistance_{period}'].notna()) & \
                                                  (df[atr_col].notna()) & \
                                                  (df[atr_col] > FLOAT_EPSILON)

                        logger.debug(f"Combined normalization mask for support (Period {period}) has {combined_valid_mask_sup.sum()} True values.")
                        logger.debug(f"Combined normalization mask for resistance (Period {period}) has {combined_valid_mask_res.sum()} True values.")

                        df_derived.loc[combined_valid_mask_sup, f'dist_to_support_norm_{period}'] = \
                            df_derived.loc[combined_valid_mask_sup, f'dist_to_support_{period}'] / df.loc[combined_valid_mask_sup, atr_col]

                        df_derived.loc[combined_valid_mask_res, f'dist_to_resistance_norm_{period}'] = \
                            df_derived.loc[combined_valid_mask_res, f'dist_to_resistance_{period}'] / df.loc[combined_valid_mask_res, atr_col]
                    else:
                        logger.warning(f"ATR column '{atr_col}' not available or all NaN/zero for S/R normalization for period {period}. Skipping normalized S/R distance.")
                        df_derived[f'dist_to_support_norm_{period}'] = np.nan
                        df_derived[f'dist_to_resistance_norm_{period}'] = np.nan
                else:
                    logger.warning(f"No ATR periods configured. Skipping normalized S/R distance for period {period}.")
                    df_derived[f'dist_to_support_norm_{period}'] = np.nan
                    df_derived[f'dist_to_resistance_norm_{period}'] = np.nan
            else:
                logger.warning(f"Base S/R features 'support_{period}' or 'resistance_{period}' not found. Skipping S/R distance calculations for this period.")
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

        next_close_change = df['close'].pct_change().shift(-1)
        violating_features = []
        ohlcv_cols = {'open', 'high', 'low', 'close', 'volume'}

        cols_to_skip_correlation = [
            col for col in ['fvg', 'volatility_regime', 'pp', 'r1', 's1', 'r2', 's2', 'r3', 's3',
                            'swing_high_pivot', 'swing_low_pivot',
                            'is_above_pp', 'is_below_pp', 'is_above_r1', 'is_below_r1',
                            'is_above_s1', 'is_below_s1', 'is_above_r2', 'is_below_r2',
                            'is_above_s2', 'is_below_s2', 'is_above_r3', 'is_below_r3',
                            'is_above_s3',
                            'is_above_swing_high', 'is_below_swing_low',
                            'is_support_broken_strong_vol', 'is_resistance_broken_strong_vol',
                            'is_bull_wick_at_resistance', 'is_bear_wick_at_support'
                           ]
            if col in df.columns and not pd.api.types.is_numeric_dtype(df[col])
        ]

        feature_cols = [col for col in df.columns if col not in ohlcv_cols and col != next_close_change.name and col not in cols_to_skip_correlation]

        for col in feature_cols:
            if not pd.api.types.is_numeric_dtype(df[col]):
                logger.debug(f"Skipping temporal safety correlation check for non-numeric column '{col}'.")
                continue

            x = df[col]
            y = next_close_change
            valid = pd.notna(x) & pd.notna(y)
            x_valid = x[valid]
            y_valid = y[valid]

            if len(x_valid) < 2 or x_valid.std() < 1e-9 or y_valid.std() < 1e-9:
                logger.debug(f"Skipping temporal validation for {col}: Insufficient valid data points ({len(x_valid)}) or near-zero standard deviation.")
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
        df_processed = df.copy()

        df_price_transforms = self._add_price_transformations(df)
        df_processed = pd.concat([df_processed, df_price_transforms], axis=1)

        df_momentum = self._add_momentum_indicators(df)
        df_processed = pd.concat([df_processed, df_momentum], axis=1)

        df_trend = self._add_trend_indicators(df)
        df_processed = pd.concat([df_processed, df_trend], axis=1)

        df_volatility = self._add_volatility_indicators(df)
        df_processed = pd.concat([df_processed, df_volatility], axis=1)

        df_volume = self._add_volume_indicators(df)
        df_processed = pd.concat([df_processed, df_volume], axis=1)

        df_stats = self._add_statistical_features(df)
        df_processed = pd.concat([df_processed, df_stats], axis=1)

        df_custom_patterns = self._add_custom_pattern_features(df)
        df_processed = pd.concat([df_processed, df_custom_patterns], axis=1)

        df_pivots = self._add_pivot_point_features(df)
        df_processed = pd.concat([df_processed, df_pivots], axis=1)

        # NEW: Add Support/Resistance features before derived features
        df_sr = self._add_support_resistance_features(df)
        df_processed = pd.concat([df_processed, df_sr], axis=1)


        df_derived = self._add_all_derived_features(df_processed)
        df_processed = pd.concat([df_processed, df_derived], axis=1)

        df_breaks = self._add_breakout_features(df_processed)
        df_processed = pd.concat([df_processed, df_breaks], axis=1)

        df_lagged = self._add_lagged_features(df_processed)
        df_processed = pd.concat([df_processed, df_lagged], axis=1)

        df_differenced = self._add_differenced_features(df_processed)
        df_processed = pd.concat([df_processed, df_differenced], axis=1)

        df_with_nan = df_processed.copy()

        categorical_cols = ['fvg', 'volatility_regime']
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
                  df_with_nan.loc[:, col] = df_with_nan[col].astype(pd.Int8Dtype())
             else:
                  logger.debug(f"Categorical column '{col}' not found in DataFrame to cast type.")

        logger.info(f"General feature engineering complete. DataFrame shape (including NaNs): {df_with_nan.shape}")

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
