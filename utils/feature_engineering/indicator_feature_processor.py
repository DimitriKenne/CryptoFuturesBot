# utils/feature_engineering/indicator_feature_processor.py

import pandas as pd
import numpy as np
import logging
from typing import Dict, Any, List, Optional, Union

# --- Import configuration ---
try:
    from config.feature import FeatureConfig
    from config.params import FLOAT_EPSILON, app_config # Need app_config to check global library flags
    from utils.feature_engineering.technical_indicator_calculator import TechnicalIndicatorCalculator
except ImportError as e:
    logging.critical(f"Failed to import necessary modules: {e}")
    raise

logger = logging.getLogger(__name__)

# Lazy imports for talib and ta.
# These will only be attempted if the respective flags in config.feature indicate availability.
# We don't import them at the top level here to avoid ImportError if they are truly missing.
talib = None
ta_lib_available = False # Runtime flag for 'ta' library
talib_available = False # Runtime flag for 'talib' library


class IndicatorFeatureProcessor:
    """
    Calculates various technical and statistical indicators (momentum, trend, volatility, volume)
    and their derived features. This class uses the TechnicalIndicatorCalculator for atomic indicator
    computations and manages the application of different periods.
    """

    def __init__(self, config: FeatureConfig):
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.info("IndicatorFeatureProcessor initialized.")

        # Pre-calculate column names for consistency for trend strength
        self.short_ema_col_name = f'ema_{self.config.trend_strength_periods[0]}' if len(self.config.trend_strength_periods) >= 2 else None
        self.long_ema_col_name = f'ema_{self.config.trend_strength_periods[1]}' if len(self.config.trend_strength_periods) >= 2 else None
        
        # Runtime check for TA-Lib and 'ta' library availability
        global talib, ta_lib_available, talib_available
        if self.config.talib_available:
            try:
                import talib as _talib
                talib = _talib
                talib_available = True
            except ImportError:
                self.logger.warning("TA-Lib indicated as available in config but could not be imported. Candlestick patterns (if used here) may be unavailable.")
                talib_available = False

        if self.config.ta_lib_available:
            try:
                # We don't need to import individual functions here as TechnicalIndicatorCalculator
                # already handles the imports from 'ta' library internally.
                # This just sets the flag.
                import ta # Just a check if the top-level package can be imported
                ta_lib_available = True
            except ImportError:
                self.logger.warning("'ta' library indicated as available in config but could not be imported. Some indicators will be unavailable.")
                ta_lib_available = False


    @property
    def required_lookback(self) -> int:
        """
        Calculates the maximum lookback required for all technical and statistical indicators
        and their derived features.
        """
        period_sizes = []

        list_period_keys = [
            'sma_periods', 'ema_periods', 'rsi_periods', 'bollinger_periods',
            'atr_periods', 'stochastic_periods', 'ao_periods', 'cci_periods',
            'mfi_periods', 'volume_periods', 'z_score_periods', 'adr_periods',
            'trend_strength_periods' # Max of these two periods
        ]
        for key in list_period_keys:
            values = getattr(self.config, key)
            if isinstance(values, list):
                for period in values:
                    if isinstance(period, int) and period > 0:
                        period_sizes.append(period)

        # Specific periods not in the lists
        period_sizes.append(self.config.volume_oscillator_long_ema)
        
        max_period_size = max(period_sizes) if period_sizes else 0
        
        # Add a buffer for calculations like ATR which might need more than just 'period' bars
        # or for shifted inputs (most of these indicators take shifted data).
        return max_period_size + 1 # +1 for the shift operation before calculation


    def _add_momentum_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Adds various momentum indicators (RSI, Stochastic, Awesome Oscillator, CCI, MFI).
        Calculations are based on past data for temporal safety.
        """
        df_momentum = pd.DataFrame(index=df.index)

        if not ta_lib_available:
            self.logger.warning("TA library not available. Skipping momentum indicator calculations.")
            return df_momentum # Return empty if library is missing

        # Shift input data for temporal safety (indicators calculated on previous bar's OHLCV)
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
             self.logger.warning(f"AO periods not correctly configured as a pair (expected 2, got {len(self.config.ao_periods)}): {self.config.ao_periods}. Skipping AO feature.")
             df_momentum['ao'] = np.nan # Ensure column exists

        for period in self.config.cci_periods:
            df_momentum[f'cci_{period}'] = TechnicalIndicatorCalculator.calculate_cci(
                high_prices=shifted_high, low_prices=shifted_low, close_prices=shifted_close, window=period
            )

        for period in self.config.mfi_periods:
             df_momentum[f'mfi_{period}'] = TechnicalIndicatorCalculator.calculate_mfi(
                 high_prices=shifted_high, low_prices=shifted_low, close_prices=shifted_close,
                 volume=shifted_volume, window=period
             )

        self.logger.debug("Momentum indicators added.")
        return df_momentum


    def _add_trend_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Adds various trend indicators (SMA, EMA, MACD).
        Calculations are based on past data for temporal safety.
        """
        df_trend = pd.DataFrame(index=df.index)

        if not ta_lib_available:
            self.logger.warning("TA library not available. Skipping trend indicator calculations.")
            return df_trend

        shifted_close = df['close'].shift(1)

        for period in self.config.sma_periods:
            df_trend[f'sma_{period}'] = TechnicalIndicatorCalculator.calculate_sma(shifted_close, period)

        for period in self.config.ema_periods:
            df_trend[f'ema_{period}'] = TechnicalIndicatorCalculator.calculate_ema(shifted_close, period)

        macd_results = TechnicalIndicatorCalculator.calculate_macd(shifted_close)
        df_trend['macd'] = macd_results['macd']
        df_trend['macd_signal'] = macd_results['macd_signal']
        df_trend['macd_diff'] = macd_results['macd_diff']

        self.logger.debug("Trend indicators added.")
        return df_trend


    def _add_volatility_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Adds various volatility indicators (ATR, Bollinger Bands).
        Calculations are based on past data for temporal safety.
        """
        df_volatility = pd.DataFrame(index=df.index)

        if not ta_lib_available:
            self.logger.warning("TA library not available. Skipping volatility indicator calculations.")
            return df_volatility

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
                 self.logger.debug(f"ATR column 'atr_{period}' status: {num_nans} NaNs, {num_zeros} zeros out of {len(df_volatility)} rows.")
             else:
                 self.logger.debug(f"ATR column 'atr_{period}' was not created.")

        for period in self.config.bollinger_periods:
            bb_results = TechnicalIndicatorCalculator.calculate_bollinger_bands(shifted_close, period)
            df_volatility[f'bb_upper_{period}'] = bb_results['hband']
            df_volatility[f'bb_lower_{period}'] = bb_results['lband']
            df_volatility[f'bb_width_{period}'] = bb_results['wband'] # Width is often used as a feature

        self.logger.debug("Volatility indicators added.")
        return df_volatility


    def _add_volume_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Adds various volume indicators (OBV, CMF, Volume Oscillator).
        Calculations are based on past data for temporal safety.
        """
        df_volume = pd.DataFrame(index=df.index)

        if not ta_lib_available:
            self.logger.warning("TA library not available. Skipping volume indicator calculations.")
            return df_volume

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
            volume=df['volume'].shift(1), # Note: uses raw volume shifted by 1
            short_ema_window=self.config.volume_oscillator_short_ema,
            long_ema_window=self.config.volume_oscillator_long_ema
        )

        self.logger.debug("Volume indicators added.")
        return df_volume


    def _add_statistical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Adds statistical features like Z-scores and Average Daily Range (ADR).
        """
        df_stats = pd.DataFrame(index=df.index)

        if not ta_lib_available:
            self.logger.warning("TA library not available. Skipping statistical feature calculations.")
            return df_stats

        shifted_close = df['close'].shift(1)

        for period in self.config.z_score_periods:
            df_stats[f'z_score_{period}'] = TechnicalIndicatorCalculator.calculate_z_score(shifted_close, period)

        for period in self.config.adr_periods:
            # Resample OHLCV data to daily frequency for ADR calculation, then shift for temporal safety
            # ADR is typically calculated on daily (or higher interval) ranges
            # Here we take the high/low of the *previous* day's aggregated data
            resampled_df = df.resample('D').agg({'high': 'max', 'low': 'min'})

            # --- CRITICAL FIX: Ensure resampled_df has data before shifting ---
            # Check if the columns exist and are not empty after resampling
            if 'high' in resampled_df.columns and not resampled_df['high'].empty and \
               'low' in resampled_df.columns and not resampled_df['low'].empty:
                
                # Check if the shifted series would result in a scalar (e.g., if only one row after shift)
                # If it's going to be a scalar, assign NaN to avoid AttributeError
                shifted_high_series = resampled_df['high'].shift(1)
                shifted_low_series = resampled_df['low'].shift(1)

                if isinstance(shifted_high_series, pd.Series) and isinstance(shifted_low_series, pd.Series):
                    daily_adr = TechnicalIndicatorCalculator.calculate_adr(
                        high_prices=shifted_high_series,
                        low_prices=shifted_low_series,
                        window=period
                    )
                    # Reindex back to original frequency, forward-filling to propagate daily ADR to intraday bars
                    df_stats[f'adr_{period}'] = daily_adr.reindex(df.index, method='ffill')
                else:
                    self.logger.warning(f"Resampled data for ADR period {period} resulted in non-Series object after shift. Setting ADR to NaN.")
                    df_stats[f'adr_{period}'] = np.nan
            else:
                self.logger.warning(f"No valid data after resampling for ADR period {period}. Setting ADR to NaN.")
                df_stats[f'adr_{period}'] = np.nan


        self.logger.debug("Statistical features added.")
        return df_stats

    def _add_derived_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Creates all secondary and tertiary features from base indicators.
        These features might use the *current* values of primary indicators, as long as
        those primary indicators themselves were generated from *past* OHLCV data.
        """
        df_derived = pd.DataFrame(index=df.index)

        # --- Trend Strength ---
        if len(self.config.trend_strength_periods) == 2:
            short_period, long_period = self.config.trend_strength_periods
            short_ema_col = f'ema_{short_period}'
            long_ema_col = f'ema_{long_period}'

            # Check if base EMA features exist in the input DataFrame
            if short_ema_col in df.columns and long_ema_col in df.columns:
                short_ema_series = df[short_ema_col]
                long_ema_series = df[long_ema_col]

                # Handle potential NaN values and division by zero in EMAs gracefully
                safe_long_ema = long_ema_series.replace(0, np.nan) 
                
                df_derived['trend_strength'] = (short_ema_series - safe_long_ema) / (safe_long_ema + FLOAT_EPSILON)
                self.logger.debug(f"Added Trend Strength feature based on {short_ema_col} and {long_ema_col}.")
            else:
                self.logger.warning(f"Base EMA features '{short_ema_col}' or '{long_ema_col}' missing for trend strength calculation. Skipping.")
                df_derived['trend_strength'] = np.nan # Ensure column exists with NaNs if skipped
        else:
            self.logger.warning(f"Trend strength periods not correctly configured as a pair (expected 2, got {len(self.config.trend_strength_periods)}): {self.config.trend_strength_periods}. Skipping trend_strength feature.")
            df_derived['trend_strength'] = np.nan # Ensure column exists with NaNs if skipped


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
                        self.logger.warning(f"Could not compute volatility regime with qcut: {e}. Filling with NaN.", exc_info=True)
                        df_derived['volatility_regime'] = pd.NA # Use pd.NA for nullable integer dtype
                else:
                    self.logger.warning("Insufficient unique ATR values or data points to compute volatility regime. Filling with NaN.")
                    df_derived['volatility_regime'] = pd.NA
            else:
                self.logger.warning(f"'{atr_col_name_for_regime}' feature missing for volatility regime calculation. Skipping.")
                df_derived['volatility_regime'] = pd.NA # Ensure column exists
        else:
            self.logger.warning("No ATR periods configured. Cannot compute volatility regime. Skipping.")
            df_derived['volatility_regime'] = pd.NA # Ensure column exists

        # --- Pattern Clustering (Simple Sum of Binary Patterns) ---
        # This feature combines the candlestick pattern signals into a single score.
        pattern_cols = [f"pattern_{p}_signal" for p in self.config.candlestick_patterns]
        # Filter to only include columns that actually exist in the DataFrame
        existing_pattern_cols = [col for col in pattern_cols if col in df.columns]

        if not existing_pattern_cols:
             self.logger.warning("No configured pattern signal columns found in DataFrame for pattern clustering. Skipping feature.")
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
                        self.logger.debug(f"Standard pivot point column '{p_col}' not found for relative distance calculation.")
            else:
                self.logger.warning(f"ATR column '{atr_col}' not available or all NaN/near-zero for Standard Pivot Point normalization. Skipping normalized Standard Pivot Point distances and binary flags.")
        else:
            self.logger.warning("Standard pivot points not configured or ATR not available for normalized pivot point distances. Skipping.")


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
                    self.logger.warning("Swing pivot columns ('swing_high_pivot', 'swing_low_pivot') not found for relative distance calculation. Skipping.")
            else:
                self.logger.warning(f"ATR column '{atr_col}' not available or all NaN/near-zero for Swing Pivot normalization. Skipping normalized Swing Pivot distances and binary flags.")
        else:
            self.logger.warning("ATR not available for normalized swing pivot distances. Skipping.")

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
                        self.logger.warning(f"ATR column '{atr_col}' not available or all NaN/near-zero for S/R normalization for period {period}. Skipping normalized S/R distance.")
                        df_derived[f'dist_to_support_norm_{period}'] = np.nan
                        df_derived[f'dist_to_resistance_norm_{period}'] = np.nan
                else:
                    self.logger.warning(f"No ATR periods configured. Skipping normalized S/R distance for period {period}.")
                    df_derived[f'dist_to_support_norm_{period}'] = np.nan
                    df_derived[f'dist_to_resistance_norm_{period}'] = np.nan
            else:
                self.logger.warning(f"Base S/R features 'support_{period}' or 'resistance_{period}' not found. Skipping S/R distance calculations for this period.")
                # Ensure columns exist with NaNs if not calculated
                df_derived[f'dist_to_support_{period}'] = np.nan
                df_derived[f'dist_to_resistance_{period}'] = np.nan
                df_derived[f'dist_to_support_norm_{period}'] = np.nan
                df_derived[f'dist_to_resistance_norm_{period}'] = np.nan

        self.logger.debug("Derived features added.")
        return df_derived


    def add_all_technical_and_derived_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Combines all technical indicator calculations and their derived features.

        Args:
            df (pd.DataFrame): Input DataFrame with OHLCV and possibly basic price transforms.

        Returns:
            pd.DataFrame: DataFrame with all technical indicators and derived features.
        """
        self.logger.info("Adding all technical and derived features.")

        df_momentum = self._add_momentum_indicators(df)
        df_trend = self._add_trend_indicators(df)
        df_volatility = self._add_volatility_indicators(df)
        df_volume = self._add_volume_indicators(df)
        df_stats = self._add_statistical_features(df)

        # Concatenate base indicators
        df_combined_indicators = pd.concat([
            df_momentum,
            df_trend,
            df_volatility,
            df_volume,
            df_stats
        ], axis=1)

        # Now add derived features based on these combined indicators
        # Pass the full df including raw (OHLCV) and base indicators for derived features
        df_derived = self._add_derived_features(pd.concat([df, df_combined_indicators], axis=1))

        # Concatenate everything and return
        return pd.concat([df_combined_indicators, df_derived], axis=1)

