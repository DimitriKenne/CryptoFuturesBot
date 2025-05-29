# utils/features_engineer.py

import sys
import pandas as pd
import talib
import numpy as np
import warnings
import logging
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple
from ta.momentum import RSIIndicator, StochasticOscillator, AwesomeOscillatorIndicator
from ta.trend import SMAIndicator, MACD, EMAIndicator, CCIIndicator
from ta.volatility import BollingerBands, AverageTrueRange
from ta.volume import OnBalanceVolumeIndicator, ChaikinMoneyFlowIndicator, MFIIndicator

# Add project root to Python path for imports
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.append(str(PROJECT_ROOT))

# Import configuration and custom exception
try:
    # Import ONLY FEATURE_CONFIG for parameters
    from config.params import FEATURE_CONFIG
    # Assuming TemporalSafetyError is defined in a custom exceptions.py file
    from utils.exceptions import TemporalSafetyError
except ImportError as e:
    logging.error(f"Failed to import necessary modules: {e}")
    # Re-raise the exception to stop execution if essential imports fail
    raise

# Set up logger for this module
logger = logging.getLogger(__name__)

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
        Initializes the FeatureEngineer with a configuration dictionary
        primarily from FEATURE_CONFIG.

        Args:
            config (Optional[Dict[str, Any]]): Configuration for feature parameters,
                                               expected to be based on FEATURE_CONFIG.
                                               Defaults to config.params.FEATURE_CONFIG.
        """
        # Use provided config or default to FEATURE_CONFIG
        self.config = config or FEATURE_CONFIG.copy()

        if self.config is None:
             logger.error("Feature configuration is not provided or found.")
             raise ValueError("Feature configuration is required.")

        self._validate_config()

        # --- Store Feature Parameters (now mostly lists) ---
        self.sma_periods = list(self.config.get('sma_periods', []))
        self.ema_periods = list(self.config.get('ema_periods', []))
        self.rsi_periods = list(self.config.get('rsi_periods', [14])) # Use list default
        self.bollinger_periods = list(self.config.get('bollinger_periods', [20])) # Use list default
        self.atr_periods = list(self.config.get('atr_periods', [14])) # Use list default
        # Note: Stochastic uses K and D periods, K period is the main lookback
        self.stochastic_periods = list(self.config.get('stochastic_periods', [14])) # Use list default for K period
        self.ao_periods = list(self.config.get('ao_periods', [5, 34])) # Use list default
        self.cci_periods = list(self.config.get('cci_periods', [20])) # Use list default
        self.mfi_periods = list(self.config.get('mfi_periods', [14])) # Use list default
        self.volume_periods = list(self.config.get('volume_periods', [20])) # Use list default
        self.support_resistance_periods = list(self.config.get('support_resistance_periods', [50])) # Use list default
        self.candlestick_patterns = list(self.config.get('candlestick_patterns', []))
        self.fvg_lookback_bars = self.config.get('fvg_lookback_bars', 2) # Still single value for this definition
        self.z_score_periods = list(self.config.get('z_score_periods', [30])) # Use list default
        self.adr_periods = list(self.config.get('adr_periods', [14])) # Use list default
        self.trend_strength_periods = list(self.config.get('trend_strength_periods', [20, 50]))


        # Read the temporal validation enabled flag
        self.validate_temporal_safety_enabled = self.config.get('temporal_validation', {}).get('enabled', True)

        logger.info("FeatureEngineer initialized with general feature configuration.")
        logger.info(f"Temporal safety validation enabled: {self.validate_temporal_safety_enabled}")


    def _validate_config(self):
        """
        Validates the provided feature configuration dictionary.
        Handles both single values (like fvg_lookback_bars) and lists of periods.

        Raises:
            KeyError: If essential configuration keys are missing.
            ValueError: If config values or list elements are invalid.
        """
        # Define required keys (most are now lists)
        required_keys = {
            'sma_periods', 'ema_periods', 'rsi_periods', 'bollinger_periods',
            'atr_periods', 'stochastic_periods', 'ao_periods', 'cci_periods',
            'mfi_periods', 'volume_periods', 'support_resistance_periods',
            'candlestick_patterns', 'fvg_lookback_bars', 'z_score_periods',
            'adr_periods', 'trend_strength_periods', 'temporal_validation'
        }
        missing = required_keys - set(self.config.keys())
        if missing:
            logger.error(f"Missing required configuration keys: {missing}")
            # Depending on criticality, could raise: raise KeyError(f"Missing config keys: {missing}")
            # For now, relying on .get() with defaults in __init__ handles missing keys gracefully
            pass

        # Validate list-based periods (SMA, EMA, RSI, BB, ATR, Stoch, AO, CCI, MFI, Volume, S/R, ZScore, ADR, Trend Strength)
        list_period_keys = {
            'sma_periods': (list, int, True),
            'ema_periods': (list, int, True),
            'rsi_periods': (list, int, True),
            'bollinger_periods': (list, int, True),
            'atr_periods': (list, int, True),
            'stochastic_periods': (list, int, True),
            'ao_periods': (list, int, True), # AO is [short, long], must be 2 positive ints
            'cci_periods': (list, int, True),
            'mfi_periods': (list, int, True),
            'volume_periods': (list, int, True),
            'support_resistance_periods': (list, int, True),
            'z_score_periods': (list, int, True),
            'adr_periods': (list, int, True),
            'trend_strength_periods': (list, int, True), # Trend strength is [short, long], must be 2 positive ints
        }
        for key, (expected_type, element_type, must_be_positive) in list_period_keys.items():
             values = self.config.get(key)
             if not isinstance(values, expected_type):
                  logger.error(f"Invalid value for '{key}': {values}. Must be a list.")
                  raise ValueError(f"Invalid value for '{key}': {values}. Must be a list.")

             # Specific check for AO and Trend Strength list length and order
             if key in ['ao_periods', 'trend_strength_periods']:
                 if len(values) != 2:
                     logger.error(f"Invalid list length for '{key}': {values}. Must contain exactly two periods.")
                     raise ValueError(f"Invalid list length for '{key}': {values}. Must contain exactly two periods.")
                 if values[0] >= values[1]:
                     logger.error(f"Invalid period order for '{key}': {values}. Short period ({values[0]}) must be less than long period ({values[1]}).")
                     raise ValueError(f"Invalid period order for '{key}': {values}. Short period must be less than long period.")


             # Validate list elements
             if not all(isinstance(v, element_type) and (v > 0 if must_be_positive else True) for v in values):
                  logger.error(f"Invalid value for '{key}': {values}. List elements must be positive integers.")
                  raise ValueError(f"Invalid value for '{key}': {values}. List elements must be positive integers.")

        # Validate single value periods (fvg_lookback_bars)
        single_period_keys = ['fvg_lookback_bars']
        for key in single_period_keys:
             value = self.config.get(key)
             if not isinstance(value, int) or value <= 0:
                  logger.error(f"Invalid value for '{key}': {value}. Must be a positive integer.")
                  raise ValueError(f"Invalid value for '{key}': {value}. Must be a positive integer.")


        # Validate candlestick_patterns is a list
        patterns = self.config.get('candlestick_patterns')
        if not isinstance(patterns, list):
             logger.error(f"Invalid value for 'candlestick_patterns': {patterns}. Must be a list of strings.")
             raise ValueError(f"Invalid value for 'candlestick_patterns': {patterns}. Must be a list of strings.")

        # Validate temporal_validation sub-config
        temp_val_cfg = self.config.get('temporal_validation', {})
        if 'enabled' not in temp_val_cfg or 'warning_correlation_threshold' not in temp_val_cfg or 'error_correlation_threshold' not in temp_val_cfg:
             logger.error("Missing 'enabled', 'warning_correlation_threshold', or 'error_correlation_threshold' in 'temporal_validation' config.")
             raise KeyError("Missing 'enabled', 'warning_correlation_threshold', or 'error_correlation_threshold' in 'temporal_validation' config.")


        logger.info("Feature configuration validated successfully.")


    @property
    def required_lookback(self) -> int:
        """
        Calculates the minimum number of data points required for feature engineering
        based on the largest lookback period configured for any indicator.
        This is the number of *previous* bars needed to calculate features for the latest bar.
        """
        # Collect all relevant period sizes from the configuration (FEATURE_CONFIG only)
        period_sizes = []

        # Add single period sizes
        single_period_keys = ['fvg_lookback_bars'] # Only FVG is a single period key now
        for key in single_period_keys:
             value = self.config.get(key)
             if isinstance(value, int) and value > 0:
                 period_sizes.append(value)


        # Add list-based period sizes
        list_period_keys = [
            'sma_periods', 'ema_periods', 'rsi_periods', 'bollinger_periods',
            'atr_periods', 'stochastic_periods', 'ao_periods', 'cci_periods',
            'mfi_periods', 'volume_periods', 'support_resistance_periods',
            'z_score_periods', 'adr_periods', 'trend_strength_periods'
        ]
        for key in list_period_keys:
            values = self.config.get(key)
            if isinstance(values, list):
                for period in values:
                    if isinstance(period, int) and period > 0:
                        period_sizes.append(period)

        # The minimum required lookback is the largest period size found.
        # We also need to consider shifts used in calculations (e.g., shift(1) for indicators, shift(2) for FVG).
        # Max period size + 2 seems a safe buffer for shifts.
        max_period_size = max(period_sizes) if period_sizes else 0
        calculated_lookback = max_period_size + 2 # Add 2 for shifts like FVG (i vs i-2) and log returns (t-1/t-2)
        logger.debug(f"Calculated required lookback for FeatureEngineer: {calculated_lookback} bars (max_period: {max_period_size}).")
        return calculated_lookback


    def _validate_dataframe(self, df: pd.DataFrame):
        """
        Validates the input DataFrame structure and integrity.

        Args:
            df (pd.DataFrame): The DataFrame to validate.

        Raises:
            ValueError: If required columns are missing, index is not time-sorted,
                        or contains NaN/Inf values in OHLCV data.
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

        # Check for NaNs or Infs in required OHLCV columns
        for col in required_cols:
            if df[col].isnull().any():
                logger.error(f"Input DataFrame contains NaN values in column: {col}")
                raise ValueError(f"Input contains NaN values in column: {col}")
            if not np.isfinite(df[col]).all():
                 logger.error(f"Input DataFrame contains Inf or non-finite values in column: {col}")
                 raise ValueError(f"Input contains Inf or non-finite values in column: {col}")

        logger.debug("Input DataFrame validated successfully.")


    def compute_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculates time-safe general technical indicators for configured periods.
        Indicators are calculated based on *past* data relative to the current row's index.
        Creates a separate feature column for each period.
        """
        # Assumes df is already validated and has a DatetimeIndex
        df_features = pd.DataFrame(index=df.index) # Create a new DataFrame for features

        # Price Transformations (calculated on past data)
        # Log returns at time t is log(close[t-1]/close[t-2]), assigned to index t
        df_features['log_returns'] = np.log(df['close'].shift(1) / df['close'].shift(2))
        df_features['typical_price'] = (df['high'].shift(1) + df['low'].shift(1) + df['close'].shift(1)) / 3


        # Support/Resistance Levels (calculated on past data for multiple periods)
        for period in self.support_resistance_periods:
            # Resistance at time t is max high in [t-period, t-1]
            df_features[f'resistance_{period}'] = df['high'].rolling(period).max().shift(1)
            # Support at time t is min low in [t-period, t-1]
            df_features[f'support_{period}'] = df['low'].rolling(period).min().shift(1)
            # Distance features calculated based on close[t-1]
            df_features[f'dist_to_support_{period}'] = df['close'].shift(1) - df_features[f'support_{period}']
            df_features[f'dist_to_resistance_{period}'] = df_features[f'resistance_{period}'] - df['close'].shift(1)


        # Momentum Indicators (calculated on past data for multiple periods)
        for period in self.rsi_periods:
             df_features[f'rsi_{period}'] = RSIIndicator(df['close'].shift(1), period).rsi()

        # Stochastic Oscillator (Calculated for multiple K periods, D uses a fixed period like 3 relative to K)
        stoch_d_period = 3 # Standard D period
        for period_k in self.stochastic_periods:
            stoch = StochasticOscillator(df['high'].shift(1), df['low'].shift(1), df['close'].shift(1), window=period_k, smooth_window=stoch_d_period)
            df_features[f'stoch_k_{period_k}'] = stoch.stoch()
            df_features[f'stoch_d_{period_k}'] = stoch.stoch_signal() # D is relative to this K calculation

        # Awesome Oscillator (AO periods are a pair [short, long])
        # AO at time t is based on median prices up to t-1
        # We use the single pair defined in config
        if len(self.ao_periods) == 2:
             ao = AwesomeOscillatorIndicator(high=df['high'].shift(1), low=df['low'].shift(1),
                                     window1=self.ao_periods[0],
                                     window2=self.ao_periods[1])
             df_features['ao'] = ao.awesome_oscillator()
        else:
             logger.warning(f"AO periods not correctly configured as a pair: {self.ao_periods}. Skipping AO feature.")
             df_features['ao'] = np.nan


        # Trend Indicators (calculated on past data for multiple periods)
        for period in self.sma_periods:
            df_features[f'sma_{period}'] = SMAIndicator(df['close'].shift(1), period).sma_indicator()

        for period in self.ema_periods:
            df_features[f'ema_{period}'] = EMAIndicator(df['close'].shift(1), period).ema_indicator()

        # MACD parameters are typically fixed (12, 26, 9), calculate just one MACD
        # Could make these configurable lists too, but less common
        macd = MACD(df['close'].shift(1))
        df_features['macd'] = macd.macd()
        df_features['macd_signal'] = macd.macd_signal()

        # Commodity Channel Index (CCI)
        for period in self.cci_periods:
            cci = CCIIndicator(high=df['high'].shift(1), low=df['low'].shift(1), close=df['close'].shift(1), window=period)
            df_features[f'cci_{period}'] = cci.cci()


        # Volatility Indicators (calculated on past data for multiple periods)
        for period in self.bollinger_periods:
            bb = BollingerBands(df['close'].shift(1), period)
            df_features[f'bb_upper_{period}'] = bb.bollinger_hband()
            df_features[f'bb_lower_{period}'] = bb.bollinger_lband()
            df_features[f'bb_width_{period}'] = df_features[f'bb_upper_{period}'] - df_features[f'bb_lower_{period}']

        # ATR calculated for multiple periods
        for period in self.atr_periods:
             if len(df) > period:
                  df_features[f'atr_{period}'] = AverageTrueRange(
                      df['high'].shift(1), df['low'].shift(1), df['close'].shift(1), period
                  ).average_true_range()
             else:
                  df_features[f'atr_{period}'] = np.nan


        # Volume Indicators (calculated on past data for multiple periods)
        # OBV is cumulative, only one needed, period param is sometimes just smoothing
        # CMF and MFI use a period for sum/average calculation
        df_features['obv'] = OnBalanceVolumeIndicator(df['close'].shift(1), df['volume'].shift(1)).on_balance_volume()

        for period in self.volume_periods: # Assuming volume_period applies to CMF/MFI window
             # CMF
             cmf = ChaikinMoneyFlowIndicator(
                 df['high'].shift(1), df['low'].shift(1), df['close'].shift(1), df['volume'].shift(1), window=period
             )
             df_features[f'cmf_{period}'] = cmf.chaikin_money_flow()

             # Money Flow Index (MFI)
             mfi = MFIIndicator(high=df['high'].shift(1), low=df['low'].shift(1), close=df['close'].shift(1), volume=df['volume'].shift(1), window=period)
             df_features[f'mfi_{period}'] = mfi.money_flow_index()


        logger.debug("Technical indicators computed.")
        return df_features


    def detect_candlestick_patterns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Identifies candlestick patterns using TALIB, ensuring temporal safety.
        Pattern detection at time t is based on candle data up to time t.
        The resulting signal is available as a feature at time t.
        """
        # Assumes df is already validated and has a DatetimeIndex
        df_patterns = pd.DataFrame(index=df.index) # Create a new DataFrame for patterns

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
            # Add more patterns from TALIB as needed
        }

        supported_patterns = self.candlestick_patterns # Use patterns from config
        for pattern in supported_patterns:
            if pattern in pattern_map:
                # Calculate pattern signal based on current and past candles (up to index t)
                # The output of TALIB.CDL functions is aligned with the input index
                pattern_values = pattern_map[pattern](
                    df['open'], df['high'], df['low'], df['close']
                )
                # Convert TALIB output (usually 100, -100, 0) to our signal convention
                # This signal is available at time t.
                df_patterns[f'pattern_{pattern}_signal'] = pattern_values.apply( # Added 'pattern_' prefix for clarity
                    lambda x: 1 if x > 0 else (-1 if x < 0 else 0)
                )

            else:
                logger.warning(f"Configured pattern '{pattern}' is not supported by the current implementation.")


        logger.debug("Candlestick patterns detected.")
        return df_patterns


    def detect_fvg(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Detects Fair Value Gaps (FVG).
        Using the standard 3-candle definition:
        A bullish FVG at candle 'i' is when low[i] > high[i-2].
        A bearish FVG at candle 'i' is when high[i] < low[i-2].
        This feature is available at time 'i'.
        """
        # Assumes df is already validated and has a DatetimeIndex
        df_fvg = pd.DataFrame(index=df.index) # Create a new DataFrame for FVG

        # Calculate standard FVG at time t based on data up to time t
        # Bullish FVG at t: low[t] > high[t-2]
        bullish_fvg_at_t = (df['low'] > df['high'].shift(2))

        # Bearish FVG at t: high[t] < df['low'].shift(2))
        bearish_fvg_at_t = (df['high'] < df['low'].shift(2))

        # Initialize FVG column with 0
        df_fvg['fvg'] = 0
        # Use .loc to avoid SettingWithCopyWarning
        df_fvg.loc[bullish_fvg_at_t, 'fvg'] = 1
        df_fvg.loc[bearish_fvg_at_t, 'fvg'] = -1

        logger.debug("Fair Value Gap detected (standard 3-candle).")
        return df_fvg


    def add_derived_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Creates secondary features from base indicators, ensuring temporal safety.
        Derived features at time t are based on base features available at time t or t-1.
        """
        # Assumes df contains base features and has a DatetimeIndex
        df_derived = pd.DataFrame(index=df.index) # Create a new DataFrame for derived features
        cfg = self.config

        # Trend Strength (example derived feature)
        # Uses the specific pair of periods from trend_strength_periods config
        if len(self.trend_strength_periods) == 2:
            short_period, long_period = self.trend_strength_periods
            short_sma_col = f'sma_{short_period}'
            long_sma_col = f'sma_{long_period}'

            # Ensure the base SMA features exist before calculating derived features
            # Use SMA values at time t as features for time t
            if short_sma_col not in df.columns or long_sma_col not in df.columns:
                 logger.error(f"Base SMA features '{short_sma_col}' or '{long_sma_col}' missing for trend strength calculation.")
                 df_derived['trend_strength'] = np.nan # Add the column with NaNs
            else:
                # Trend strength at time t is based on SMA values at t
                short_sma_t = df[short_sma_col]
                long_sma_t = df[long_sma_col]
                epsilon = 1e-8 # Avoid division by zero
                # Calculate the percentage difference between the short and long SMA
                df_derived['trend_strength'] = (short_sma_t - long_sma_t) / (long_sma_t + epsilon)
        else:
            logger.warning(f"Trend strength periods not correctly configured as a pair: {self.trend_strength_periods}. Skipping trend_strength feature.")
            df_derived['trend_strength'] = np.nan


        # Volatility Regime (categorical) - Now based on one of the generated ATR features
        # Let's use the ATR with the shortest period from the configured list for responsiveness
        if self.atr_periods:
            atr_period_for_regime = min(self.atr_periods)
            atr_col_name_for_regime = f'atr_{atr_period_for_regime}'

            if atr_col_name_for_regime not in df.columns:
                 logger.error(f"'{atr_col_name_for_regime}' feature missing for volatility regime calculation.")
                 df_derived['volatility_regime'] = pd.NA # Use pandas NA for nullable integer
            else:
                # Use the ATR value at time t for calculation
                atr_t = df[atr_col_name_for_regime]
                # Use qcut on the ATR at time t to determine regimes based on the distribution up to t
                # Ensure enough non-NaN values for qcut
                atr_t_dropna = atr_t.dropna().copy()
                # Need at least 2 unique values for 3 quantiles if duplicates='drop'
                if atr_t_dropna.shape[0] >= 3 and len(atr_t_dropna.unique()) >= 2:
                    try:
                        # Compute qcut and explicitly convert the category codes to integer
                        df_derived.loc[atr_t_dropna.index, 'volatility_regime'] = pd.qcut(
                            atr_t_dropna, # Apply qcut on the non-NaN data
                            q=3,
                            labels=False, # Use integer labels directly (0, 1, 2)
                            duplicates='drop' # Handle cases with fewer than q unique values
                        ).astype(pd.Int8Dtype()) # Use nullable integer type

                    except Exception as e:
                        logger.warning(f"Could not compute volatility regime: {e}. Filling with NaN.", exc_info=True)
                        # If qcut fails, fill with NA for the rows where it was attempted
                        df_derived['volatility_regime'] = pd.NA # Use pandas NA for nullable integer type

                else:
                     logger.warning("Insufficient data points or unique ATR values to compute volatility regime. Filling with NaN.")
                     df_derived['volatility_regime'] = pd.NA # Use pandas NA
        else:
            logger.warning("No ATR periods configured. Cannot compute volatility regime. Filling with NaN.")
            df_derived['volatility_regime'] = pd.NA


        # Pattern Clustering (example derived feature)
        # Pattern cluster at time t is based on pattern signals at time t
        # Ensure configured candlestick patterns list is used and uses the new column naming
        pattern_cols = [f"pattern_{p}_signal" for p in self.candlestick_patterns]
        # Ensure all required pattern columns exist in the input df
        existing_pattern_cols = [col for col in pattern_cols if col in df.columns]

        if not existing_pattern_cols:
             logger.warning("No configured pattern signal columns found in DataFrame for pattern clustering. Skipping feature.")
             df_derived['pattern_cluster'] = np.nan # Add the column with NaNs
        else:
            # Sum the existing pattern signals at time t
            df_derived['pattern_cluster'] = df[existing_pattern_cols].sum(axis=1)


        logger.debug("Derived features added.")
        return df_derived


    def _validate_temporal_safety(self, df: pd.DataFrame) -> List[str]:
        """
        Performs enhanced temporal safety validation by checking correlation
        between each feature and the *next* period's close price change.
        Identifies and returns a list of features that violate the thresholds.
        Now checks all generated general features.

        Args:
            df (pd.DataFrame): DataFrame containing features and the 'close' price.
                               Expected to have NaNs for initial lookback period.

        Returns:
            List[str]: A list of feature names that failed the temporal safety check.
        """
        cfg = self.config.get('temporal_validation', {})
        warn_threshold = cfg.get('warning_correlation_threshold', 0.3)
        error_threshold = cfg.get('error_correlation_threshold', 0.5)

        if 'close' not in df.columns:
            logger.error("Cannot perform temporal safety validation: 'close' column is missing.")
            return [] # Cannot validate without close price

        # Calculate the next period's close price change (target variable proxy)
        # This is the value we are trying to potentially predict
        next_close_change = df['close'].pct_change().shift(-1)

        violating_features = []

        # Iterate through all columns except the original OHLCV columns and the target proxy
        ohlcv_cols = {'open', 'high', 'low', 'close', 'volume'}

        # Explicitly skip specific categorical or known non-numeric/constant features
        # This list should include any non-numeric columns added by feature engineering
        cols_to_skip_correlation = [col for col in ['volatility_regime', 'fvg'] if col in df.columns] # 'fvg' is also categorical (-1, 0, 1)


        feature_cols = [col for col in df.columns if col not in ohlcv_cols and col != next_close_change.name and col not in cols_to_skip_correlation]


        for col in feature_cols:
            # Ensure the column is numeric before calculating correlation
            if not pd.api.types.is_numeric_dtype(df[col]):
                logger.debug(f"Skipping temporal safety correlation check for non-numeric column '{col}'.")
                continue

            x = df[col]
            y = next_close_change
            # Only consider rows where both the feature and the next close change are available
            # Use pd.notna() for checks
            valid = pd.notna(x) & pd.notna(y)
            x_valid = x[valid]
            y_valid = y[valid]

            # Need at least 2 data points to compute correlation, and variability in both series
            if len(x_valid) < 2 or x_valid.std() < 1e-9 or y_valid.std() < 1e-9:
                logger.debug(f"Skipping temporal validation for {col}: Insufficient valid data points ({len(x_valid)}) or near-zero standard deviation.")
                continue

            try:
                # Compute the absolute correlation between the feature at time t and the close change at time t+1
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
                # Catch any unexpected errors during correlation calculation
                logger.warning(f"Could not compute correlation for feature '{col}': {e}")

        logger.info(f"Temporal safety validation complete. {len(violating_features)} features violated the error threshold.")
        return violating_features


    def process(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Processes raw OHLCV data to generate general-purpose technical and
        statistical features for configured periods.
        Initial rows affected by lookback periods will contain NaN values.
        Performs temporal safety checks if enabled.

        Args:
            df (pd.DataFrame): Input DataFrame with OHLCV data and a DatetimeIndex.

        Returns:
            pd.DataFrame: DataFrame containing the original OHLCV data plus engineered features,
                          including initial rows with NaN values due to lookback periods.

        Raises:
            ValueError: If input DataFrame is invalid.
            TemporalSafetyError: If features violate temporal safety thresholds after processing
                                 (temporal safety is applied to the DataFrame including NaNs).
        """
        logger.info("Starting general feature engineering process.")
        # Initial validation of clean input data (should have no NaNs/Infs in OHLCV)
        self._validate_dataframe(df)

        # Create a copy to avoid modifying the original DataFrame
        df_processed = df.copy()

        # Compute base technical indicators for all configured periods
        df_indicators = self.compute_technical_indicators(df_processed)
        df_processed = pd.concat([df_processed, df_indicators], axis=1)

        # Detect candlestick patterns
        df_patterns = self.detect_candlestick_patterns(df_processed)
        df_processed = pd.concat([df_processed, df_patterns], axis=1)

        # Detect FVG (using standard 3-candle definition)
        df_fvg = self.detect_fvg(df_processed)
        df_processed = pd.concat([df_processed, df_fvg], axis=1)

        # Add derived features (uses base indicators generated above)
        df_derived = self.add_derived_features(df_processed)
        df_processed = pd.concat([df_processed, df_derived], axis=1)

        # The DataFrame will contain NaN values for the initial rows
        # due to lookback periods of the indicators.
        # The calling code (e.g., Training, Backtesting, Bot) must handle these NaNs before use.
        df_with_nan = df_processed.copy()


        # Ensure categorical features have correct nullable types (apply to the df with NaNs)
        # This should match the columns where we assign -1, 0, or 1/pd.NA
        categorical_cols = ['fvg', 'volatility_regime'] # FVG and Volatility Regime are treated as categories

        for col in categorical_cols:
             if col in df_with_nan.columns: # Double check existence
                  # Use .loc to avoid SettingWithCopyWarning
                  # Use Int8Dtype as values are small integers (-1, 0, 1, or NA)
                  df_with_nan.loc[:, col] = df_with_nan[col].astype(pd.Int8Dtype())
             else:
                  logger.debug(f"Categorical column '{col}' not found in DataFrame to cast type.")


        logger.info(f"General feature engineering complete. DataFrame shape (including NaNs): {df_with_nan.shape}")


        # Conditionally perform temporal safety validation (on the DataFrame with NaNs)
        if self.validate_temporal_safety_enabled:
            logger.info("Performing temporal safety validation on general features...")
            # Pass the DataFrame with NaNs to the validation method
            violating_features = self._validate_temporal_safety(df_with_nan)

            if violating_features:
                # Raise a TemporalSafetyError listing all violating features
                error_msg = f"Temporal safety violations detected in general features: {', '.join(violating_features)}"
                logger.error(error_msg)
                # Pass the list of violating features with the exception
                raise TemporalSafetyError(error_msg, features=violating_features)
            else:
                logger.info("Temporal safety validation passed for general features.")
        else:
            logger.info("Temporal safety validation skipped as per configuration.")


        # Return the DataFrame with original index preserved, including initial NaNs
        return df_with_nan
