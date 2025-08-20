# utils/feature_engineering/technical_indicator_calculator.py

import pandas as pd
import talib
import numpy as np
import logging
from pathlib import Path
import sys

# Add project root to Python path for imports
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.append(str(PROJECT_ROOT))

# Import the 'ta' library indicators directly
from ta.momentum import RSIIndicator, StochasticOscillator, AwesomeOscillatorIndicator
from ta.trend import SMAIndicator, MACD, EMAIndicator, CCIIndicator
from ta.volatility import BollingerBands, AverageTrueRange
from ta.volume import OnBalanceVolumeIndicator, ChaikinMoneyFlowIndicator, MFIIndicator

logger = logging.getLogger(__name__)

# Import FLOAT_EPSILON from the central constants file (config.params)
try:
    from config.params import FLOAT_EPSILON
except ImportError as e:
    logger.error(f"Failed to import FLOAT_EPSILON from config.params: {e}")
    FLOAT_EPSILON = 1e-9

class TechnicalIndicatorCalculator:
    """
    A utility class for calculating raw technical indicators.

    This class provides static methods to compute various technical analysis
    indicators using pandas Series as input. It focuses solely on the
    calculation logic, without managing configuration lists or temporal shifts,
    which are handled by the FeatureEngineer.
    """

    @staticmethod
    def calculate_sma(close_prices: pd.Series, window: int) -> pd.Series:
        """Calculates Simple Moving Average (SMA)."""
        return SMAIndicator(close=close_prices, window=window, fillna=False).sma_indicator()

    @staticmethod
    def calculate_ema(close_prices: pd.Series, window: int) -> pd.Series:
        """Calculates Exponential Moving Average (EMA)."""
        return EMAIndicator(close=close_prices, window=window, fillna=False).ema_indicator()

    @staticmethod
    def calculate_rsi(close_prices: pd.Series, window: int) -> pd.Series:
        """Calculates Relative Strength Index (RSI)."""
        return RSIIndicator(close=close_prices, window=window, fillna=False).rsi()

    @staticmethod
    def calculate_bollinger_bands(close_prices: pd.Series, window: int, window_dev: float = 2.0) -> pd.DataFrame:
        """
        Calculates Bollinger Bands (BB).
        Returns a DataFrame with 'hband', 'lband', 'wband', 'pband'.
        """
        bb = BollingerBands(close=close_prices, window=window, window_dev=window_dev, fillna=False)
        return pd.DataFrame({
            'hband': bb.bollinger_hband(),
            'lband': bb.bollinger_lband(),
            'wband': bb.bollinger_wband(),
            'pband': bb.bollinger_pband()
        }, index=close_prices.index)

    @staticmethod
    def calculate_atr(high_prices: pd.Series, low_prices: pd.Series, close_prices: pd.Series, window: int) -> pd.Series:
        """Calculates Average True Range (ATR)."""
        return AverageTrueRange(high=high_prices, low=low_prices, close=close_prices, window=window, fillna=False).average_true_range()

    @staticmethod
    def calculate_stochastic_oscillator(high_prices: pd.Series, low_prices: pd.Series, close_prices: pd.Series, window: int, smooth_window: int = 3) -> pd.DataFrame:
        """
        Calculates Stochastic Oscillator (%K and %D).
        Returns a DataFrame with 'stoch' (%K) and 'stoch_signal' (%D).
        """
        stoch = StochasticOscillator(high=high_prices, low=low_prices, close=close_prices, window=window, smooth_window=smooth_window, fillna=False)
        return pd.DataFrame({
            'stoch': stoch.stoch(),
            'stoch_signal': stoch.stoch_signal()
        }, index=close_prices.index)

    @staticmethod
    def calculate_awesome_oscillator(high_prices: pd.Series, low_prices: pd.Series, window1: int, window2: int) -> pd.Series:
        """Calculates Awesome Oscillator (AO)."""
        ao = AwesomeOscillatorIndicator(high=high_prices, low=low_prices, window1=window1, window2=window2, fillna=False)
        return ao.awesome_oscillator()

    @staticmethod
    def calculate_cci(high_prices: pd.Series, low_prices: pd.Series, close_prices: pd.Series, window: int) -> pd.Series:
        """Calculates Commodity Channel Index (CCI)."""
        return CCIIndicator(high=high_prices, low=low_prices, close=close_prices, window=window, fillna=False).cci()

    @staticmethod
    def calculate_mfi(high_prices: pd.Series, low_prices: pd.Series, close_prices: pd.Series, volume: pd.Series, window: int) -> pd.Series:
        """Calculates Money Flow Index (MFI)."""
        return MFIIndicator(high=high_prices, low=low_prices, close=close_prices, volume=volume, window=window, fillna=False).money_flow_index()

    @staticmethod
    def calculate_obv(close_prices: pd.Series, volume: pd.Series) -> pd.Series:
        """Calculates On-Balance Volume (OBV)."""
        return OnBalanceVolumeIndicator(close=close_prices, volume=volume, fillna=False).on_balance_volume()

    @staticmethod
    def calculate_cmf(high_prices: pd.Series, low_prices: pd.Series, close_prices: pd.Series, volume: pd.Series, window: int) -> pd.Series:
        """Calculates Chaikin Money Flow (CMF)."""
        return ChaikinMoneyFlowIndicator(high=high_prices, low=low_prices, close=close_prices, volume=volume, window=window, fillna=False).chaikin_money_flow()

    @staticmethod
    def calculate_macd(close_prices: pd.Series, window_fast: int = 12, window_slow: int = 26, window_sign: int = 9) -> pd.DataFrame:
        """
        Calculates Moving Average Convergence Divergence (MACD).
        Returns a DataFrame with 'macd', 'macd_signal', 'macd_diff'.
        """
        macd = MACD(close=close_prices, window_fast=window_fast, window_slow=window_slow, window_sign=window_sign, fillna=False)
        return pd.DataFrame({
            'macd': macd.macd(),
            'macd_signal': macd.macd_signal(),
            'macd_diff': macd.macd_diff()
        }, index=close_prices.index)

    @staticmethod
    def calculate_z_score(close_prices: pd.Series, window: int) -> pd.Series:
        """Calculates Z-Score for close prices."""
        mean = close_prices.rolling(window=window).mean()
        std = close_prices.rolling(window=window).std()
        return (close_prices - mean) / (std + FLOAT_EPSILON)

    @staticmethod
    def calculate_adr(high_prices: pd.Series, low_prices: pd.Series, window: int, interval: str) -> pd.Series:
        """
        Calculates Average Daily Range (ADR) for any interval.
        For intraday data, aggregates to daily high/low, computes daily range,
        then rolling mean, and maps back to the original index.
        For daily or higher, computes ADR directly.

        Args:
            high_prices (pd.Series): High prices with DatetimeIndex.
            low_prices (pd.Series): Low prices with DatetimeIndex.
            window (int): Lookback window for ADR.
            current_interval (str): Data interval (e.g., '5m', '1d').

        Returns:
            pd.Series: ADR values indexed to the original data.
        """
        df_temp = pd.DataFrame({'high': high_prices, 'low': low_prices})
        is_intraday = interval not in ["1d", "1D", "daily", "1w", "1W", "weekly", "1M", "monthly"]

        if is_intraday:
            # Aggregate to daily OHLC to get true daily high and low
            daily_ohlc = df_temp.resample('D').agg({'high': 'max', 'low': 'min'})
        else:
            # If already daily/weekly/monthly, use the provided series directly
            daily_ohlc = df_temp
            
        # Calculate daily range (current day's high - current day's low)
        daily_range = daily_ohlc['high'] - daily_ohlc['low']
        adr_daily = daily_range.shift(1).rolling(window=window, min_periods=window).mean()
        adr_mapped_to_original_index = adr_daily.reindex(high_prices.index, method='ffill')
        return adr_mapped_to_original_index


    @staticmethod
    def calculate_volume_oscillator(volume: pd.Series, short_ema_window: int, long_ema_window: int) -> pd.Series:
        """
        Calculates the volume oscillator as per Pine Script: 100 * (EMA(vol, short) - EMA(vol, long)) / EMA(vol, long).
        """
        short_ema = volume.ewm(span=short_ema_window, adjust=False).mean()
        long_ema = volume.ewm(span=long_ema_window, adjust=False).mean()
        safe_long_ema = long_ema.replace(0, np.nan)
        return 100 * (short_ema - safe_long_ema) / safe_long_ema

    @staticmethod
    def detect_candlestick_pattern(open_prices: pd.Series, high_prices: pd.Series, low_prices: pd.Series, close_prices: pd.Series, pattern_func: callable) -> pd.Series:
        """
        Detects a specific candlestick pattern using TA-Lib.
        `pattern_func` should be a TA-Lib CDL_* function.
        Returns a Series with values (100, -100, or 0).
        """
        return pattern_func(open_prices, high_prices, low_prices, close_prices)
