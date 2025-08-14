# utils/strategy_execution/entry_filters.py

import logging
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional

# Import FLOAT_EPSILON from params.py
from config.params import FLOAT_EPSILON # Corrected import path

logger = logging.getLogger(__name__)

class EntryFilters:
    """
    Applies various entry conditions (filters) to a potential trade signal.
    This class centralizes the logic for confidence thresholds, volatility regimes,
    and trend filters, ensuring a trade only proceeds if all conditions are met.
    """

    def __init__(self,
                 confidence_filter_enabled: bool,
                 confidence_threshold_long: float,
                 confidence_threshold_short: float,
                 volatility_regime_filter_enabled: bool,
                 volatility_regime_col_name: str,
                 allow_trading_in_volatility_regime: Dict[int, bool],
                 trend_filter_enabled: bool,
                 ema_filter_col_name: str,
                 price_precision: int # For logging formatting
                 ):
        """
        Initializes the EntryFilters with configuration parameters.

        Args:
            confidence_filter_enabled (bool): Whether to apply confidence filtering.
            confidence_threshold_long (float): Minimum probability for a long signal.
            confidence_threshold_short (float): Minimum probability for a short signal.
            volatility_regime_filter_enabled (bool): Whether to apply volatility regime filtering.
            volatility_regime_col_name (str): The name of the column in the OHLCV data that
                                              contains the volatility regime (e.g., 'volatility_regime').
            allow_trading_in_volatility_regime (Dict[int, bool]): A dictionary mapping
                                                                  volatility regime integers (0, 1, 2)
                                                                  to boolean indicating if trading is allowed.
            trend_filter_enabled (bool): Whether to apply trend filtering (EMA crossover).
            ema_filter_col_name (str): The name of the EMA column used for trend filtering.
            price_precision (int): Decimal precision for price values (for logging).
        """
        self.logger = logging.getLogger(self.__class__.__name__)
        self.confidence_filter_enabled = confidence_filter_enabled
        self.confidence_threshold_long = confidence_threshold_long
        self.confidence_threshold_short = confidence_threshold_short
        self.volatility_regime_filter_enabled = volatility_regime_filter_enabled
        self.volatility_regime_col_name = volatility_regime_col_name # Store the new parameter
        self.allow_trading_in_volatility_regime = allow_trading_in_volatility_regime
        self.trend_filter_enabled = trend_filter_enabled
        self.ema_filter_col_name = ema_filter_col_name
        self.price_precision = price_precision
        self.logger.info("EntryFilters initialized.")

    def apply_filters(self,
                      signal: int,
                      current_bar: pd.Series,
                      current_long_proba: Optional[float], # For signal = 1
                      current_short_proba: Optional[float], # For signal = -1
                      price_precision_for_methods: int # Replaces self.price_precision in arguments
                     ) -> bool:
        """
        Applies a series of filters to a given signal to determine if a trade should proceed.

        Args:
            signal (int): The trading signal (-1 for short, 0 for neutral, 1 for long).
            current_bar (pd.Series): A pandas Series representing the current OHLCV bar,
                                     including feature columns like EMA and volatility regime.
            current_long_proba (Optional[float]): The probability score for the long signal (label 1).
            current_short_proba (Optional[float]): The probability score for the short signal (label -1).
            price_precision_for_methods (int): Decimal precision for price values, used for logging.

        Returns:
            bool: True if all filters pass, False otherwise.
        """
        current_timestamp = current_bar.name # Get timestamp from index
        current_close_price = current_bar['close']

        if signal == 0:
            self.logger.debug(f"Signal is neutral (0) at {current_timestamp}. Blocking trade.")
            return False

        # --- 1. Confidence Filter ---
        if self.confidence_filter_enabled:
            if signal == 1: # Long signal
                if pd.isna(current_long_proba): # Check if value is NaN
                    self.logger.warning(f"Entry filter: Confidence filter enabled for LONG but probability value is NaN at {current_timestamp}. Bypassing filter.")
                elif current_long_proba < self.confidence_threshold_long:
                    self.logger.debug(f"Entry filter: LONG signal confidence ({current_long_proba:.4f}) below threshold ({self.confidence_threshold_long:.4f}) at {current_timestamp}. Blocking trade.")
                    return False
            elif signal == -1: # Short signal
                if pd.isna(current_short_proba): # Check if value is NaN
                    self.logger.warning(f"Entry filter: Confidence filter enabled for SHORT but probability value is NaN at {current_timestamp}. Bypassing filter.")
                elif current_short_proba < self.confidence_threshold_short:
                    self.logger.debug(f"Entry filter: SHORT signal confidence ({current_short_proba:.4f}) below threshold ({self.confidence_threshold_short:.4f}) at {current_timestamp}. Blocking trade.")
                    return False
            self.logger.debug(f"Confidence filter passed for signal {signal} at {current_timestamp}.")

        # --- 2. Volatility Regime Filter ---
        if self.volatility_regime_filter_enabled:
            # Correctly use the stored volatility_regime_col_name
            vol_regime_col = self.volatility_regime_col_name
            if vol_regime_col not in current_bar or pd.isna(current_bar[vol_regime_col]):
                self.logger.warning(f"Entry filter: Volatility regime filter enabled but column '{vol_regime_col}' missing or NaN at {current_timestamp}. Blocking trade.")
                return False # Cannot apply filter if regime is missing

            try:
                current_regime_int = int(current_bar[vol_regime_col])
                if current_regime_int not in self.allow_trading_in_volatility_regime:
                    self.logger.warning(f"Entry filter: Volatility regime {current_regime_int} not configured for 'allow_trading'. Blocking trade.")
                    return False # Regime not explicitly handled
                
                if not self.allow_trading_in_volatility_regime[current_regime_int]:
                    self.logger.debug(f"Entry filter: Trading not allowed in volatility regime {current_regime_int} for signal {signal} at {current_timestamp}. Blocking trade.")
                    return False
            except (ValueError, TypeError):
                self.logger.warning(f"Entry filter: Volatility regime value '{current_bar[vol_regime_col]}' is not an integer. Blocking trade.")
                return False
            self.logger.debug(f"Volatility regime filter passed for signal {signal} (Regime: {current_regime_int}) at {current_timestamp}.")


        # --- 3. Trend Filter (EMA Crossover) ---
        if self.trend_filter_enabled:
            ema_col = self.ema_filter_col_name
            if ema_col not in current_bar or pd.isna(current_bar[ema_col]):
                self.logger.warning(f"Entry filter: Trend filter enabled but EMA column '{ema_col}' missing or NaN at {current_timestamp}. Blocking trade.")
                return False # Cannot apply filter if EMA is missing

            latest_ema = current_bar[ema_col]
            
            if signal == 1 and not (current_close_price > latest_ema):
                self.logger.debug(f"Entry filter: LONG signal but price ({current_close_price:.{self.price_precision}f}) not above EMA ({latest_ema:.{self.price_precision}f}) at {current_timestamp}. Blocking.")
                return False
            elif signal == -1 and not (current_close_price < latest_ema):
                self.logger.debug(f"Entry filter: SHORT signal but price ({current_close_price:.{self.price_precision}f}) not below EMA ({latest_ema:.{self.price_precision}f}) at {current_timestamp}. Blocking.")
                return False
            self.logger.debug(f"Trend filter passed for signal {signal} at {current_timestamp}.")

        self.logger.debug(f"All entry filters passed for signal {signal} at {current_timestamp}.")
        return True
