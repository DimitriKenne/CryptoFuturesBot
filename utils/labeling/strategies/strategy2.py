# utils/labeling_strategies/strategy2.py

import pandas as pd
import numpy as np
import logging
from typing import Any, List, Optional
from .base_strategy import BaseLabelingStrategy, logger, FLOAT_EPSILON

# Import the specific config dataclass for this strategy
from config.label_config_schema import Strategy2Config

class Strategy2(BaseLabelingStrategy):
    """
    Strategy 2: Net Forward Return Quantile.

    This strategy labels data based on future net returns (accounting for fees and slippage)
    relative to dynamically calculated quantile thresholds.

    - A label of '1' (Buy) is assigned if the future net return for a long position
      is greater than or equal to a specified positive quantile threshold.
    - A label of '-1' (Sell) is assigned if the future net return for a short position
      is greater than or equal to a specified negative quantile threshold (meaning price
      dropped sufficiently to yield a positive short return).
    - A label of '0' (Neutral) is assigned otherwise.
    """

    def __init__(self, config: Strategy2Config, logger: logging.Logger, trading_fee_rate: float, slippage_tolerance_pct: float):
        """
        Initializes Strategy 2 (Net Forward Return Quantile Strategy).

        Args:
            config (Strategy2Config): The configuration dataclass for this strategy.
            logger (logging.Logger): A logger instance.
            trading_fee_rate (float): The transaction fee rate.
            slippage_tolerance_pct (float): The estimated slippage rate.
        """
        # Pass config to the superclass, which now also accepts fee/slippage
        super().__init__(config, logger, trading_fee_rate, slippage_tolerance_pct)
        self.logger.info("Strategy 2 (Net Forward Return Quantile) initializing...")
        self._validate_strategy_config()

        # Access parameters directly from the Strategy2Config dataclass
        self.future_return_window = self.config.future_return_window
        self.quantile_threshold_long = self.config.quantile_threshold_long
        self.quantile_threshold_short = self.config.quantile_threshold_short
        self.return_type = self.config.return_type

        self.logger.info(f"  Forward Window (future_return_window): {self.future_return_window} bars")
        self.logger.info(f"  Transaction Fee: {self.trading_fee_rate}")
        self.logger.info(f"  Slippage: {self.slippage_tolerance_pct}")
        self.logger.info(f"  Buy Quantile Percentile (Long): {self.quantile_threshold_long}")
        self.logger.info(f"  Sell Quantile Percentile (Short): {self.quantile_threshold_short}")
        self.logger.info(f"  Return Type: {self.return_type}")


    def _validate_strategy_config(self):
        """
        Validates configuration parameters specific to Strategy 2,
        now accessing directly from self.config (Strategy2Config).
        """
        if not isinstance(self.config.future_return_window, int) or self.config.future_return_window <= 0:
            raise ValueError("'future_return_window' must be a positive integer.")
        
        if not isinstance(self.config.quantile_threshold_long, (int, float)) or not (0 < self.config.quantile_threshold_long < 100):
            raise ValueError("'quantile_threshold_long' must be a number between 0 and 100 (exclusive).")
        
        if not isinstance(self.config.quantile_threshold_short, (int, float)) or not (0 < self.config.quantile_threshold_short < 100):
            raise ValueError("'quantile_threshold_short' must be a number between 0 and 100 (exclusive).")

        if self.config.return_type not in ['log_returns', 'simple_returns']:
            raise ValueError("'return_type' must be 'log_returns' or 'simple_returns'.")

        self.logger.debug("Strategy 2 config validated.")


    def calculate_raw_labels(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculates raw labels for Strategy 2 (Net Forward Return Quantile).

        Args:
            df (pd.DataFrame): DataFrame with OHLCV data, indexed by time.
                               Assumed to be cleaned (no NaNs in OHLCV) by LabelGenerator.

        Returns:
            pd.DataFrame: DataFrame with 'label' column (1, -1, or 0).
        """
        self.logger.debug("Calculating raw labels for Strategy 2 (Net Forward Return Quantile).")
        self._validate_input_df(df, ['close'])

        df_copy = df.copy() # Work on a copy

        # Compute future close
        df_copy['Future_Close'] = df_copy['close'].shift(-self.future_return_window)

        # Calculate Net Return (accounting for fees and slippage)
        # Using self.trading_fee_rate and self.slippage_tolerance_pct from base class
        entry_cost_long_factor = (1 + self.trading_fee_rate + self.slippage_tolerance_pct)
        exit_revenue_long_factor = (1 - self.trading_fee_rate - self.slippage_tolerance_pct)

        safe_current_close = df_copy['close'].replace(0, np.nan)

        # Calculate potential net return assuming a long position
        df_copy['Net_Return_Long'] = (df_copy['Future_Close'] * exit_revenue_long_factor - safe_current_close * entry_cost_long_factor) / (safe_current_close * entry_cost_long_factor)

        # Calculate potential net return assuming a short position
        entry_revenue_short_factor = (1 - self.trading_fee_rate - self.slippage_tolerance_pct)
        exit_cost_short_factor = (1 + self.trading_fee_rate + self.slippage_tolerance_pct)
        safe_current_close_short_entry = df_copy['close'].replace(0, np.nan) * entry_revenue_short_factor
        df_copy['Net_Return_Short'] = (safe_current_close_short_entry - df_copy['Future_Close'] * exit_cost_short_factor) / safe_current_close_short_entry


        # Convert to percentage if not already, or apply log returns if specified
        if self.return_type == 'log_returns':
            df_copy['Net_Return_Long'] = np.log(1 + df_copy['Net_Return_Long'])
            df_copy['Net_Return_Short'] = np.log(1 + df_copy['Net_Return_Short'])


        # Drop NaN values (caused by shifting)
        initial_rows = len(df_copy)
        df_copy.dropna(subset=['Future_Close', 'Net_Return_Long', 'Net_Return_Short'], inplace=True)

        if len(df_copy) < initial_rows:
            self.logger.warning(f"Dropped {initial_rows - len(df_copy)} rows with NaNs after calculating future net returns.")
        if df_copy.empty:
            self.logger.error("DataFrame is empty after dropping NaNs. Cannot generate labels.")
            return pd.DataFrame(index=df_copy.index, data={'label': 0})

        # Calculate quantiles for positive and negative returns separately
        positive_returns = df_copy['Net_Return_Long'][df_copy['Net_Return_Long'] > 0].dropna()
        # For short returns, we are looking for positive values after the transformation (price drop results in positive return)
        negative_returns = df_copy['Net_Return_Short'][df_copy['Net_Return_Short'] > 0].dropna()

        buy_threshold = 0.0 # Default to no signal if no positive returns
        if not positive_returns.empty:
            buy_threshold = np.percentile(positive_returns, self.quantile_threshold_long)
            buy_threshold = max(buy_threshold, FLOAT_EPSILON) # Ensure buy_threshold is positive
        else:
            self.logger.warning("No positive net returns found for buy quantile calculation. Buy signals will be 0.")

        sell_threshold = 0.0 # Default to no signal if no profitable short returns
        if not negative_returns.empty:
            sell_threshold = np.percentile(negative_returns, self.quantile_threshold_short)
            sell_threshold = max(sell_threshold, FLOAT_EPSILON) # Ensure sell_threshold is positive
        else:
            self.logger.warning("No profitable short net returns found for sell quantile calculation. Sell signals will be 0.")


        self.logger.debug(f"Calculated Buy Threshold: {buy_threshold}, Sell Threshold: {sell_threshold}")

        # Assign Labels
        df_copy['label'] = 0 # Default to neutral

        # Buy (1): Net Return for long position is above threshold
        df_copy.loc[df_copy['Net_Return_Long'] >= buy_threshold, 'label'] = 1

        # Sell (-1): Net Return for short position is above threshold (meaning price dropped enough for profit)
        df_copy.loc[df_copy['Net_Return_Short'] >= sell_threshold, 'label'] = -1

        # Drop temporary columns
        df_copy.drop(columns=['Future_Close', 'Net_Return_Long', 'Net_Return_Short'], inplace=True)

        self.logger.debug("Raw labels calculated for Strategy 2 (Net Forward Return Quantile).")
        
        return pd.DataFrame({'label': df_copy['label']}, index=df_copy.index)
