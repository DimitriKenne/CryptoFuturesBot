# utils/labeling_strategies/net_forward_return_quantile.py

import pandas as pd
import numpy as np
import logging
from typing import Dict, Any, List, Optional
from .base_strategy import BaseLabelingStrategy, logger, FLOAT_EPSILON

class NetForwardReturnQuantileStrategy(BaseLabelingStrategy):
    """
    Labels data based on future net returns (accounting for fees and slippage)
    relative to quantile thresholds.

    - 1: Future Net Return >= positive_threshold (Buy)
    - -1: Future Net Return <= negative_threshold (Sell)
    - 0: Otherwise (Neutral)

    Parameters:
    - 'f_window': Forward lookahead window for future close (int).
    - 'fee': Transaction fee (float, e.g., 0.0005).
    - 'slippage': Estimated slippage (float, e.g., 0.0001 for 0.01%).
    - 'buy_quantile_pct': Percentile for positive return threshold (float, e.g., 90 for 90th percentile).
    - 'sell_quantile_pct': Percentile for negative return threshold (float, e.g., 10 for 10th percentile).
    """

    def __init__(self, config: Dict[str, Any], logger: logging.Logger):
        """
        Initializes the NetForwardReturnQuantileStrategy.
        """
        super().__init__(config, logger)
        self.logger.info("NetForwardReturnQuantileStrategy initializing...")
        self._validate_strategy_config()

        self.f_window = self.config['f_window']
        self.fee = self.config['fee']
        self.slippage = self.config['slippage']
        self.buy_quantile_pct = self.config['buy_quantile_pct']
        self.sell_quantile_pct = self.config['sell_quantile_pct']

        self.logger.info(f"  Forward Window (f_window): {self.f_window} bars")
        self.logger.info(f"  Transaction Fee: {self.fee}")
        self.logger.info(f"  Slippage: {self.slippage}")
        self.logger.info(f"  Buy Quantile Percentile: {self.buy_quantile_pct}")
        self.logger.info(f"  Sell Quantile Percentile: {self.sell_quantile_pct}")


    def _validate_strategy_config(self):
        """
        Validates configuration parameters specific to the Net Forward Return Quantile strategy.
        """
        required_keys = ['f_window', 'fee', 'slippage', 'buy_quantile_pct', 'sell_quantile_pct']
        for key in required_keys:
            if key not in self.config:
                raise KeyError(f"Missing required configuration key for NetForwardReturnQuantileStrategy: '{key}'")

        if not isinstance(self.config['f_window'], int) or self.config['f_window'] <= 0:
            raise ValueError("'f_window' must be a positive integer.")
        if not isinstance(self.config['fee'], (int, float)) or self.config['fee'] < 0:
            raise ValueError("'fee' must be a non-negative number.")
        if not isinstance(self.config['slippage'], (int, float)) or self.config['slippage'] < 0:
            raise ValueError("'slippage' must be a non-negative number.")
        if not isinstance(self.config['buy_quantile_pct'], (int, float)) or not (0 < self.config['buy_quantile_pct'] < 100):
            raise ValueError("'buy_quantile_pct' must be a number between 0 and 100 (exclusive).")
        if not isinstance(self.config['sell_quantile_pct'], (int, float)) or not (0 < self.config['sell_quantile_pct'] < 100):
            raise ValueError("'sell_quantile_pct' must be a number between 0 and 100 (exclusive).")

        self.logger.debug("NetForwardReturnQuantileStrategy config validated.")


    def calculate_raw_labels(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculates raw labels for the Net Forward Return Quantile strategy.

        Args:
            df (pd.DataFrame): DataFrame with OHLCV data, indexed by time.
                               Assumed to be cleaned (no NaNs in OHLCV) by LabelGenerator.

        Returns:
            pd.DataFrame: DataFrame with 'label' column (1, -1, or 0).
        """
        self.logger.debug("Calculating raw labels for Net Forward Return Quantile strategy.")
        self._validate_input_df(df, ['close'])

        df_copy = df.copy() # Work on a copy

        # Compute future close
        df_copy['Future_Close'] = df_copy['close'].shift(-self.f_window)

        # Calculate Net Return (accounting for fees and slippage)
        # Entry cost factor: (1 + fee + slippage) for long, (1 - fee - slippage) for short
        # Exit cost factor: (1 - fee - slippage) for long, (1 + fee + slippage) for short
        # We calculate a generic 'potential' net return as if going long, then use its sign.
        # The true net return for short would be inverted.

        # For simplicity in labeling, we'll calculate a 'long-biased' net return.
        # The quantile thresholds will then be applied symmetrically or asymmetrically.
        # This return represents: (Future_Close - Current_Close - 2 * (fee + slippage) * Current_Close) / Current_Close
        # More precisely, relative to a long entry:
        entry_cost_factor = (1 + self.fee + self.slippage)
        exit_revenue_factor = (1 - self.fee - self.slippage)

        # Calculate potential net return assuming a long position
        # (Future_Close * exit_revenue_factor - Current_Close * entry_cost_factor) / (Current_Close * entry_cost_factor)
        safe_current_close_long = df_copy['close'].replace(0, np.nan) * entry_cost_factor
        df_copy['Net_Return_Long'] = (df_copy['Future_Close'] * exit_revenue_factor - df_copy['close'] * entry_cost_factor) / safe_current_close_long

        # Calculate potential net return assuming a short position
        # (Current_Close * exit_revenue_factor - Future_Close * entry_cost_factor) / (Current_Close * exit_revenue_factor)
        safe_current_close_short = df_copy['close'].replace(0, np.nan) * exit_revenue_factor
        df_copy['Net_Return_Short'] = (df_copy['close'] * exit_revenue_factor - df_copy['Future_Close'] * entry_cost_factor) / safe_current_close_short


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
        negative_returns = df_copy['Net_Return_Short'][df_copy['Net_Return_Short'] > 0].dropna() # Short return is positive if price drops

        buy_threshold = 0.0 # Default to no signal if no positive returns
        if not positive_returns.empty:
            buy_threshold = np.percentile(positive_returns, self.buy_quantile_pct)
            # Ensure buy_threshold is at least slightly positive to overcome fees if quantile is very low
            buy_threshold = max(buy_threshold, FLOAT_EPSILON)
        else:
            self.logger.warning("No positive net returns found for buy quantile calculation. Buy signals will be 0.")

        sell_threshold = 0.0 # Default to no signal if no negative returns
        if not negative_returns.empty:
            sell_threshold = np.percentile(negative_returns, self.sell_quantile_pct)
            # Ensure sell_threshold is at least slightly positive to overcome fees if quantile is very low
            sell_threshold = max(sell_threshold, FLOAT_EPSILON)
        else:
            self.logger.warning("No positive net returns found for sell quantile calculation. Sell signals will be 0.")


        self.logger.debug(f"Calculated Buy Threshold: {buy_threshold}, Sell Threshold: {sell_threshold}")

        # Assign Labels
        df_copy['label'] = 0 # Default to neutral

        # Buy (1): Net Return for long position is above threshold
        df_copy.loc[df_copy['Net_Return_Long'] >= buy_threshold, 'label'] = 1

        # Sell (-1): Net Return for short position is above threshold (meaning price dropped enough)
        df_copy.loc[df_copy['Net_Return_Short'] >= sell_threshold, 'label'] = -1

        # Drop temporary columns
        df_copy.drop(columns=['Future_Close', 'Net_Return_Long', 'Net_Return_Short'], inplace=True)

        self.logger.debug("Raw labels calculated for Net Forward Return Quantile strategy.")
        
        return pd.DataFrame({'label': df_copy['label']}, index=df_copy.index)
