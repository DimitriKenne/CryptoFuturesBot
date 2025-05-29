# utils/labeling_strategies/max_return_quantile.py

import pandas as pd
import numpy as np
import logging # Added logging import
from typing import Dict, Any, List, Optional
from .base_strategy import BaseLabelingStrategy, logger, FLOAT_EPSILON

class MaxReturnQuantileStrategy(BaseLabelingStrategy):
    """
    Labels data based on extreme returns (max positive or max negative)
    within a lookahead window, compared against quantile thresholds.

    - 1: Max positive return > quantile_threshold_pct
    - -1: Max negative return > quantile_threshold_pct (absolute value)
    - 0: No threshold breach or filtered noise
    """

    def __init__(self, config: Dict[str, Any], logger: logging.Logger):
        """
        Initializes the MaxReturnQuantileStrategy.

        Config parameters:
        - 'quantile_forward_window_bars': Lookahead window size for calculating max returns (int).
        - 'quantile_threshold_pct': The percentile threshold (float, e.g., 90.0 for 90th percentile).
                                    This threshold is applied to the *absolute* distribution
                                    of returns to determine significance.
        """
        super().__init__(config, logger)
        self.logger.info("MaxReturnQuantileStrategy initializing...")
        self._validate_strategy_config()

        self.quantile_forward_window_bars = self.config['quantile_forward_window_bars']
        self.quantile_threshold_pct = self.config['quantile_threshold_pct']

        self.logger.info(f"  Quantile Forward Window: {self.quantile_forward_window_bars} bars")
        self.logger.info(f"  Quantile Threshold: {self.quantile_threshold_pct}th percentile")


    def _validate_strategy_config(self):
        """
        Validates configuration parameters specific to the Max Return Quantile strategy.
        """
        required_keys = ['quantile_forward_window_bars', 'quantile_threshold_pct']
        for key in required_keys:
            if key not in self.config:
                raise KeyError(f"Missing required configuration key for MaxReturnQuantileStrategy: '{key}'")

        if not isinstance(self.config['quantile_forward_window_bars'], int) or self.config['quantile_forward_window_bars'] <= 0:
            raise ValueError("'quantile_forward_window_bars' must be a positive integer.")
        if not isinstance(self.config['quantile_threshold_pct'], (int, float)) or not (0 < self.config['quantile_threshold_pct'] < 100):
            raise ValueError("'quantile_threshold_pct' must be a number between 0 and 100 (exclusive).")

        self.logger.debug("MaxReturnQuantileStrategy config validated.")


    def calculate_raw_labels(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculates raw labels for the Max Return Quantile strategy.

        Args:
            df (pd.DataFrame): DataFrame with OHLCV data, indexed by time.
                               Assumed to be cleaned (no NaNs in OHLCV) by LabelGenerator.

        Returns:
            pd.DataFrame: DataFrame with 'label' column (1, -1, or 0).
        """
        self.logger.debug("Calculating raw labels for Max Return Quantile strategy...")
        self._validate_input_df(df, ['close'])

        df_copy = df.copy() # Work on a copy

        n = len(df_copy)
        labels = np.zeros(n, dtype=int) # Use numpy array for efficiency

        # Calculate future returns for each bar within its forward window
        # This can be done efficiently using rolling window operations or vectorized shifts
        # For each bar i, we need to find the max/min return in df_copy.iloc[i+1 : i + self.quantile_forward_window_bars + 1]

        # Calculate future returns for all possible horizons within the max window
        # This creates a DataFrame where each column is the return for a specific future bar
        # from the current close.
        returns_matrix = pd.DataFrame(index=df_copy.index)
        for k in range(1, self.quantile_forward_window_bars + 1):
            future_close = df_copy['close'].shift(-k)
            # Handle division by zero for the current close price
            safe_current_close = df_copy['close'].replace(0, np.nan)
            returns_matrix[f'return_h{k}'] = ((future_close - df_copy['close']) / safe_current_close) * 100.0

        # Calculate the maximum favorable excursion (MFE) and maximum adverse excursion (MAE)
        # for each row over its respective future window.
        # This requires iterating or using a custom rolling apply.
        # A more efficient way is to compute rolling max/min over the *shifted* returns matrix.

        # Let's re-approach this with a loop for clarity, though it's less performant for very large data.
        # For very large data, a more advanced vectorized approach or numba might be considered.

        # Calculate max_return and min_return for each row's future window
        max_returns = pd.Series(np.nan, index=df_copy.index)
        min_returns = pd.Series(np.nan, index=df_copy.index)

        for i in range(n):
            window_end_iloc = min(i + self.quantile_forward_window_bars + 1, n)
            # Slice the close prices for the future window (from i+1 to window_end_iloc-1)
            future_closes = df_copy['close'].iloc[i + 1 : window_end_iloc]

            if future_closes.empty:
                continue

            current_close = df_copy['close'].iloc[i]
            if pd.isna(current_close) or current_close <= FLOAT_EPSILON:
                continue # Cannot calculate returns from invalid current close

            # Calculate returns from current_close to all future_closes in the window
            returns_in_window = ((future_closes - current_close) / current_close) * 100.0

            if not returns_in_window.empty:
                max_returns.iloc[i] = returns_in_window.max()
                min_returns.iloc[i] = returns_in_window.min()

        df_copy['max_return'] = max_returns
        df_copy['min_return'] = min_returns

        # Calculate the positive and negative thresholds based on the overall distribution
        # of absolute returns. We need to consider both positive and negative extremes.
        # For the quantile, we consider the absolute values of all returns.
        all_returns_for_threshold = pd.concat([df_copy['max_return'].dropna().abs(), df_copy['min_return'].dropna().abs()])
        all_returns_for_threshold = all_returns_for_threshold.replace([np.inf, -np.inf], np.nan).dropna()

        if all_returns_for_threshold.empty:
            self.logger.warning("No valid returns to calculate quantile thresholds. All labels will be 0.")
            # CRITICAL FIX: Ensure the returned DataFrame has the original DatetimeIndex
            # and only the 'label' column. This prevents index corruption.
            return pd.DataFrame(index=df_copy.index, data={'label': 0})

        # Calculate the threshold value (e.g., 90th percentile of absolute returns)
        # Use a small epsilon to ensure we don't get a zero threshold from very small movements
        threshold_value = np.percentile(all_returns_for_threshold, self.quantile_threshold_pct)
        if threshold_value < FLOAT_EPSILON: # Ensure threshold is not effectively zero
            threshold_value = FLOAT_EPSILON
            self.logger.warning(f"Calculated quantile threshold is near zero ({threshold_value}). Setting to FLOAT_EPSILON.")


        # Apply labeling logic
        # Label 1 (Long): Max positive return is above the threshold
        df_copy.loc[df_copy['max_return'] >= threshold_value, 'label'] = 1

        # Label -1 (Short): Max negative return (absolute value) is above the threshold
        # (i.e., min_return is below -threshold_value)
        df_copy.loc[df_copy['min_return'] <= -threshold_value, 'label'] = -1

        # Drop temporary columns
        df_copy.drop(columns=['max_return', 'min_return'], inplace=True)

        self.logger.debug("Raw labels calculated for Max Return Quantile strategy.")
        
        # CRITICAL FIX: Ensure the returned DataFrame has the original DatetimeIndex
        # and only the 'label' column. This prevents index corruption.
        return pd.DataFrame({'label': df_copy['label']}, index=df_copy.index)
