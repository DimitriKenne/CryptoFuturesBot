# utils/labeling_strategies/swing_pivot.py

import pandas as pd
import numpy as np
import logging
from typing import Dict, Any, List, Optional
from .base_strategy import BaseLabelingStrategy, logger, FLOAT_EPSILON

class SwingPivotStrategy(BaseLabelingStrategy):
    """
    Labels data based on the detection of Pine Script-style swing highs and swing lows.

    - 1: Swing Low detected (potential buy signal)
    - -1: Swing High detected (potential sell signal)
    - 0: Otherwise (neutral)

    A swing high is a high that has 'left_bars' lower highs before it and 'right_bars' lower highs after it.
    A swing low is a low that has 'left_bars' higher lows before it and 'right_bars' higher lows after it.

    The label is applied at the bar where the pivot is confirmed (i.e., `right_bars` after the pivot point itself)
    to avoid lookahead bias.

    Parameters:
    - 'left_bars': Number of bars to the left to consider for pivot detection (int).
    - 'right_bars': Number of bars to the right to consider for pivot confirmation (int).
    """

    def __init__(self, config: Dict[str, Any], logger: logging.Logger):
        """
        Initializes the SwingPivotStrategy.
        """
        super().__init__(config, logger)
        self.logger.info("SwingPivotStrategy initializing...")
        self._validate_strategy_config()

        self.left_bars = self.config['left_bars']
        self.right_bars = self.config['right_bars']

        self.logger.info(f"  Left Bars: {self.left_bars}")
        self.logger.info(f"  Right Bars: {self.right_bars}")

    def _validate_strategy_config(self):
        """
        Validates configuration parameters specific to the Swing Pivot strategy.
        """
        required_keys = ['left_bars', 'right_bars']
        for key in required_keys:
            if key not in self.config:
                raise KeyError(f"Missing required configuration key for SwingPivotStrategy: '{key}'")

        if not isinstance(self.config['left_bars'], int) or self.config['left_bars'] <= 0:
            raise ValueError("'left_bars' must be a positive integer.")
        if not isinstance(self.config['right_bars'], int) or self.config['right_bars'] <= 0:
            raise ValueError("'right_bars' must be a positive integer.")

        self.logger.debug("SwingPivotStrategy config validated.")

    def calculate_raw_labels(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculates raw labels for the Swing Pivot strategy.

        Args:
            df (pd.DataFrame): DataFrame with OHLCV data, indexed by time.
                               Assumed to be cleaned (no NaNs in OHLCV) by LabelGenerator.

        Returns:
            pd.DataFrame: DataFrame with 'label' column (1, -1, or 0).
        """
        self.logger.debug("Calculating raw labels for Swing Pivot strategy.")
        self._validate_input_df(df, ['high', 'low']) # Need high and low for pivots

        df_copy = df.copy() # Work on a copy
        n = len(df_copy)

        # Initialize label column with 0 (neutral)
        labels = pd.Series(0, index=df_copy.index, dtype=np.int8)

        # Iterate through the DataFrame to find pivots
        # The pivot is confirmed after 'right_bars' have passed.
        # So, the loop needs to go up to `n - self.right_bars`.
        # The pivot point itself is at index `i`.
        # The label for this pivot is applied at `i + self.right_bars`.
        for i in range(self.left_bars, n - self.right_bars):
            # Check for Pivot High: current high is highest in the window [i-left_bars, i+right_bars]
            # And values to the left/right are strictly lower to ensure a clear peak
            is_pivot_high = True
            current_high = df_copy['high'].iloc[i]

            # Check left side
            for j in range(1, self.left_bars + 1):
                if df_copy['high'].iloc[i - j] >= current_high:
                    is_pivot_high = False
                    break
            
            # Check right side (only if left side is good)
            if is_pivot_high:
                for j in range(1, self.right_bars + 1):
                    if df_copy['high'].iloc[i + j] >= current_high:
                        is_pivot_high = False
                        break
            
            if is_pivot_high:
                # Assign -1 (sell signal) at the confirmation bar
                # The confirmation bar is 'right_bars' after the pivot point 'i'
                labels.iloc[i + self.right_bars] = -1
                self.logger.debug(f"Swing High detected at index {i} (value: {current_high}), label -1 assigned at {i + self.right_bars}")


            # Check for Pivot Low: current low is lowest in the window [i-left_bars, i+right_bars]
            # And values to the left/right are strictly higher to ensure a clear trough
            is_pivot_low = True
            current_low = df_copy['low'].iloc[i]

            # Check left side
            for j in range(1, self.left_bars + 1):
                if df_copy['low'].iloc[i - j] <= current_low:
                    is_pivot_low = False
                    break

            # Check right side (only if left side is good)
            if is_pivot_low:
                for j in range(1, self.right_bars + 1):
                    if df_copy['low'].iloc[i + j] <= current_low:
                        is_pivot_low = False
                        break

            if is_pivot_low:
                # Assign 1 (buy signal) at the confirmation bar
                # The confirmation bar is 'right_bars' after the pivot point 'i'
                labels.iloc[i + self.right_bars] = 1
                self.logger.debug(f"Swing Low detected at index {i} (value: {current_low}), label 1 assigned at {i + self.right_bars}")

        self.logger.debug("Raw labels calculated for Swing Pivot strategy.")
        
        # Return only the 'label' column with the original index
        return pd.DataFrame({'label': labels}, index=df_copy.index)

