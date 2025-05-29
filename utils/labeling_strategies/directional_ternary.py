# utils/labeling_strategies/directional_ternary.py

import pandas as pd
import numpy as np
import logging # Added logging import
from typing import Dict, Any, List, Optional
from .base_strategy import BaseLabelingStrategy, logger, FLOAT_EPSILON

class DirectionalTernaryStrategy(BaseLabelingStrategy):
    """
    Labels data based on fixed future window price movement.
    - 1: Price increased by price_threshold_pct within forward_window_bars.
    - -1: Price decreased by price_threshold_pct within forward_window_bars.
    - 0: No significant move or noise-filtered.
    """

    def __init__(self, config: Dict[str, Any], logger: logging.Logger):
        """
        Initializes the DirectionalTernaryStrategy.

        Config parameters:
        - 'forward_window_bars': Lookahead window size (int).
        - 'price_threshold_pct': Required percentage move (float, e.g., 2.0 for 2%).
        """
        super().__init__(config, logger)
        self.logger.info("DirectionalTernaryStrategy initializing...")
        self._validate_strategy_config()

        self.forward_window_bars = self.config['forward_window_bars']
        self.price_threshold_pct = self.config['price_threshold_pct']
        # min_holding_period is handled by LabelGenerator for propagation, not by raw labeling.

        self.logger.info(f"  Forward Window: {self.forward_window_bars} bars")
        self.logger.info(f"  Price Threshold: {self.price_threshold_pct}%")


    def _validate_strategy_config(self):
        """
        Validates configuration parameters specific to the Directional Ternary strategy.
        """
        required_keys = ['forward_window_bars', 'price_threshold_pct']
        for key in required_keys:
            if key not in self.config:
                raise KeyError(f"Missing required configuration key for DirectionalTernaryStrategy: '{key}'")

        if not isinstance(self.config['forward_window_bars'], int) or self.config['forward_window_bars'] <= 0:
            raise ValueError("'forward_window_bars' must be a positive integer.")
        if not isinstance(self.config['price_threshold_pct'], (int, float)) or self.config['price_threshold_pct'] <= 0:
            raise ValueError("'price_threshold_pct' must be a positive number.")

        self.logger.debug("DirectionalTernaryStrategy config validated.")


    def calculate_raw_labels(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculates raw labels for the Directional Ternary strategy.

        Args:
            df (pd.DataFrame): DataFrame with OHLCV data, indexed by time.
                               Assumed to be cleaned (no NaNs in OHLCV) by LabelGenerator.

        Returns:
            pd.DataFrame: DataFrame with 'label' column (1, -1, or 0).
        """
        self.logger.debug("Calculating raw labels for Directional Ternary strategy...")
        # Validate input DataFrame for essential columns
        self._validate_input_df(df, ['close'])

        df_copy = df.copy() # Work on a copy to avoid modifying original

        # Calculate future price within the forward window
        # We look at the close price 'forward_window_bars' bars ahead
        df_copy['future_close'] = df_copy['close'].shift(-self.forward_window_bars)

        # Calculate percentage change from current close to future close
        # Handle division by zero for the current close price
        safe_current_close = df_copy['close'].replace(0, np.nan) # Replace 0 with NaN to avoid division by zero
        df_copy['price_change_pct'] = ((df_copy['future_close'] - df_copy['close']) / safe_current_close) * 100.0

        # Initialize labels to 0 (neutral)
        df_copy['label'] = 0

        # Apply labeling logic
        # Label 1 (Long): Price increased by threshold
        df_copy.loc[df_copy['price_change_pct'] >= self.price_threshold_pct, 'label'] = 1

        # Label -1 (Short): Price decreased by threshold
        df_copy.loc[df_copy['price_change_pct'] <= -self.price_threshold_pct, 'label'] = -1

        # Drop temporary columns
        df_copy.drop(columns=['future_close', 'price_change_pct'], inplace=True)

        self.logger.debug("Raw labels calculated for Directional Ternary strategy.")
        
        # CRITICAL FIX: Ensure the returned DataFrame has the original DatetimeIndex
        # and only the 'label' column. This prevents index corruption.
        return pd.DataFrame({'label': df_copy['label']}, index=df_copy.index)
