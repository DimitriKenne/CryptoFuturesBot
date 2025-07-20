# utils/labeling_strategies/future_range_dominance.py

import pandas as pd
import numpy as np
import logging
from typing import Dict, Any, List, Optional
from .base_strategy import BaseLabelingStrategy, logger, FLOAT_EPSILON

class FutureRangeDominanceStrategy(BaseLabelingStrategy):
    """
    Labels data based on the dominance of future maximum or minimum net returns
    within a forward window, accounting for fees and slippage.

    - 1: Upward movement is dominantly stronger than downward movement.
    - -1: Downward movement is dominantly stronger than upward movement.
    - 0: Otherwise (Neutral).

    Parameters:
    - 'f_window_range': Forward lookahead window (int).
    - 'fee_range': Transaction fee (float, e.g., 0.0005).
    - 'slippage_range': Estimated slippage (float, e.g., 0.0001 for 0.01%).
    - 'long_ratio_quantile_pct': Percentile for the long dominance ratio threshold (float, e.g., 90).
    - 'short_ratio_quantile_pct': Percentile for the short dominance ratio threshold (float, e.g., 90).
    """

    def __init__(self, config: Dict[str, Any], logger: logging.Logger):
        """
        Initializes the FutureRangeDominanceStrategy.
        """
        super().__init__(config, logger)
        self.logger.info("FutureRangeDominanceStrategy initializing...")
        self._validate_strategy_config()

        self.f_window_range = self.config['f_window_range']
        self.fee_range = self.config['fee_range']
        self.slippage_range = self.config['slippage_range']
        self.long_ratio_quantile_pct = self.config['long_ratio_quantile_pct']
        self.short_ratio_quantile_pct = self.config['short_ratio_quantile_pct']
        # Removed min_profit_threshold_pct
        # self.min_profit_threshold_pct = self.config['min_profit_threshold_pct'] / 100.0 # Convert to fraction

        self.logger.info(f"  Forward Window (f_window_range): {self.f_window_range} bars")
        self.logger.info(f"  Transaction Fee (range): {self.fee_range}")
        self.logger.info(f"  Slippage (range): {self.slippage_range}")
        self.logger.info(f"  Long Ratio Quantile Percentile: {self.long_ratio_quantile_pct}")
        self.logger.info(f"  Short Ratio Quantile Percentile: {self.short_ratio_quantile_pct}")
        # Removed min_profit_threshold_pct from log
        # self.logger.info(f"  Minimum Profit Threshold: {self.min_profit_threshold_pct*100:.2f}%")


    def _validate_strategy_config(self):
        """
        Validates configuration parameters specific to the Future Range Dominance strategy.
        """
        required_keys = [
            'f_window_range', 'fee_range', 'slippage_range',
            'long_ratio_quantile_pct', 'short_ratio_quantile_pct'
            # Removed min_profit_threshold_pct
            # 'min_profit_threshold_pct'
        ]
        for key in required_keys:
            if key not in self.config:
                raise KeyError(f"Missing required configuration key for FutureRangeDominanceStrategy: '{key}'")

        if not isinstance(self.config['f_window_range'], int) or self.config['f_window_range'] <= 0:
            raise ValueError("'f_window_range' must be a positive integer.")
        if not isinstance(self.config['fee_range'], (int, float)) or self.config['fee_range'] < 0:
            raise ValueError("'fee_range' must be a non-negative number.")
        if not isinstance(self.config['slippage_range'], (int, float)) or self.config['slippage_range'] < 0:
            raise ValueError("'slippage_range' must be a non-negative number.")
        if not isinstance(self.config['long_ratio_quantile_pct'], (int, float)) or not (0 < self.config['long_ratio_quantile_pct'] < 100):
            raise ValueError("'long_ratio_quantile_pct' must be a number between 0 and 100 (exclusive).")
        if not isinstance(self.config['short_ratio_quantile_pct'], (int, float)) or not (0 < self.config['short_ratio_quantile_pct'] < 100):
            raise ValueError("'short_ratio_quantile_pct' must be a number between 0 and 100 (exclusive).")
        # Removed min_profit_threshold_pct validation
        # if not isinstance(self.config['min_profit_threshold_pct'], (int, float)) or self.config['min_profit_threshold_pct'] < 0:
        #     raise ValueError("'min_profit_threshold_pct' must be a non-negative number.")

        self.logger.debug("FutureRangeDominanceStrategy config validated.")


    def calculate_raw_labels(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculates raw labels for the Future Range Dominance strategy.

        Args:
            df (pd.DataFrame): DataFrame with OHLCV data, indexed by time.
                               Assumed to be cleaned (no NaNs in OHLCV) by LabelGenerator.

        Returns:
            pd.DataFrame: DataFrame with 'label' column (1, -1, or 0).
        """
        self.logger.debug("Calculating raw labels for Future Range Dominance strategy.")
        self._validate_input_df(df, ['open', 'high', 'low', 'close'])

        df_copy = df.copy() # Work on a copy

        # Compute future max high and min low within f_window_range
        # Shift by -self.f_window_range + 1 to align the window correctly:
        # max/min of current bar up to f_window_range bars into the future.
        # .shift(-(self.f_window_range - 1)) would be the start of the window
        # .shift(-self.f_window_range) would be the end of the window
        # We want the max/min over the window [current_bar_idx + 1, current_bar_idx + f_window_range]
        # This requires a forward-looking rolling window. Pandas' rolling doesn't directly support this
        # without reversing the DataFrame.
        # A common way is to shift the future prices to the current row, then take max/min.

        # Create future price series for max/min calculation
        future_highs = df_copy['high'].iloc[::-1].rolling(window=self.f_window_range).max().iloc[::-1].shift(1)
        future_lows = df_copy['low'].iloc[::-1].rolling(window=self.f_window_range).min().iloc[::-1].shift(1)
        
        df_copy['Future_Max_High'] = future_highs
        df_copy['Future_Min_Low'] = future_lows

        # Calculate net returns for max/min moves from current close
        # These factors account for fees and slippage on both entry and exit
        # For long entry: current_close * (1 + fee + slippage)
        # For long exit: future_price * (1 - fee - slippage)
        # For short entry: current_close * (1 - fee - slippage)
        # For short exit: future_price * (1 + fee + slippage)

        entry_cost_long_factor = (1 + self.fee_range + self.slippage_range)
        exit_revenue_long_factor = (1 - self.fee_range - self.slippage_range)

        entry_revenue_short_factor = (1 - self.fee_range - self.slippage_range)
        exit_cost_short_factor = (1 + self.fee_range + self.slippage_range)
        
        safe_current_close = df_copy['close'].replace(0, np.nan)

        # Potential Net Profit if going long from current close to Future_Max_High
        df_copy['Net_Return_Long_Potential'] = (df_copy['Future_Max_High'] * exit_revenue_long_factor - safe_current_close * entry_cost_long_factor) / (safe_current_close * entry_cost_long_factor)

        # Potential Net Profit if going short from current close to Future_Min_Low
        # This is (Entry_Value - Exit_Value) / Entry_Value
        df_copy['Net_Return_Short_Potential'] = (safe_current_close * entry_revenue_short_factor - df_copy['Future_Min_Low'] * exit_cost_short_factor) / (safe_current_close * entry_revenue_short_factor)


        # Drop NaN values (caused by shifting and rolling)
        initial_rows = len(df_copy)
        df_copy.dropna(subset=['Future_Max_High', 'Future_Min_Low', 'Net_Return_Long_Potential', 'Net_Return_Short_Potential'], inplace=True)
        if len(df_copy) < initial_rows:
            self.logger.warning(f"Dropped {initial_rows - len(df_copy)} rows with NaNs after calculating future range potentials.")
        if df_copy.empty:
            self.logger.error("DataFrame is empty after dropping NaNs. Cannot generate labels.")
            return pd.DataFrame(index=df_copy.index, data={'label': 0})

        # Removed min_profit_threshold_pct filtering
        # df_copy['Net_Return_Long_Potential_Filtered'] = df_copy['Net_Return_Long_Potential'].apply(lambda x: x if x > self.min_profit_threshold_pct else np.nan)
        # df_copy['Net_Return_Short_Potential_Filtered'] = df_copy['Net_Return_Short_Potential'].apply(lambda x: x if x > self.min_profit_threshold_pct else np.nan)

        # Calculate dominance ratios based on potentials (now directly using Net_Return_X_Potential)
        # Ratio = (Dominant Potential) / (Opposing Potential Magnitude + epsilon)
        # If opposing potential is not filtered (i.e., it's a loss or below min_profit_threshold), use its absolute value.
        # If both are filtered out, ratio is NaN.

        # Upward dominance ratio (only if Net_Return_Long_Potential is positive)
        df_copy['Long_Dominance_Ratio'] = np.where(
            df_copy['Net_Return_Long_Potential'] > FLOAT_EPSILON, # Only if long potential is positive
            df_copy['Net_Return_Long_Potential'] / (np.abs(df_copy['Net_Return_Short_Potential']) + FLOAT_EPSILON),
            np.nan
        )

        # Downward dominance ratio (only if Net_Return_Short_Potential is positive)
        df_copy['Short_Dominance_Ratio'] = np.where(
            df_copy['Net_Return_Short_Potential'] > FLOAT_EPSILON, # Only if short potential is positive
            df_copy['Net_Return_Short_Potential'] / (np.abs(df_copy['Net_Return_Long_Potential']) + FLOAT_EPSILON),
            np.nan
        )

        # Calculate quantiles for dominance ratios separately
        long_ratios_dropna = df_copy['Long_Dominance_Ratio'].dropna()
        short_ratios_dropna = df_copy['Short_Dominance_Ratio'].dropna()

        long_ratio_threshold = 0.0
        if not long_ratios_dropna.empty:
            long_ratio_threshold = np.percentile(long_ratios_dropna, self.long_ratio_quantile_pct)
            long_ratio_threshold = max(long_ratio_threshold, 1.0 + FLOAT_EPSILON) # Must be > 1 to show dominance
        else:
            self.logger.warning("No valid long dominance ratios found for quantile calculation. Long signals will be 0.")

        short_ratio_threshold = 0.0
        if not short_ratios_dropna.empty:
            short_ratio_threshold = np.percentile(short_ratios_dropna, self.short_ratio_quantile_pct)
            short_ratio_threshold = max(short_ratio_threshold, 1.0 + FLOAT_EPSILON) # Must be > 1 to show dominance
        else:
            self.logger.warning("No valid short dominance ratios found for quantile calculation. Short signals will be 0.")

        self.logger.debug(f"Calculated Long Ratio Threshold: {long_ratio_threshold}, Short Ratio Threshold: {short_ratio_threshold}")

        # Assign Labels
        df_copy['label'] = 0 # Default to neutral

        # Condition for Long (1):
        # 1. Long potential is positive
        # 2. Long potential is greater than short potential
        # 3. Long dominance ratio meets its quantile threshold
        long_condition = (
            (df_copy['Net_Return_Long_Potential'] > FLOAT_EPSILON) & # Long potential must be positive
            (df_copy['Net_Return_Long_Potential'] > df_copy['Net_Return_Short_Potential']) & # Long potential must be greater than short potential
            (df_copy['Long_Dominance_Ratio'] >= long_ratio_threshold)
        )
        df_copy.loc[long_condition, 'label'] = 1

        # Condition for Short (-1):
        # 1. Short potential is positive
        # 2. Short potential is greater than long potential
        # 3. Short dominance ratio meets its quantile threshold
        short_condition = (
            (df_copy['Net_Return_Short_Potential'] > FLOAT_EPSILON) & # Short potential must be positive
            (df_copy['Net_Return_Short_Potential'] > df_copy['Net_Return_Long_Potential']) & # Short potential must be greater than long potential
            (df_copy['Short_Dominance_Ratio'] >= short_ratio_threshold)
        )
        df_copy.loc[short_condition, 'label'] = -1


        # Drop temporary columns
        df_copy.drop(columns=[
            'Future_Max_High', 'Future_Min_Low',
            'Net_Return_Long_Potential', 'Net_Return_Short_Potential',
            # Removed filtered columns
            # 'Net_Return_Long_Potential_Filtered', 'Net_Return_Short_Potential_Filtered',
            'Long_Dominance_Ratio', 'Short_Dominance_Ratio'
        ], inplace=True)

        self.logger.debug("Raw labels calculated for Future Range Dominance strategy.")
        
        return pd.DataFrame({'label': df_copy['label']}, index=df_copy.index)
