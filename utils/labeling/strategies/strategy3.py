# utils/labeling_strategies/strategy3.py

import pandas as pd
import numpy as np
import logging
from typing import Any, List, Optional
from .base_strategy import BaseLabelingStrategy, logger, FLOAT_EPSILON

# Import the specific config dataclass for this labeling strategy
from config.label import LabelingStrategy3Config

class Strategy3(BaseLabelingStrategy):
    """
    Labeling Strategy 3: Future Range Dominance.

    This labeling strategy labels data based on the relative strength (dominance) of
    potential future upward movement versus potential future downward movement
    within a defined forward lookahead window. It considers net returns,
    accounting for fees and slippage.

    - A label of '1' (Buy) is assigned if the potential net profit from an upward move
      is positive and significantly greater than the potential net profit (or loss)
      from a downward move, and its "long dominance ratio" meets a specified quantile threshold.
    - A label of '-1' (Sell) is assigned if the potential net profit from a downward move
      is positive and significantly greater than the potential net profit (or loss)
      from an upward move, and its "short dominance ratio" meets a specified quantile threshold.
    - A label of '0' (Neutral) is assigned otherwise.
    """

    def __init__(self, config: LabelingStrategy3Config, logger: logging.Logger, trading_fee_rate: float, slippage_tolerance_pct: float):
        """
        Initializes Labeling Strategy 3 (Future Range Dominance).

        Args:
            config (LabelingStrategy3Config): The configuration dataclass for this labeling strategy.
            logger (logging.Logger): A logger instance.
            trading_fee_rate (float): The transaction fee rate.
            slippage_tolerance_pct (float): The estimated slippage rate.
        """
        super().__init__(config, logger, trading_fee_rate, slippage_tolerance_pct)
        self.logger.info("Labeling Strategy 3 (Future Range Dominance) initializing...")
        self._validate_strategy_config()

        # Access parameters directly from the LabelingStrategy3Config dataclass
        self.future_return_window = self.config.future_return_window
        self.long_ratio_quantile_pct = self.config.long_ratio_quantile_pct
        self.short_ratio_quantile_pct = self.config.short_ratio_quantile_pct
        self.min_profit_threshold = self.config.min_profit_threshold

        self.logger.info(f"  Forward Window (future_return_window): {self.future_return_window} bars")
        self.logger.info(f"  Transaction Fee: {self.trading_fee_rate}")
        self.logger.info(f"  Slippage: {self.slippage_tolerance_pct}")
        self.logger.info(f"  Long Ratio Quantile Percentile: {self.long_ratio_quantile_pct}")
        self.logger.info(f"  Short Ratio Quantile Percentile: {self.short_ratio_quantile_pct}")
        self.logger.info(f"  Minimum Profit Threshold: {self.min_profit_threshold}")

    def _validate_strategy_config(self):
        """
        Validates configuration parameters specific to Labeling Strategy 3.
        """
        if not isinstance(self.config.future_return_window, int) or self.config.future_return_window <= 0:
            raise ValueError("'future_return_window' must be a positive integer.")
        if not isinstance(self.config.long_ratio_quantile_pct, (int, float)) or not (0 < self.config.long_ratio_quantile_pct < 100):
            raise ValueError("'long_ratio_quantile_pct' must be a number between 0 and 100 (exclusive).")
        if not isinstance(self.config.short_ratio_quantile_pct, (int, float)) or not (0 < self.config.short_ratio_quantile_pct < 100):
            raise ValueError("'short_ratio_quantile_pct' must be a number between 0 and 100 (exclusive).")
        if not isinstance(self.config.min_profit_threshold, (int, float)) or self.config.min_profit_threshold < 0:
            raise ValueError("'min_profit_threshold' must be a non-negative number.")
        self.logger.debug("Labeling Strategy 3 config validated.")

    def calculate_raw_labels(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculates raw labels for Labeling Strategy 3 (Future Range Dominance).

        Args:
            df (pd.DataFrame): DataFrame with OHLCV data, indexed by time.
                               Assumed to be cleaned (no NaNs in OHLCV) by LabelGenerator.

        Returns:
            pd.DataFrame: DataFrame with 'label' column (1, -1, or 0).
        """
        self.logger.debug("Calculating raw labels for Labeling Strategy 3 (Future Range Dominance).")
        self._validate_input_df(df, ['open', 'high', 'low', 'close'])

        df_copy = df.copy()

        # Compute future max high and min low within future_return_window
        future_highs = df_copy['high'].iloc[::-1].rolling(window=self.future_return_window).max().iloc[::-1].shift(1)
        future_lows = df_copy['low'].iloc[::-1].rolling(window=self.future_return_window).min().iloc[::-1].shift(1)
        
        df_copy['Future_Max_High'] = future_highs
        df_copy['Future_Min_Low'] = future_lows

        # Calculate net returns for max/min moves from current close
        entry_cost_long_factor = (1 + self.trading_fee_rate + self.slippage_tolerance_pct)
        exit_revenue_long_factor = (1 - self.trading_fee_rate - self.slippage_tolerance_pct)

        entry_revenue_short_factor = (1 - self.trading_fee_rate - self.slippage_tolerance_pct)
        exit_cost_short_factor = (1 + self.trading_fee_rate + self.slippage_tolerance_pct)
        
        safe_current_close = df_copy['close'].replace(0, np.nan)

        # Potential Net Profit if going long from current close to Future_Max_High
        df_copy['Net_Return_Long_Potential'] = (df_copy['Future_Max_High'] * exit_revenue_long_factor - safe_current_close * entry_cost_long_factor) / (safe_current_close * entry_cost_long_factor)

        # Potential Net Profit if going short from current close to Future_Min_Low
        df_copy['Net_Return_Short_Potential'] = (safe_current_close * entry_revenue_short_factor - df_copy['Future_Min_Low'] * exit_cost_short_factor) / (safe_current_close * entry_revenue_short_factor)

        # Drop NaN values (caused by shifting and rolling)
        initial_rows = len(df_copy)
        df_copy.dropna(subset=['Future_Max_High', 'Future_Min_Low', 'Net_Return_Long_Potential', 'Net_Return_Short_Potential'], inplace=True)
        if len(df_copy) < initial_rows:
            self.logger.warning(f"Dropped {initial_rows - len(df_copy)} rows with NaNs after calculating future range potentials.")
        if df_copy.empty:
            self.logger.error("DataFrame is empty after dropping NaNs. Cannot generate labels.")
            return pd.DataFrame(index=df_copy.index, data={'label': 0})

        # Calculate dominance ratios based on potentials
        df_copy['Long_Dominance_Ratio'] = np.where(
            df_copy['Net_Return_Long_Potential'] > self.min_profit_threshold,
            df_copy['Net_Return_Long_Potential'] / (np.abs(df_copy['Net_Return_Short_Potential']) + FLOAT_EPSILON),
            np.nan
        )

        df_copy['Short_Dominance_Ratio'] = np.where(
            df_copy['Net_Return_Short_Potential'] > self.min_profit_threshold,
            df_copy['Net_Return_Short_Potential'] / (np.abs(df_copy['Net_Return_Long_Potential']) + FLOAT_EPSILON),
            np.nan
        )

        # Calculate quantiles for dominance ratios separately
        long_ratios_dropna = df_copy['Long_Dominance_Ratio'].dropna()
        short_ratios_dropna = df_copy['Short_Dominance_Ratio'].dropna()

        long_ratio_threshold = 0.0
        if not long_ratios_dropna.empty:
            long_ratio_threshold = np.percentile(long_ratios_dropna, self.long_ratio_quantile_pct)
            long_ratio_threshold = max(long_ratio_threshold, 1.0 + FLOAT_EPSILON)
        else:
            self.logger.warning("No valid long dominance ratios found for quantile calculation. Long signals will be 0.")

        short_ratio_threshold = 0.0
        if not short_ratios_dropna.empty:
            short_ratio_threshold = np.percentile(short_ratios_dropna, self.short_ratio_quantile_pct)
            short_ratio_threshold = max(short_ratio_threshold, 1.0 + FLOAT_EPSILON)
        else:
            self.logger.warning("No valid short dominance ratios found for quantile calculation. Short signals will be 0.")

        self.logger.debug(f"Calculated Long Ratio Threshold: {long_ratio_threshold}, Short Ratio Threshold: {short_ratio_threshold}")

        # Assign Labels
        df_copy['label'] = 0

        # Condition for Long (1)
        long_condition = (
            (df_copy['Net_Return_Long_Potential'] > self.min_profit_threshold) &
            (df_copy['Net_Return_Long_Potential'] > df_copy['Net_Return_Short_Potential']) &
            (df_copy['Long_Dominance_Ratio'] >= long_ratio_threshold)
        )
        df_copy.loc[long_condition, 'label'] = 1

        # Condition for Short (-1)
        short_condition = (
            (df_copy['Net_Return_Short_Potential'] > self.min_profit_threshold) &
            (df_copy['Net_Return_Short_Potential'] > df_copy['Net_Return_Long_Potential']) &
            (df_copy['Short_Dominance_Ratio'] >= short_ratio_threshold)
        )
        df_copy.loc[short_condition, 'label'] = -1

        # Drop temporary columns
        df_copy.drop(columns=[
            'Future_Max_High', 'Future_Min_Low',
            'Net_Return_Long_Potential', 'Net_Return_Short_Potential',
            'Long_Dominance_Ratio', 'Short_Dominance_Ratio'
        ], inplace=True)

        self.logger.debug("Raw labels calculated for Labeling Strategy 3 (Future Range Dominance).")
        
        return pd.DataFrame({'label': df_copy['label']}, index=df_copy.index)
