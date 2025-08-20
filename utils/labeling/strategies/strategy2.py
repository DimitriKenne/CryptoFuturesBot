# utils/labeling_strategies/strategy2.py

import pandas as pd
import numpy as np
import logging
from typing import Any, List, Optional
from pathlib import Path # Import Path for type hinting
from .base_strategy import BaseLabelingStrategy, logger, FLOAT_EPSILON

# Import the specific config dataclass for this labeling strategy
from config.label import LabelingStrategy2Config

class Strategy2(BaseLabelingStrategy):
    """
    Labeling Strategy 2: Net Forward Return Quantile.

    This labeling strategy labels data based on future net returns (accounting for fees and slippage)
    relative to dynamically calculated quantile thresholds.

    - A label of '1' (Buy) is assigned if the future net return for a long position
      is greater than or equal to a specified positive quantile threshold.
    - A label of '-1' (Sell) is assigned if the future net return for a short position
      is greater than or equal to a specified negative quantile threshold (meaning price
      dropped sufficiently to yield a positive short return).
    - A label of '0' (Neutral) is assigned otherwise.
    """

    def __init__(self, config: LabelingStrategy2Config, logger: logging.Logger, trading_fee_rate: float, slippage_tolerance_rate: float):
        """
        Initializes Labeling Strategy 2 (Net Forward Return Quantile).

        Args:
            config (LabelingStrategy2Config): The configuration dataclass for this labeling strategy.
            logger (logging.Logger): A logger instance.
            trading_fee_rate (float): The transaction fee rate (0-1).
            slippage_tolerance_rate (float): The estimated slippage rate (0-1).
        """
        super().__init__(config, logger, trading_fee_rate, slippage_tolerance_rate)
        self.logger.info("Labeling Strategy 2 (Net Forward Return Quantile) initializing...")
        self._validate_strategy_config()

        # Access parameters directly from the LabelingStrategy2Config dataclass
        self.future_return_window = self.config.future_return_window
        # Quantile thresholds are percentages (0-100) from config
        self.quantile_threshold_long_pct = self.config.quantile_threshold_long_pct
        self.quantile_threshold_short_pct = self.config.quantile_threshold_short_pct
        self.return_type = self.config.return_type

        self.logger.info(f"  Forward Window (future_return_window): {self.future_return_window} bars")
        self.logger.info(f"  Transaction Fee Rate: {self.trading_fee_rate:.6f}")
        self.logger.info(f"  Slippage: {self.slippage_tolerance_rate:.6f}")
        self.logger.info(f"  Buy Quantile Percentile (Long): {self.quantile_threshold_long_pct:.2f}%")
        self.logger.info(f"  Sell Quantile Percentile (Short): {self.quantile_threshold_short_pct:.2f}%")
        self.logger.info(f"  Return Type: {self.return_type}")

    def _validate_strategy_config(self):
        """
        Validates configuration parameters specific to Labeling Strategy 2,
        now accessing directly from self.config (LabelingStrategy2Config).
        """
        if not isinstance(self.config.future_return_window, int) or self.config.future_return_window <= 0:
            raise ValueError("'future_return_window' must be a positive integer.")
        
        if not isinstance(self.config.quantile_threshold_long_pct, (int, float)) or not (0 <= self.config.quantile_threshold_long_pct < 100):
            raise ValueError("'quantile_threshold_long_pct' must be a number between 0 and 100 (inclusive of 0).")
        
        if not isinstance(self.config.quantile_threshold_short_pct, (int, float)) or not (0 <= self.config.quantile_threshold_short_pct < 100):
            raise ValueError("'quantile_threshold_short_pct' must be a number between 0 and 100 (inclusive of 0).")

        if self.config.return_type not in ['log_returns', 'simple_returns']:
            raise ValueError("'return_type' must be 'log_returns' or 'simple_returns'.")

        self.logger.debug("Labeling Strategy 2 config validated.")

    def calculate_raw_labels(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculates raw labels for Labeling Strategy 2 (Net Forward Return Quantile).

        Args:
            df (pd.DataFrame): DataFrame with OHLCV data, indexed by time.
                               Assumed to be cleaned (no NaNs in OHLCV) by LabelGenerator.

        Returns:
            pd.DataFrame: DataFrame with 'label' column (1, -1, or 0).
        """
        self.logger.debug("Calculating raw labels for Labeling Strategy 2 (Net Forward Return Quantile).")
        self._validate_input_df(df, ['close'])

        df_copy = df.copy() # Work on a copy to avoid modifying original df

        # Compute future close
        df_copy['Future_Close'] = df_copy['close'].shift(-self.future_return_window)

        # Calculate Net Return (accounting for fees and slippage)
        # These are calculated as RATIOS (e.g., 0.01 for 1% return)
        entry_cost_long_factor = (1 + self.trading_fee_rate + self.slippage_tolerance_rate)
        exit_revenue_long_factor = (1 - self.trading_fee_rate - self.slippage_tolerance_rate)
        safe_current_close_long = df_copy['close'].replace(0, np.nan)
        df_copy['Net_Return_Long'] = (
            (df_copy['Future_Close'] * exit_revenue_long_factor - safe_current_close_long * entry_cost_long_factor) /
            (safe_current_close_long * entry_cost_long_factor)
        )

        entry_revenue_short_factor = (1 - self.trading_fee_rate - self.slippage_tolerance_rate)
        exit_cost_short_factor = (1 + self.trading_fee_rate + self.slippage_tolerance_rate)
        safe_current_close_short_entry = df_copy['close'].replace(0, np.nan)
        df_copy['Net_Return_Short'] = (
            (safe_current_close_short_entry * entry_revenue_short_factor - df_copy['Future_Close'] * exit_cost_short_factor) /
            (safe_current_close_short_entry * entry_revenue_short_factor)
        )

        # Apply log returns if specified (conversion from simple return)
        if self.return_type == 'log_returns':
            df_copy['Net_Return_Long'] = np.log(1 + df_copy['Net_Return_Long'] + FLOAT_EPSILON)
            df_copy['Net_Return_Short'] = np.log(1 + df_copy['Net_Return_Short'] + FLOAT_EPSILON)

        # Drop NaN values (caused by shifting `Future_Close` and potentially from safe_current_close)
        initial_rows = len(df_copy)
        # Temporarily store the DataFrame with intermediate columns for later analysis
        self._intermediate_analysis_df = df_copy.copy() # This copy still holds RATIOS

        df_copy.dropna(subset=['Future_Close', 'Net_Return_Long', 'Net_Return_Short'], inplace=True)
        if len(df_copy) < initial_rows:
            self.logger.warning(f"Dropped {initial_rows - len(df_copy)} rows with NaNs after calculating future net returns.")
        if df_copy.empty:
            self.logger.error("DataFrame is empty after dropping NaNs. Cannot generate labels.")
            return pd.DataFrame(index=df.index, data={'label': 0}) # Return a DataFrame of zeros with original index


        # Calculate quantiles for positive net returns (0-1 range, still ratios)
        positive_returns = df_copy['Net_Return_Long'][df_copy['Net_Return_Long'] > 0].dropna()
        profitable_short_returns = df_copy['Net_Return_Short'][df_copy['Net_Return_Short'] > 0].dropna()

        # Buy Threshold (long positions)
        buy_threshold = 0.0 # Default to no signal if no profitable returns
        if not positive_returns.empty:
            buy_threshold = np.percentile(positive_returns, self.quantile_threshold_long_pct)
            buy_threshold = max(buy_threshold, FLOAT_EPSILON) # Ensure threshold is not negative or zero
        else:
            self.logger.warning("No positive net returns found for buy quantile calculation. Buy signals will be 0.")

        # Sell Threshold (short positions)
        sell_threshold = 0.0 # Default to no signal if no profitable short returns
        if not profitable_short_returns.empty:
            sell_threshold = np.percentile(profitable_short_returns, self.quantile_threshold_short_pct)
            sell_threshold = max(sell_threshold, FLOAT_EPSILON) # Ensure threshold is not negative or zero
        else:
            self.logger.warning("No profitable short net returns found for sell quantile calculation. Sell signals will be 0.")

        self.logger.debug(f"Calculated Buy Threshold (Rate): {buy_threshold}, Sell Threshold (Rate): {sell_threshold}")

        # Assign Labels
        df_copy['label'] = 0 # Default to neutral

        # Buy (1): Net Return for long position is above or equal to threshold
        df_copy.loc[df_copy['Net_Return_Long'] >= buy_threshold, 'label'] = 1

        # Sell (-1): Net Return for short position is above or equal to threshold
        df_copy.loc[df_copy['Net_Return_Short'] >= sell_threshold, 'label'] = -1

        # Drop temporary columns before returning the final labels-only DataFrame
        df_copy.drop(columns=['Future_Close', 'Net_Return_Long', 'Net_Return_Short'], inplace=True)

        self.logger.debug("Raw labels calculated for Labeling Strategy 2 (Net Forward Return Quantile).")
        
        return pd.DataFrame({'label': df_copy['label']}, index=df_copy.index)

    def perform_strategy_specific_analysis(
        self,
        df_original_input: pd.DataFrame,
        symbol: str,
        interval: str,
        output_dir: Path,
        plotter: Any, # AnalysisPlotter instance
        calculator: Any # AnalysisCalculator instance
    ) -> None:
        """
        Performs analysis specific to Labeling Strategy 2 (plotting Net Return distributions).

        Args:
            df_original_input (pd.DataFrame): The original input DataFrame (for context).
            symbol (str): Trading pair symbol.
            interval (str): Time interval.
            output_dir (Path): Directory to save analysis results.
            plotter (Any): AnalysisPlotter instance.
            calculator (Any): AnalysisCalculator instance.
        """
        self.logger.info(f"Performing strategy-specific analysis for Strategy 2 ({symbol.upper()} {interval}): Net Return Distributions...")

        if self._intermediate_analysis_df is None or self._intermediate_analysis_df.empty:
            self.logger.warning("No intermediate analysis data found. Skipping.")
            return

        # Crucial fix: Convert ratio returns to percentages for plotting
        df_net_returns_for_plot = self._intermediate_analysis_df[['Net_Return_Long', 'Net_Return_Short']].dropna().copy()
        df_net_returns_for_plot['Net_Return_Long'] *= 100.0
        df_net_returns_for_plot['Net_Return_Short'] *= 100.0

        if df_net_returns_for_plot.empty:
            self.logger.warning("No valid net returns for plotting after conversion. Skipping plot.")
            return

        # Call the plotter to generate and save the distribution plots
        plotter.plot_net_return_distributions(df_net_returns_for_plot, symbol, interval, output_dir)
        self.logger.info(f"Strategy-specific analysis (Net Return Distributions) completed and saved for {symbol.upper()} {interval}.")
